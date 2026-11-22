"""Rebuild the measured receipt and the model used by the offline bench."""
import json
from pathlib import Path

import numpy as np

from spotlight.inverse import Unknown, extend, extrapolate, forecast, recover, routing
from spotlight.oracle import Head

ROOT = Path(__file__).parent
WORDS = 'the storm moves across dark sea moon .'.split()
CAL = 16
END = 23
TV_GATE = .01
OUTPUT_GATE = .01


def prompts(seed, n, length):
    rng = np.random.default_rng(seed)
    pool = set()
    while len(pool) < n:
        ids = tuple(map(int, rng.integers(len(WORDS), size=length)))
        if len(set(ids)) >= 4:
            pool.add(ids)
    return sorted(pool)


def candidates(table):
    obs = table[..., 1:]
    def pack(arr):
        t = np.zeros_like(table)
        t[..., 1:] = arr
        return t
    return {'uniform': np.zeros_like(table), 'lag': pack(obs.mean((0,1))),
            'token pair': pack(obs.mean(-1, keepdims=True)),
            'pair + query-specific lag': pack(obs.mean(-1, keepdims=True) +
                                             obs.mean(1, keepdims=True)-obs.mean((1,2), keepdims=True)),
            'pair × lag': table}


def measure_outputs(head, values, ids, rules):
    actual = np.stack([head.observe(row) for row in ids])
    scale = np.mean((actual-values.mean(0))**2)
    pred = {}
    for name, rule in rules.items():
        weights = np.stack([routing(rule, row) for row in ids])
        replay = np.einsum('bi,bij->bj', weights, values[np.array(ids)])
        error = np.mean((actual-replay)**2)
        pred[name] = {'output_nrmse': float(np.sqrt(error/max(scale, 1e-20))),
                      'weights': weights, 'replay': replay}
    return pred, actual


def audit(head, ids, pred):
    truth = np.stack([head.reveal(row) for row in ids])
    results = {}
    for name, p in pred.items():
        est = p['weights']
        results[name] = {'output_nrmse': p['output_nrmse'],
                         'mean_tv': float(.5*np.abs(est-truth).sum(1).mean()),
                         'distant_log_odds_rmse': None}
        if len(ids[0]) > CAL+1:
            distant = slice(0, len(ids[0])-CAL-1)
            diff = np.log(est[:, distant]/est[:, -1:])-np.log(truth[:, distant]/truth[:, -1:])
            results[name]['distant_log_odds_rmse'] = float(np.sqrt(np.mean(diff**2)))
    return results


def main():
    result = {'vocabulary': WORDS, 'measured_max_lag': CAL, 'forecast_max_lag': END,
              'validation_seed': 102, 'test_seeds': [301,302,303],
              'subjects': {}, 'gates': {'validation_output_nrmse': OUTPUT_GATE, 'audit_mean_tv': TV_GATE}}
    viewer = {}
    result['position_encoding_key'] = (100.**(-np.arange(0, 12, 2)/12)).tolist()
    for layer, h in [(0,0),(0,1),(1,0),(1,1)]:
        head = Head(WORDS, layer, h)
        key = f'L{layer+1}H{h+1}'
        try:
            values, score, info = recover(head.observe, len(WORDS), CAL)
        except Unknown as exc:
            result['subjects'][key] = {'status': 'unresolved', 'reason': str(exc)}
            continue
        rules = candidates(score)
        val_ids = prompts(102, 128, 12)
        val, _ = measure_outputs(head, values, val_ids, rules)
        # Model selection sees only vector errors. The weights below are a sealed audit.
        chosen = next((k for k,p in val.items() if p['output_nrmse'] <= OUTPUT_GATE), None)
        extrap, fit = forecast(score, END)
        tests = {}
        for seed in result['test_seeds']:
            near = prompts(seed, 128, 12)
            far = prompts(seed+10, 128, 24)
            assert set(near).isdisjoint(val_ids)
            pred, _ = measure_outputs(head, values, near, rules)
            long, _ = measure_outputs(head, values, far, {'hold last lag': extend(score,0,END),
                                                          'shared recurrence': extrap})
            # Intervention: rotate the source word at position 2, preserve final query.
            changed = [row[:2]+((row[2]+1)%len(WORDS),)+row[3:] for row in far]
            cf, y1 = measure_outputs(head, values, changed, {'shared recurrence': extrap})
            y0 = np.stack([head.observe(row) for row in far])
            delta = cf['shared recurrence']['replay']-long['shared recurrence']['replay']
            actual = y1-y0
            tests[str(seed)] = {'within': audit(head, near, pred), 'beyond': audit(head, far, long),
                                'edit_relative_rmse': float(np.linalg.norm(delta-actual)/np.linalg.norm(actual))}
        result['subjects'][key] = {'status': 'selected' if chosen else 'unresolved', 'selected': chosen,
                                   'validation_nrmse': {k:p['output_nrmse'] for k,p in val.items()},
                                   'probe': info, 'forecast': fit, 'test': tests}
        if chosen and layer == 0:
            passed = all(s['within'][chosen]['mean_tv'] <= TV_GATE for s in tests.values())
            result['subjects'][key]['routing_audit_pass'] = passed
            checked = result['subjects'][key]
            checked['controls'] = {}
            for dtype in ['float32', 'float16']:
                try:
                    v, t, _ = recover(lambda ids: head.observe(ids).astype(dtype), len(WORDS), CAL)
                    projected, detail = forecast(t, END)
                    rows = prompts(311, 128, 24)
                    outputs, _ = measure_outputs(head, v, rows, {'forecast': projected})
                    checked['controls'][dtype] = {'order': detail['order'], **audit(head, rows, outputs)['forecast']}
                except Unknown as exc:
                    checked['controls'][dtype] = {'status': 'unresolved', 'reason': str(exc)}
            # Intervention on the known box, after blind selection and audit.
            ablated = Head(WORDS, layer, h, rotary=False)
            av, at, _ = recover(ablated.observe, len(WORDS), CAL)
            ap, _ = measure_outputs(ablated, av, val_ids, candidates(at))
            checked['controls']['rotations_disabled'] = {
                'selected': next((k for k,p in ap.items() if p['output_nrmse'] <= OUTPUT_GATE), None),
                'validation_nrmse': {k:p['output_nrmse'] for k,p in ap.items()}}
            viewer[key] = {'values': values.tolist(), 'table': extrap.tolist()}
    # Transfer only the recurrence. New head and disjoint words get their own short probes.
    new_words = 'train river night wind rain light sky shore'.split()
    target = Head(new_words, 0, 1)
    vec, short, counts = recover(target.observe, len(new_words), CAL)
    coef = np.array(result['subjects']['L1H1']['forecast']['coefficients_with_intercept'])
    transfer_rules = {'hold last lag': extend(short, 0, END),
                      'borrowed recurrence': extrapolate(short, coef, END)}
    transfer = {}
    for seed in result['test_seeds']:
        rows = prompts(seed+10, 128, 24)
        pred, _ = measure_outputs(target, vec, rows, transfer_rules)
        transfer[str(seed)] = audit(target, rows, pred)
    result['transfer'] = {'source': 'L1H1', 'target': 'L1H2', 'target_words': new_words,
                          'target_calibration': counts, 'test': transfer}
    dest = ROOT/'results'
    dest.mkdir(exist_ok=True)
    for filename, data in [('receipt', result), ('recovered', viewer)]:
        (dest/f'{filename}.json').write_text(json.dumps(data, indent=2 if filename=='receipt' else None,
                                                        allow_nan=False)+'\n')
    for key, obj in result['subjects'].items():
        print(key, obj.get('selected', obj.get('reason')), obj.get('probe',''))
        if 'test' in obj:
            print(obj['validation_nrmse'], obj['forecast'])
            print(obj['test']['301']['beyond'], obj['test']['301']['edit_relative_rmse'])


if __name__ == '__main__':
    main()
