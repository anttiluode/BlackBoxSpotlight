"""Interrogate a callable tokens -> vector. Token IDs, maximum lag and alphabet are public."""
import numpy as np


class Unknown(ValueError):
    """Measurements cannot identify this family within the declared tolerance."""


def recover(observe, count, max_lag):
    if count < 3 or max_lag < 2:
        raise ValueError('need at least three tokens and two lags')
    cache = {}

    def ask(ids):
        key = tuple(ids)
        if key not in cache:
            value = np.asarray(observe(key), dtype=float)
            if value.ndim != 1 or not value.size or not np.isfinite(value).all():
                raise Unknown('Expected a finite output vector')
            if cache and value.shape != next(iter(cache.values())).shape:
                raise Unknown('Output dimension changed between probes')
            cache[key] = value.copy()
        return cache[key]

    values = np.stack([ask([k]) for k in range(count)])
    scale = max(np.max(np.linalg.norm(values-values.mean(0), axis=1)), 1e-12)
    worst = 0.
    # score[q,k,d] is log weight relative to final q's weight. Lag zero is a gauge.
    score = np.zeros((count, count, max_lag+1))

    def measure(ids, origin, axes):
        nonlocal worst
        delta = ask(ids)-values[origin]
        coeff, _, rank, singular = np.linalg.lstsq(axes, delta, rcond=1e-9)
        if rank != axes.shape[1] or singular[-1] < 1e-8*scale:
            raise Unknown(f'Indistinguishable source values after {len(cache)} probes')
        residue = np.linalg.norm(axes @ coeff-delta)/scale
        worst = max(residue, worst)
        if residue > 1e-6:
            raise Unknown(f'Output outside measured value span ({residue:.3g} relative residual, {len(cache)} probes)')
        return coeff

    def odds(p, base):
        if min(p, base) < 1e-10 or max(p, base) > 1+1e-10:
            raise Unknown(f'Unresolved source mass after {len(cache)} probes')
        return np.log(p/base)

    for q in range(count):
        for k in range(count):
            if k == q:
                continue
            edge = values[k]-values[q]
            ee = edge @ edge
            if ee < (1e-8*scale)**2:
                raise Unknown('Two singleton outputs coincide')
            # The instrument chooses a filler whose value is far from the k--q line.
            other = values-values[q]
            separation = np.linalg.norm(other-np.outer(other @ edge/ee, edge), axis=1)
            f = int(np.argmax(separation))
            if separation[f] < 1e-8*scale:
                raise Unknown('No third token resolves the mixture')
            axes = np.stack([edge, other[f]], axis=1)
            p, = measure([k,q], q, axes[:, :1])
            score[q,k,1] = odds(p, 1-p)
            for d in range(2, max_lag+1):
                pk, pf = measure([k]+[f]*(d-1)+[q], q, axes)
                score[q,k,d] = odds(pk, 1-pk-pf)

        # Repeated q is invisible as a separate value. Insert a calibrated marker.
        # Sum marker odds at all its positions and use the measured total marker mass.
        # Several markers are available. Choose one near 50% under unit unknown odds.
        for d in range(1, max_lag+1):
            length = max(3, d+1)
            lags = [lag for lag in range(1, length) if lag != d]
            totals = np.exp(score[q][:, lags]).sum(axis=1)
            candidates = [k for k in range(count) if k != q]
            k = min(candidates, key=lambda k: abs(np.log(totals[k]/2)))
            ids = [k]*length
            ids[-1] = ids[-1-d] = q
            pm, = measure(ids, q, (values[k]-values[q])[:, None])
            mass = totals[k]*np.exp(odds(1-pm, pm))-1
            if mass < 1e-10:
                raise Unknown(f'Repeated-token odds unresolved after {len(cache)} probes')
            score[q,q,d] = np.log(mass)
    return values, score, {'queries': len(cache), 'max_relative_span_error': worst}


def routing(score, ids):
    ids = np.asarray(ids)
    log_odds = score[ids[-1], ids, np.arange(len(ids)-1, -1, -1)]
    w = np.exp(log_odds-log_odds.max())
    return w/w.sum()


def fit_recurrence(seq, order):
    windows = np.lib.stride_tricks.sliding_window_view(seq, order+1, axis=-1).reshape(-1, order+1)
    x = np.column_stack([windows[:, :-1], np.ones(len(windows))])
    return np.linalg.lstsq(x, windows[:, -1], rcond=1e-8)[0]


def extend(score, order, end):
    """One shared affine recurrence for all token-pair log-odds. No frequency input."""
    seq = score[..., 1:]
    if order == 0:
        return np.concatenate([score, np.repeat(score[..., -1:], end-seq.shape[-1], axis=-1)], axis=-1)
    coef = fit_recurrence(seq, order)
    return extrapolate(score, coef, end)


def extrapolate(score, coef, end):
    order = len(coef)-1
    pred = np.zeros((*score.shape[:2], end+1))
    pred[..., :score.shape[-1]] = score
    for d in range(score.shape[-1], end+1):
        pred[..., d] = pred[..., d-order:d] @ coef[:-1] + coef[-1]
    return pred


def forecast(score, end):
    # Reserve the four most distant measured offsets for model selection, then refit.
    n = score.shape[-1]-1
    losses = []
    for order in range(0, min(10, n-6)+1):
        pred = extend(score[..., :n-3], order, n)
        losses.append(float(np.sqrt(np.mean((pred[..., -4:]-score[..., -4:])**2))))
    order = int(np.argmin(losses))
    coef = fit_recurrence(score[..., 1:], order) if order else np.array([])
    roots = np.roots(np.r_[1., -coef[-2::-1]]) if order else []
    modes = sorted(({'radians_per_token': float(np.angle(r)), 'radius': float(abs(r))}
                    for r in roots if r.imag > 1e-8), key=lambda m: -m['radians_per_token'])
    return extend(score, order, end), {'order': order, 'validation_log_odds_rmse': losses,
                                       'coefficients_with_intercept': coef.tolist(), 'modes': modes}
