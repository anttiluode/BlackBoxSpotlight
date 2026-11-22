"""Bundle a working, file://-friendly bench. No remote scripts, assets or model calls."""
import json
from pathlib import Path

from experiment import WORDS

ROOT = Path(__file__).parent
weights = json.loads((ROOT/'spotlight/weights.json').read_text())
meta = weights['meta']
d = meta['d']
ids = [meta['vocab'].index(w) for w in WORDS]
bundle = {
    'words': WORDS, 'base': meta['base'], 'width': meta['dh'], 'dimension': d,
    'emb': [weights['embed'][i*d:(i+1)*d] for i in ids],
    'qkv': {k: weights['blocks'][0][k] for k in ['wq','wk','wv']},
    'laws': json.loads((ROOT/'results/recovered.json').read_text()),
    'receipt': json.loads((ROOT/'results/receipt.json').read_text())
}
template = (ROOT/'web/bench.html').read_text()
(ROOT/'index.html').write_text(template.replace('/* BUNDLE */null',json.dumps(bundle, separators=(',',':'))))
