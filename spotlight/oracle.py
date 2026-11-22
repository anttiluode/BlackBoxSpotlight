"""Frozen OneTinyTransformer, measured at the final position before head concatenation."""
import json
from pathlib import Path

import numpy as np


class Head:
    def __init__(self, words, layer=0, head=0, rotary=True):
        data = json.loads(Path(__file__).with_name('weights.json').read_text())
        meta = data['meta']
        self.head = head
        self.base = meta['base']
        self.rotary = rotary
        self.context = meta['context']
        self.width = meta['dh']
        ids = [meta['vocab'].index(w) for w in words]
        self.emb = np.array(data['embed']).reshape(-1, meta['d'])[ids]
        self.blocks = []
        for block in data['blocks'][:layer+1]:
            self.blocks.append({k: np.array(v).reshape(-1, meta['ff'] if k == 'fc2' else meta['d'])
                                for k, v in block.items()})

    def _run(self, ids):
        x = self.emb[list(ids)]
        n = len(x)
        if not 1 <= n <= self.context:
            raise ValueError('prompt outside model context')
        h = self.width
        theta = np.arange(n)[:, None, None] * self.base ** (-np.arange(0, h, 2) / h)
        if not self.rotary:
            theta = np.zeros_like(theta)
        cos, sin = np.cos(theta), np.sin(theta)

        def rotate(projected):
            pairs = projected.reshape(n, -1, h)
            a, b = pairs[..., ::2], pairs[..., 1::2]
            return np.stack([a*cos-b*sin, a*sin+b*cos], axis=-1).reshape(n, -1, h)

        for level, block in enumerate(self.blocks):
            z = norm(x)
            q = rotate(z @ block['wq'].T)
            k = rotate(z @ block['wk'].T)
            v = (z @ block['wv'].T).reshape(n, -1, h)
            if level == len(self.blocks)-1:
                weights = softmax(k[:, self.head] @ q[-1, self.head] / np.sqrt(h))
                return weights @ v[:, self.head], weights
            scores = np.einsum('ihd,jhd->hij', q, k) / np.sqrt(h)
            future_row, future_col = np.triu_indices(n, 1)
            scores[:, future_row, future_col] = -np.inf
            mix = np.einsum('hij,jhd->ihd', softmax(scores), v).reshape(n, -1)
            mid = x + mix @ block['wo'].T
            y = norm(mid) @ block['fc1'].T
            x = mid + (.5*y*(1+np.tanh(np.sqrt(2/np.pi)*(y+.044715*y**3)))) @ block['fc2'].T

    def observe(self, ids):
        return self._run(ids)[0]

    def reveal(self, ids):
        return self._run(ids)[1]


def norm(x):
    return (x-x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True)+1e-5)


def softmax(x):
    e = np.exp(x-x.max(-1, keepdims=True))
    return e/e.sum(-1, keepdims=True)
