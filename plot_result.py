"""Optional figure. Read the attention key only after loading the frozen inverse."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment import WORDS, CAL, END
from spotlight.oracle import Head

ROOT = Path(__file__).parent
law = json.loads((ROOT/'results/recovered.json').read_text())['L1H1']
score = np.array(law['table'])
head = Head(WORDS,0,0)
truth = np.zeros_like(score)
for q,k in np.ndindex(score.shape[:2]):
    for lag in range(1, END+1):
        att = head.reveal([k]+[0]*(lag-1)+[q])
        truth[q,k,lag] = np.log(att[0]/att[-1])

fig, (trace, err) = plt.subplots(1, 2, figsize=(10,3.8), constrained_layout=True,
                                gridspec_kw={'width_ratios':[1.45,1]})
d = np.arange(1, END+1)
q,k = WORDS.index('the'),WORDS.index('storm')
trace.axvspan(CAL+.5, END+.3, color='#faefdd')
trace.plot(d, truth[q,k,1:], color='#293b42', linewidth=2, label='Revealed answer')
trace.scatter(d[:CAL], score[q,k,1:CAL+1], color='#16866d', s=15, label='Measured by seeker',zorder=3)
trace.scatter(d[CAL:], score[q,k,CAL+1:], facecolors='none', edgecolors='#bd6308', s=48, label='Blind forecast',zorder=3)
trace.set(title='When “the” reads “storm”', xlabel='Token distance', ylabel='Log attention odds vs. the current word')
trace.legend(fontsize=8,loc='lower left')
for label, array, color in [('Hold last measurement',score[...,CAL:CAL+1],'#677775'),
                             ('Shared recurrence',score[...,CAL+1:],'#b15d07')]:
    rmse = np.sqrt(((array-truth[...,CAL+1:])**2).mean((0,1)))
    err.semilogy(d[CAL:],rmse,'o-',color=color,label=label,markersize=4)
err.set(title='All 64 word pairs', xlabel='Unmeasured token distance', ylabel='Log-odds RMSE (log scale)',xticks=d[CAL:])
err.legend(fontsize=8)
for ax in (trace,err):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.savefig(ROOT/'results/forecast.png',dpi=150)
