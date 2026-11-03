import csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SF, INK, MUT, GRID, BLUE = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
def style(a):
    a.set_facecolor(SF); a.grid(axis='y', color=GRID, lw=.8, zorder=0); a.tick_params(colors=MUT, labelsize=10, length=0)
    for s in ('top', 'right', 'left'): a.spines[s].set_visible(False)
    a.spines['bottom'].set_color('#c9c8c2')
# 1. 트럭 x 구간별
bins = ['0.45–0.52', '0.52–0.59', '0.59–0.66', '0.66–0.73', '0.73–0.80']
P = [('Policy success rate (%), model ②', [65, 55, 66, 36, 35], 'Success (%)'), ('Successful Mimic demos kept for training', [238, 249, 222, 192, 99], 'Demos'), ('Failed Mimic generation attempts (discarded)', [279, 272, 280, 334, 461], 'Attempts')]
fig, ax = plt.subplots(1, 3, figsize=(13, 4), facecolor=SF)
for a, (t, v, y) in zip(ax, P):
    style(a); b = a.bar(bins, v, width=.72, color=BLUE, zorder=3); a.bar_label(b, padding=3, fontsize=10, color=INK)
    a.set_title(t, fontsize=11.5, color=INK, loc='left', pad=12); a.set_ylabel(y, fontsize=10, color=MUT); a.set_xlabel('Truck initial x (m)', fontsize=10, color=MUT); a.tick_params(axis='x', labelsize=9); a.set_ylim(0, max(v) * 1.15)
fig.suptitle('Mimic keeps only successes, so the hardest truck positions are under-represented', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig('logs/hw2/figs/truck_x_bias.png', dpi=170); plt.close(fig)
# 2. 접근 기울기
A = list(csv.DictReader(open('logs/hw2/approach_tilt.csv')))
g = ['TA 20', 'Own s1-s4', 'Own s5', 'team01_kim', 'team06', 'kgh', 'team13', 'Selected 100', 'Mimic 1000']
lab = dict(zip(g, ['TA demos', 'Own s1–s4', 'Own s5', 'Teammate: kim', 'Teammate: team06', 'Teammate: kgh', 'Teammate: team13', 'Selected 100 (source)', 'Mimic 1000 (generated)']))
fig, ax = plt.subplots(3, 3, figsize=(11, 7.6), sharex=True, facecolor=SF); e = np.arange(0, 95, 5)
for a, n in zip(ax.flat, g):
    v = np.array([float(r['ang_z']) for r in A if r['group'] == n]); h, _ = np.histogram(v, e); style(a)
    a.bar(e[:-1] + 2.5, h, width=4.2, color=BLUE, zorder=3); a.axvline(25, color=MUT, lw=1, ls='--', zorder=2)
    a.set_title(f'{lab[n]}\nn={len(v)}, median {np.median(v):.0f}°', fontsize=10.5, color=INK, loc='left')
for a in ax[-1]: a.set_xlabel('Gripper tilt from vertical at grasp (°)', fontsize=10, color=MUT)
for a in ax[:, 0]: a.set_ylabel('Demos', fontsize=10, color=MUT)
ax[0, 0].text(26.5, ax[0, 0].get_ylim()[1] * .9, '25° split', fontsize=9, color=MUT)
fig.suptitle('Approach posture differs by demonstrator: clusters split near 25°', x=.01, ha='left', fontsize=14, color=INK)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig('logs/hw2/approach_tilt_hist.png', dpi=170); fig.savefig('logs/hw2/figs/approach_tilt_hist.png', dpi=170); plt.close(fig)
# 3. 실패 단계 (seed 101, 100 rollouts)
C = ['A: TA 20 + Mimic 192', 'E: best 100 + Mimic 1000', '②: E + RNN memory 50']
S = [('Approach failure', [36, 27, 15], '#2a78d6'), ('Grasp failure', [17, 9, 11], '#eb6834'), ('Transport failure', [22, 25, 14], '#1baf7a'), ('Place failure', [2, 6, 5], '#eda100'), ('Success', [23, 33, 55], '#b9b8b2')]
fig, a = plt.subplots(figsize=(10, 3.6), facecolor=SF); a.set_facecolor(SF); left = np.zeros(3)
for n, v, c in S:
    bars = a.barh(C, v, left=left, color=c, edgecolor=SF, linewidth=2, label=n, height=.62, zorder=3)
    for b, x in zip(bars, v):
        if x >= 5: a.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, str(x), ha='center', va='center', fontsize=10, color=INK)
    left += np.array(v)
a.invert_yaxis(); a.set_xlim(0, 100); a.set_xlabel('Rollouts out of 100 (training seed 101, eval seed 42)', fontsize=10, color=MUT); a.tick_params(colors=MUT, labelsize=10, length=0); a.tick_params(axis='y', colors=INK)
for s in ('top', 'right', 'left'): a.spines[s].set_visible(False)
a.spines['bottom'].set_color('#c9c8c2'); a.legend(ncol=5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), fontsize=9.5, labelcolor=MUT, handlelength=1, columnspacing=1.2)
a.set_title('Where rollouts fail', fontsize=14, color=INK, loc='left', pad=34)
fig.tight_layout(); fig.savefig('logs/hw2/figs/failure_stages.png', dpi=170); plt.close(fig)
