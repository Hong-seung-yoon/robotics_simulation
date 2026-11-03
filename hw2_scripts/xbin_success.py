"""실패 분류 기록에서 트럭 초기 x 구간별 성공률을 출력. usage: python xbin_success.py DIR [DIR ...]"""
import sys, glob, numpy as np
E = np.linspace(0.45, 0.80, 6)
R = []
for t in sys.argv[1:]:
    for f in glob.glob(t + '/trial_*.npz'):
        d = np.load(f); R.append((float(d['toy_truck_positions'][0, 0]), bool(d['success'])))
R = np.array(R)
print('n', len(R), 'mean', round(R[:, 1].mean(), 3))
for i in range(5):
    m = (R[:, 0] >= E[i]) & (R[:, 0] <= E[i + 1] + 1e-6) if i == 4 else (R[:, 0] >= E[i]) & (R[:, 0] < E[i + 1])
    print(f'  x {E[i]:.2f}-{E[i+1]:.2f}: n={int(m.sum()):3d} success={R[m, 1].mean():.2f}')
m = R[:, 0] > 0.66
print(f'  x>0.66: n={int(m.sum())} success={R[m, 1].mean():.2f}; x<=0.66: n={int((~m).sum())} success={R[~m, 1].mean():.2f}')
