import sys
import h5py, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/toy2box_agibot.hdf5'
f = h5py.File(path, 'r')
D = f['data']
demos = sorted(D, key=lambda k: int(k.split('_')[1]))
# merge 결과 등에는 attrs['total']이 없을 수 있음 → num_samples 합으로 대체
total = D.attrs['total'] if 'total' in D.attrs else sum(int(D[k].attrs['num_samples']) for k in demos)
print('file:', path)
print('episodes:', len(demos), '| total:', total)
for k in demos:
    a = D[k]['actions'][:]
    n = D[k].attrs['num_samples']
    ok = int(D[k].attrs['success']) if 'success' in D[k].attrs else -1
    sw = int((np.diff(a[:, 6]) != 0).sum())  # 그리퍼 전환
    idle = (a[:, :6] == 0).all(1).mean()     # 멈춤 비율
    print(f'{k:<8s} len={n:4d} ({n/20:.1f}s) '
          f'sw={sw:2d} idle={idle:.0%} ok={ok}')
