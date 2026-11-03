"""두 생성 풀을 합쳐 트럭 초기 x 구간별로 같은 수만 뽑아 성공 시범 N개를 만든다. 구간이 모자라면 있는 만큼 쓰고 나머지 구간에 고르게 채운다.
usage: python balance_truck_x.py OUT.hdf5 MAP.csv N POOL1.hdf5 POOL2.hdf5 ..."""
import sys, h5py, numpy as np, subprocess
out, mp, N, pools = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
E = np.linspace(0.45, 0.80, 6); rng = np.random.default_rng(0)
rows = []
for p in pools:
    D = h5py.File(p, 'r')['data']
    for k in D:
        if bool(D[k].attrs.get('success', 1)):
            rows.append((p, k, float(D[k]['obs']['toy_truck_positions'][0, 0])))
bins = [[r for r in rows if E[i] <= r[2] <= E[i + 1] + 1e-6 and (i == 4 or r[2] < E[i + 1])] for i in range(5)]
print('pool per bin', [len(b) for b in bins], 'total', len(rows))
take = [0] * 5; left = N
while left > 0:
    c = [i for i in range(5) if take[i] < len(bins[i])]
    if not c: break
    for i in sorted(c, key=lambda i: take[i]):
        if left and take[i] < len(bins[i]):
            take[i] += 1; left -= 1
print('take per bin', take)
sel = {}
for i in range(5):
    for j in rng.permutation(len(bins[i]))[:take[i]]:
        sel.setdefault(bins[i][j][0], []).append(bins[i][j][1])
specs = [f"{p}:{','.join(v)}" for p, v in sel.items()]
subprocess.run([sys.executable, '-I', __file__.replace('balance_truck_x.py', 'merge_with_map.py'), out, mp] + specs, check=True)
