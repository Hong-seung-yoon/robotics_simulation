"""필수 시범(annotate 통과분) + 후보 중 annotate 통과한 시범에서 점수 순으로 골라 총 N개 annotated 데이터셋을 만든다.
점수(우선순위): 그리퍼 전환 2회(단일 파지) > 길이 800 이하 > 짧은 길이 > 낮은 idle > 주인 s5 우선.
usage: python select100.py  (경로는 아래 상수)"""
import h5py, hashlib, csv, subprocess, sys, numpy as np
D = 'datasets/'
N = 100
MAND_SRC = [D + f'toy2box_s{i}.hdf5' for i in range(1, 5)]                   # 필수(datasets.zip)
MAND_ANN = D + 'ta20own20_raw_annotated.hdf5'                                # 그 annotate 결과(원본 annotate)
CAND_RAW, CAND_MAP, CAND_ANN = D + 'cand120_raw.hdf5', D + 'cand120_raw_map.csv', D + 'cand120_raw_annotated.hdf5'
OUT = D + 'own20_sel80_raw_annotated.hdf5'
CSV = 'logs/hw2/sel100_selection.csv'

h = lambda a: hashlib.md5(a.tobytes()).hexdigest()
def feats(a):
    return dict(len=len(a), sw=int((np.diff(a[:, 6]) != 0).sum()), idle=float((a[:, :6] == 0).all(1).mean()))

rows = []
ann_m = h5py.File(MAND_ANN, 'r')['data']; ann_m_h = {h(ann_m[k]['actions'][:]): k for k in ann_m}
for f in MAND_SRC:
    S = h5py.File(f, 'r')['data']
    for k in sorted(S, key=lambda k: int(k.split('_')[1])):
        a = S[k]['actions'][:]; ak = ann_m_h.get(h(a))
        rows.append(dict(file=f.split('/')[-1], demo=k, **feats(a), mandatory=1, ann_file=MAND_ANN, ann_key=ak))
src = {r['demo']: (r['src_file'], r['src_demo']) for r in csv.DictReader(open(CAND_MAP))}
raw = h5py.File(CAND_RAW, 'r')['data']; ann_c = h5py.File(CAND_ANN, 'r')['data']
ann_c_h = {h(ann_c[k]['actions'][:]): k for k in ann_c}
for k in sorted(raw, key=lambda k: int(k.split('_')[1])):
    a = raw[k]['actions'][:]; sf, sd = src[k]
    rows.append(dict(file=sf, demo=sd, **feats(a), mandatory=0, ann_file=CAND_ANN, ann_key=ann_c_h.get(h(a))))

for r in rows:
    r['annotate_pass'] = int(r['ann_key'] is not None)
mand = [r for r in rows if r['mandatory'] and r['annotate_pass']]
pool = [r for r in rows if not r['mandatory'] and r['annotate_pass']]
pool.sort(key=lambda r: (r['sw'] != 2, r['len'] > 800, r['len'], r['idle'], r['file'] != 'toy2box_s5.hdf5'))
need = N - len(mand)
if len(pool) < need:
    sys.exit(f'ABORT: only {len(pool)} passing candidates for {need} slots')
chosen = set(id(r) for r in mand + pool[:need])
cut = pool[need - 1]
for r in rows:
    r['chosen'] = int(id(r) in chosen)
    r['reason'] = ('mandatory (datasets.zip)' if r['mandatory'] and r['annotate_pass'] else
                   'mandatory but failed annotate' if r['mandatory'] else
                   'failed annotate' if not r['annotate_pass'] else
                   f'rank {pool.index(r)+1}/{len(pool)}' + ('' if r['chosen'] else ' (below cut)'))
with open(CSV, 'w', newline='') as fo:
    keys = ['file', 'demo', 'len', 'sw', 'idle', 'annotate_pass', 'chosen', 'reason']
    w = csv.DictWriter(fo, fieldnames=keys, extrasaction='ignore'); w.writeheader()
    for r in rows: w.writerow({**r, 'idle': f"{r['idle']:.2f}"})
print(f'mandatory pass {len(mand)}/20, candidates pass {len(pool)}/120, picked {need}; cut-off: len={cut["len"]} sw={cut["sw"]} idle={cut["idle"]:.0%}')
for f in sorted(set(r['file'] for r in rows)):
    g = [r for r in rows if r['file'] == f]
    print(f'  {f:<22s} total {len(g):2d} pass {sum(r["annotate_pass"] for r in g):2d} chosen {sum(r["chosen"] for r in g):2d}')
specs = []
for af in [MAND_ANN, CAND_ANN]:
    ks = [r['ann_key'] for r in rows if r['chosen'] and r['ann_file'] == af]
    specs.append(af + ':' + ','.join(ks))
subprocess.run([sys.executable, '-I', '/home/cai/IsaacLab_RS_hw2/hw2_scripts/merge_with_map.py', OUT, OUT.replace('.hdf5', '_map.csv')] + specs, check=True)
