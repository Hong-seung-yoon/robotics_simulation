"""연속 0-action(멈춤) 구간을 최대 N step으로 줄인 새 hdf5를 만든다. 원본은 건드리지 않음.

멈춤 step = 팔 6축 action이 모두 0이고 그리퍼 명령이 직전 step과 같은 step.
길이 > N인 연속 멈춤 구간은 앞 N step만 남기고, 시간축 데이터(actions/obs/states/...)를 같은 인덱스로 자른다.
usage: python trim_idle.py IN.hdf5 OUT.hdf5 [--max_idle 10]
"""
import argparse
import h5py, numpy as np

p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('dst')
p.add_argument('--max_idle', type=int, default=10)
args = p.parse_args()
assert args.src != args.dst

def keep_mask(a, N):
    idle = (a[:, :6] == 0).all(1)
    idle[1:] &= a[1:, 6] == a[:-1, 6]
    keep = np.ones(len(a), bool); run = 0
    for t in range(len(a)):
        run = run + 1 if idle[t] else 0
        if run > N:
            keep[t] = False
    return keep

def copy_sliced(src_g, dst_g, idx, T):
    for k, v in src_g.attrs.items():
        dst_g.attrs[k] = v
    for name, obj in src_g.items():
        if isinstance(obj, h5py.Group):
            if name == 'initial_state':
                src_g.copy(obj, dst_g, name)
            else:
                copy_sliced(obj, dst_g.create_group(name), idx, T)
        elif obj.shape and obj.shape[0] == T:
            d = dst_g.create_dataset(name, data=obj[:][idx], compression=obj.compression)
            for k, v in obj.attrs.items():
                d.attrs[k] = v
        else:
            src_g.copy(obj, dst_g, name)

with h5py.File(args.src, 'r') as fi, h5py.File(args.dst, 'w') as fo:
    for k, v in fi.attrs.items():
        fo.attrs[k] = v
    for name in fi:
        if name != 'data':
            fi.copy(fi[name], fo, name)
    Di, Do = fi['data'], fo.create_group('data')
    for k, v in Di.attrs.items():
        Do.attrs[k] = v
    total = 0
    for dk in sorted(Di, key=lambda k: int(k.split('_')[1])):
        g = Di[dk]; a = g['actions'][:]; T = len(a)
        idx = np.nonzero(keep_mask(a, args.max_idle))[0]
        go = Do.create_group(dk)
        copy_sliced(g, go, idx, T)
        go.attrs['num_samples'] = len(idx); total += len(idx)
        print(f'{dk:<8s} {T:5d} -> {len(idx):5d}')
    Do.attrs['total'] = total
    print('total', total)
