"""성공 시범 수를 최대 K개로 자른 새 hdf5 생성 (demo 번호 순 앞 K개)."""
import sys, h5py
src, dst, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
with h5py.File(src, 'r') as fi, h5py.File(dst, 'w') as fo:
    for k, v in fi.attrs.items():
        fo.attrs[k] = v
    for name in fi:
        if name != 'data':
            fi.copy(fi[name], fo, name)
    Di, Do = fi['data'], fo.create_group('data')
    for k, v in Di.attrs.items():
        Do.attrs[k] = v
    keys = sorted((k for k in Di if bool(Di[k].attrs.get('success', 1))), key=lambda k: int(k.split('_')[1]))[:K]
    total = 0
    for i, k in enumerate(keys):
        fi.copy(Di[k], Do, f'demo_{i}'); total += int(Di[k].attrs['num_samples'])
    Do.attrs['total'] = total
    print(f'{len(Di)} -> {len(keys)} demos, total {total}')
