"""여러 hdf5를 demo 번호 순서대로 합치고 (새 demo -> 원본 파일:demo) 매핑 csv를 남긴다.
usage: python merge_with_map.py OUT.hdf5 MAP.csv IN1.hdf5 [IN2.hdf5 ...]   (IN은 FILE 또는 FILE:demo_a,demo_b 형식)"""
import sys, csv, os, h5py
out, mp, ins = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []; total = 0
with h5py.File(out, 'w') as fo:
    Do = fo.create_group('data'); i = 0
    for spec in ins:
        path, _, sel = spec.partition(':')
        with h5py.File(path, 'r') as fi:
            Di = fi['data']
            if 'env_args' not in Do.attrs:
                Do.attrs['env_args'] = Di.attrs['env_args']
            keys = sel.split(',') if sel else sorted(Di, key=lambda k: int(k.split('_')[1]))
            for k in keys:
                fi.copy(Di[k], Do, f'demo_{i}'); total += int(Di[k].attrs['num_samples'])
                rows.append((f'demo_{i}', os.path.basename(path), k)); i += 1
    Do.attrs['total'] = total
with open(mp, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['demo', 'src_file', 'src_demo']); w.writerows(rows)
print(f'{len(rows)} demos, total {total} -> {out}')
