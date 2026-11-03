"""파지 순간(그리퍼 닫힘 첫 step)의 eef 자세를 시범 그룹별로 계산해 CSV로 저장한다. 위치와 자세는 로봇 base frame."""
import csv, glob, h5py, numpy as np
D = 'datasets/'
G = {'TA 20': [D+'toy2box_agibot.hdf5'], 'Own s1-s4': [D+f'toy2box_s{i}.hdf5' for i in range(1,5)], 'Own s5': [D+'toy2box_s5.hdf5'],
     'team01_kim': sorted(glob.glob(D+'cand/team01_kim_s*.hdf5')), 'team06': [D+'cand/team06_v1_20.hdf5'], 'kgh': [D+'cand/toy2box_kgh.hdf5'], 'team13': [D+'cand/toy2box_team13.hdf5'],
     'Selected 100': [D+'own20_sel80_raw_annotated.hdf5'], 'Mimic 1000': [D+'own20_sel80_raw_mimic1000.hdf5']}
def R(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
rows = []
for g, fs in G.items():
    for f in fs:
        S = h5py.File(f, 'r')['data']
        for k in S:
            a = S[k]['actions'][:]; o = S[k]['obs']; idx = np.nonzero(a[:, 6] < 0)[0]
            if len(idx) == 0: continue
            t = int(idx[0]); q = o['eef_quat'][t]; M = R(q); e = o['eef_pos'][t]; tr = o['toy_truck_positions'][t]
            y0 = o['toy_truck_orientations'][t]; yaw = np.arctan2(2*(y0[0]*y0[3]+y0[1]*y0[2]), 1-2*(y0[2]**2+y0[3]**2))
            ang = [float(np.degrees(np.arccos(np.clip(-M[2, i], -1, 1)))) for i in range(3)]
            hdg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            rows.append(dict(group=g, file=f.split('/')[-1], demo=k, t_close=t, ang_x=ang[0], ang_y=ang[1], ang_z=ang[2], heading=hdg, truck_yaw=float(np.degrees(yaw)), dist=float(np.linalg.norm(e-tr)), dz=float(e[2]-tr[2])))
with open('logs/hw2/approach_tilt.csv', 'w', newline='') as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for g in G:
    r = [x for x in rows if x['group'] == g]
    print(f"{g:<13s} n={len(r):4d} ang_x={np.mean([x['ang_x'] for x in r]):6.1f} ang_y={np.mean([x['ang_y'] for x in r]):6.1f} ang_z={np.mean([x['ang_z'] for x in r]):6.1f}±{np.std([x['ang_z'] for x in r]):5.1f} dist={np.median([x['dist'] for x in r])*100:5.1f}cm dz={np.median([x['dz'] for x in r])*100:5.1f}cm t={np.median([x['t_close'] for x in r]):5.0f}")
