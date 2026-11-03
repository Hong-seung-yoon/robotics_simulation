"""평가 1건 요약: 성공 수, 실패 단계, x 구간별 성공, 그리퍼 닫힘 시 기울기 중앙값."""
import sys, glob, numpy as np
t = sys.argv[1]; d = f'logs/hw2/failrec_{t}_eval42'
E = np.linspace(.45, .8, 6); R = []
for f in sorted(glob.glob(d + '/trial_*.npz')):
    z = np.load(f); e, tr, b = z['eef_pos'], z['toy_truck_positions'], z['box_positions']; lift = tr[:, 2] - tr[0, 2]; l = lift > .03
    xy = np.linalg.norm(tr[:, :2] - b[:, :2], axis=1); dm = np.linalg.norm(e - tr, axis=1).min(); s = bool(z['success'])
    st = 'S' if s else 'A' if dm > .051 else 'G' if lift.max() < .03 else 'T' if not (l.any() and xy[l].min() < .1) else 'P'
    i = np.nonzero(z['actions'][:, 6] < 0)[0]; ang = np.nan
    if len(i):
        w, x, y, zz = z['eef_quat'][i[0]]; ang = np.degrees(np.arccos(np.clip(-(1 - 2 * (x * x + y * y)), -1, 1)))
    fl = float(z['posture_flag'].mean()) if 'posture_flag' in z else np.nan
    R.append((tr[0, 0], s, st, ang, fl))
x = np.array([r[0] for r in R]); s = np.array([r[1] for r in R]); st = [r[2] for r in R]; a = np.array([r[3] for r in R])
b = [f"{int(s[m].sum())}/{int(m.sum())}" for m in [((x >= E[i]) & (x < E[i + 1] + (1e-6 if i == 4 else 0))) for i in range(5)]]
print(f"RESULT {t}: {int(s.sum())}/{len(s)} A/G/T/P={st.count('A')}/{st.count('G')}/{st.count('T')}/{st.count('P')} bins={b} tilt median={np.nanmedian(a):.1f} (succ {np.nanmedian(a[s]):.1f}, fail {np.nanmedian(a[~s]):.1f}) flag={np.nanmean([r[4] for r in R]):.2f}")
