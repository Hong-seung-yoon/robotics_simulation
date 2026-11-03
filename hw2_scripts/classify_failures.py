"""play_record.py가 저장한 rollout별 npz로 실패 단계를 분류한다.

단계(위치는 모두 로봇 base frame):
  approach : eef가 트럭에 충분히 다가가지 못함        (min |eef-truck| > APPROACH)
  grasp    : 다가갔지만 트럭을 들어올리지 못함         (max 상승 < LIFT)
  transport: 들었지만 박스 위(xy<XY)까지 옮기지 못함 (떨어뜨림 포함)
  place    : 박스 위까지 갔지만 성공 조건 미충족       (놓기/그리퍼 열기/자세 문제)
APPROACH 기본값은 성공 rollout들의 min 거리 분포(최댓값 x1.5)로 보정한다.
usage: python classify_failures.py RECORD_DIR [--lift 0.03] [--xy 0.10] [--csv out.csv]
"""
import argparse, glob, os, csv
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('record_dir')
p.add_argument('--lift', type=float, default=0.03)
p.add_argument('--xy', type=float, default=0.10)  # 성공 판정 xy_threshold와 동일
p.add_argument('--approach', type=float, default=None)
p.add_argument('--csv', default=None)
a = p.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.record_dir, 'trial_*.npz'))):
    d = np.load(f)
    eef, truck, box = d['eef_pos'], d['toy_truck_positions'], d['box_positions']
    lift = truck[:, 2] - truck[0, 2]
    lifted = lift > a.lift
    xy = np.linalg.norm(truck[:, :2] - box[:, :2], axis=1)
    rows.append(dict(
        trial=int(os.path.basename(f)[6:9]), success=bool(d['success']), early_stop=bool(d['early_stop']),
        steps=len(eef), d_min=float(np.linalg.norm(eef - truck, axis=1).min()),
        lift_max=float(lift.max()),
        xy_min_lifted=float(xy[lifted].min()) if lifted.any() else np.nan,
        xy_final=float(xy[-1]), dz_final=float(truck[-1, 2] - box[-1, 2]),
        grip_final=float(d['gripper_pos'][-1].mean())))

succ = [r for r in rows if r['success']]
if a.approach is None:
    a.approach = 1.5 * max(r['d_min'] for r in succ) if succ else 0.06
print(f"thresholds: approach={a.approach:.3f} m, lift={a.lift:.3f} m, xy={a.xy:.3f} m  "
      f"(성공 {len(succ)}개 d_min 범위 {min((r['d_min'] for r in succ), default=np.nan):.3f}"
      f"~{max((r['d_min'] for r in succ), default=np.nan):.3f})")

def stage(r):
    if r['success']: return 'success'
    if r['d_min'] > a.approach: return 'approach'
    if r['lift_max'] < a.lift: return 'grasp'
    if not (r['xy_min_lifted'] < a.xy): return 'transport'
    return 'place'

for r in rows:
    r['stage'] = stage(r)
order = ['approach', 'grasp', 'transport', 'place']
fails = [r for r in rows if not r['success']]
print(f"rollouts {len(rows)} | success {len(succ)} | fail {len(fails)}")
for s in order:
    g = [r for r in fails if r['stage'] == s]
    es = sum(r['early_stop'] for r in g)
    print(f"  {s:<9s} {len(g):3d} ({len(g)/max(len(fails),1):.0%})  early_stop(트럭 낙하 종료)={es}  "
          f"trials={[r['trial'] for r in g]}")
if a.csv:
    with open(a.csv, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('csv:', a.csv)
