"""play 시점 env obs(eef_to_truck, truck_to_box)가 오프라인 계산식(truck-eef, box-truck)과 같은지 확인."""
import sys, glob, numpy as np
m1 = m2 = 0.0; n = 0
for f in glob.glob(sys.argv[1] + '/trial_*.npz'):
    d = np.load(f)
    m1 = max(m1, np.abs(d['eef_to_truck'] - (d['toy_truck_positions'] - d['eef_pos'])).max())
    m2 = max(m2, np.abs(d['truck_to_box'] - (d['box_positions'] - d['toy_truck_positions'])).max()); n += 1
print(f'relobs check over {n} rollouts: max|eef_to_truck - (truck-eef)|={m1:.2e}, max|truck_to_box - (box-truck)|={m2:.2e}', 'OK' if max(m1, m2) < 1e-5 else 'MISMATCH')
