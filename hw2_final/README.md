# Toy2Box BC 최종 제출 모델

## 최종 모델
- 사람 시범 100개 (조교 시범 없음) + Isaac Lab Mimic 성공 시범 1,000개 + BC-RNN
- 기본 `bc_rnn_low_dim.json`에서 `algo.rnn.horizon`과 `train.seq_length`만 10 → 50으로 변경 (`configs/`)
- 학습 seed 101, 111. 2000 epoch, 20만 gradient step

## 결과 (평가 seed 42, rollout 100회, horizon 800)
| 모델 | seed 101 | seed 111 | 평균 ± 표본 표준편차 |
|---|---|---|---|
| Baseline (조교 20 + Mimic 192, 기본 설정) | 23 | 23 | 23.0 ± 0.0 |
| 최종 모델 | 55 | 47 | 51.0 ± 5.7 |

## 사람 시범 100개 선택 기준 (`hw2_final/sel100_selection.csv`)
1. 필수: 시연자 1 s1~s4 20개 중 자동 annotation을 통과한 18개
2. 후보 120개(시연자 1 s5, 시연자 2–5 파일) 중 자동 annotation 통과 111개를 다음 순서로 정렬해 상위 82개 선택:
   그리퍼 전환 2회(단일 파지) → 길이 800 step 이하 → 짧은 길이 → 낮은 idle 비율 → 동점이면 시연자 1 s5
3. 구성: 시연자 1 39개(s1~s4 18, s5 21), 시연자 2–5 61개(시연자 2 22, 시연자 4 20, 시연자 5 11, 시연자 3 8). 트리밍 없음
4. 파일 대응: 시연자 1 = datasets.zip(s1~s4) + toy2box_agibot.hdf5(s5), 시연자 2 = team01_kim_datasets.zip, 시연자 3 = team06_v1_20.hdf5, 시연자 4 = toy2box_kgh.hdf5, 시연자 5 = toy2box_team13.hdf5

## 학습 명령어 (저장소 루트, seed 101 예시)
데이터셋은 용량 때문에 저장소에 포함하지 않았습니다.
```
./isaaclab.sh -p hw2_scripts/train_cfg.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc \
  --config hw2_final/configs/cfg_rnn50_seed101.json \
  --dataset ./datasets/own20_sel80_raw_mimic1000.hdf5 --name own100m1000_rnn50_seed101
```

## 평가 명령어 (`eval_commands.txt`와 동일)
```
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint hw2_final/checkpoints/seed101/model_epoch_2000.pth \
  --headless --seed 42 --num_rollouts 100 --horizon 800

PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint hw2_final/checkpoints/seed111/model_epoch_2000.pth \
  --headless --seed 42 --num_rollouts 100 --horizon 800
```

## 실험에 쓴 스크립트 (`hw2_scripts/`)
최종 모델에 필요한 것은 `select100.py`, `merge_with_map.py`, `cap_demos.py`, `train_cfg.py`이고, 나머지는 비교 실험과 진단용입니다.
- `select100.py`: 위 기준으로 시범 100개 선택, 선택표 저장
- `merge_with_map.py`: 시범 병합과 원본 매핑 기록
- `cap_demos.py`: 생성 결과를 성공 시범 N개로 자르기
- `pipeline.sh`, `finish.sh`: annotation → Mimic 생성 → 학습 → 평가 자동 실행
- `train_cfg.py`, `train_cfg.sh`: 별도 config로 학습. 환경 변수로 가중 샘플링(`HW2_SAMPLER`), 학습 중 관측 노이즈(`HW2_OBS_NOISE`) 지원
- `generate_dataset_seed.py`: 생성 seed를 지정하는 Mimic 생성(`HW2_GEN_SEED`)
- `balance_truck_x.py`: 생성 풀 2,000개에서 트럭 x 구간별 200개씩 1,000개 선택(생성 후 선택. 생성 중 할당은 아님)
- `approach_tilt.py`: 파지 순간 그리퍼 기울기 계산. 25° 필터(`postfilt`)와 자세 플래그(`postcond`) 데이터의 기준
- `trim_idle.py`: 연속 멈춤 구간 트리밍
- `play_record.py`, `failrec.sh`: 평가하며 rollout 상태 기록
- `classify_failures.py`, `eval_summary.py`, `xbin_success.py`: 접근/파지/운반/배치 실패 분류, 구간별 집계
- `make_figs.py`: 발표 그림 생성
