# 최종 후보별 제출 파일 (git 작업은 아직 하지 않음)

경로 접두: logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/
`logs/*`가 .gitignore에 있어 그대로는 올라가지 않음: gitignore 예외 추가 또는 `git add -f`, 혹은 models/final_*/ 같은 전용 폴더로 복사 필요. 모델은 각 약 8 MB.

| 후보 | 데이터 규칙 | seed 101 체크포인트 | seed 111 체크포인트 | 코드 변경 필요 |
|---|---|---|---|---|
| E (best 100 + Mimic 1000) | 타 조 시범 포함 여부 확인 필요 | own20_sel80_raw_seed101/20261029220420/models/model_epoch_2000.pth | own20_sel80_raw_seed111/20261029220620/models/model_epoch_2000.pth | 없음 |
| ① E + 상대 위치 obs | 동일 | own100m1000_relobs_seed101/20261031215916/models/model_epoch_2000.pth | own100m1000_relobs_seed111/20261031215916/models/model_epoch_2000.pth | **필수**: relobs_obs_terms.patch (mdp/observations.py, place_toy2box_rmp_rel_env_cfg.py) |
| ② E + RNN 메모리 50 | 동일 | own100m1000_rnn50_seed101/20261031215943/models/model_epoch_2000.pth | own100m1000_rnn50_seed111/20261101194500/models/model_epoch_2000.pth (학습 중) | 없음 |
| F (주인 50 + Mimic 1000) | 안전 | own45_raw_m1000_seed101/20261031203905/models/model_epoch_2000.pth | own45_raw_m1000_seed111/20261031204105/models/model_epoch_2000.pth | 없음 |
| F + RNN 메모리 50 | 안전 | own45m1000_rnn50_seed101/(학습 중) | own45m1000_rnn50_seed111/(학습 중) | 없음 |

주의
- 제출 저장소에는 최종 모델 2개(seed 101, 111)만 포함.
- 학습에 쓴 seed는 체크포인트 config로 확인 가능. 제출 전 각 run의 config.json으로 재확인.
- ①을 제출하면 조교의 play.py가 env에서 eef_to_truck, truck_to_box를 받아야 하므로 패치가 저장소에 있어야 함. 다른 후보는 패치가 있어도 없어도 실행됨(추가 obs 키는 정책이 무시).
