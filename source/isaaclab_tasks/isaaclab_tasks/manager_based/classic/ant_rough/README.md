# Isaac-Ant-Rough: walking on unseen terrain

Extension of `Isaac-Ant-v0` that trains the Ant with terrain and physics domain randomization
while keeping the observation (60-dim), action (8-dim) and reward definitions of the original task
unchanged. A policy trained here is therefore a drop-in replacement for the baseline checkpoint.

## Tasks

| Task id | Purpose | Terrain | Ground friction | Robot randomization |
| --- | --- | --- | --- | --- |
| `Isaac-Ant-Rough-v0` | training | flat, random rough, slopes, pyramid stairs, boxes (20 difficulty levels, curriculum) | 1.0 | foot friction 0.3-1.5, torso mass x0.7-1.3, random pushes |
| `Isaac-Ant-Rough-Play-v0` | in-distribution evaluation | same as above, 12 rows, start on easiest row | 1.0 | friction and mass only |
| `Isaac-Ant-Unseen-v0` | unseen-terrain evaluation | wave, discrete obstacles, repeated boxes, hard rough, stepping stones (12 rows, start on easiest row) | 1.0 | none |
| `Isaac-Ant-Unseen-Slippery-v0` | unseen terrain + low friction | same as unseen | 0.3 | none |
| `Isaac-Ant-Unseen-Sticky-v0` | unseen terrain + high friction | same as unseen | 1.6 | none |
| `Isaac-Ant-Flat-Slippery-v0` | original flat terrain, low friction | plane | 0.3 | none |
| `Isaac-Ant-Flat-Sticky-v0` | original flat terrain, high friction | plane | 1.6 | none |
| `Isaac-Ant-Rough-Mild-v0` | training (ablation) | same 7 types, 40 percent flat, milder obstacles | 1.0 | same as Rough |
| `Isaac-Ant-Rough-TerrainOnly-v0` | training (ablation) | same as Rough | 1.0 | none |
| `Isaac-Ant-Flat-PhysicsOnly-v0` | training (ablation) | plane | 1.0 | same as Rough |
| `Isaac-Ant-Rough-All-v0` | training (final, blind) | all 12 types (7 training + 5 of the unseen set) | 1.0 | same as Rough |
| `Isaac-Ant-Rough-All-Stable-v0` | training (ablation) | all 12 types | 1.0 | same as Rough, plus angular-velocity and tilt penalties |
| `Isaac-Ant-Rough-All-Scan-v0` | training (final, with height scanner) | all 12 types | 1.0 | same as Rough |
| `Isaac-Ant-Unseen2-v0` | second held-out evaluation | repeated cylinders, repeated pyramids, height-field stairs, star bars, box steps | 1.0 | none |
| `Isaac-Ant-Unseen-Scan-v0`, `Isaac-Ant-Unseen2-Scan-v0`, `Isaac-Ant-Flat-Scan-v0`, `Isaac-Ant-Flat-Scan-Slippery-v0`, `Isaac-Ant-Flat-Scan-Sticky-v0` | evaluation for the scanner policy | same terrains as the non-Scan tasks | as named | none |

## What changed relative to the baseline

- `terrains.py`: procedural terrain sets for training (`ANT_ROUGH_TERRAINS_CFG`) and evaluation
  (`ANT_UNSEEN_TERRAINS_CFG`). The evaluation set contains only terrain types absent from training.
- `mdp/curriculums.py`: `terrain_levels_progress` moves an environment to a harder terrain row when the
  Ant travelled more than 6 m along +x in the episode and to an easier one below 1.5 m.
- `mdp/terminations.py`: `root_height_below_minimum_relative` replaces the world-frame height check of
  the baseline with a height measured by a downward ray caster, so the fall detection works on
  elevated terrain. Outside the terrain mesh it falls back to the world-frame check. This changes only the termination, not the policy inputs.
- `ant_rough_env_cfg.py`: scene with terrain generator and height scanner, startup randomization of
  rigid-body material (friction, restitution) and torso mass, interval pushes, small yaw randomization at reset.
- Fabric cloning is disabled for these tasks because the ray caster needs every environment on the USD stage.

## Final submission

Policy: `Isaac-Ant-Final-Train-Smooth-v0` (14 terrain types incl. pillars, physics randomization, height scanner,
3-step observation history, original reward + gait-smoothness penalties, **baseline PPO settings, 1000 iterations**, seed 42).
Checkpoint: `logs/rsl_rl/ant_rough/2026-10-05_15-55-59_fin_smooth/model_999.pt`.
Evaluate with `Isaac-Ant-Final-Eval-v0` (hard held-out terrain), `Isaac-Ant-Final-Obstacles-v0` (held-out obstacle course)
or `Isaac-Ant-Final-Flat-v0` (plane); all use the original Isaac-Ant-v0 reward terms. To evaluate on another terrain,
edit `AntFinalEvalEnvCfg.__post_init__`. Earlier submission candidates (3000-iteration, entropy 0.005 runs) are kept
under `logs/rsl_rl/ant_rough/*_ant_final_*` for reference only.

Other registered variants (training): `Isaac-Ant-Rough-All-Grid-v0`, `-Grid-Scan-v0` (30 percent grid boxes),
`Isaac-Ant-Diverse-v0`, `-Diverse-Scan-v0`, `-Diverse-Scan-Pd-v0` (pillars, steep slopes, PD control),
`Isaac-Ant-Rough-All-Death-v0`, `-Scan-Death-v0` (one-off fall penalty), `Isaac-Ant-Rough-All-Scan-Hist-v0`,
`-Scan-Explore-v0`, `-Scan-Hist-Explore-v0`. Evaluation: `Isaac-Ant-Grid(-Slippery)(-Scan)(-Hist|-Pd)-v0`,
`Isaac-Ant-Unseen3-Hard(-Scan)(-Hist|-Pd)-v0`, `Isaac-Ant-Unseen2-Scan(-Hist|-Pd)-v0`, `Isaac-Ant-Flat-Scan(-Slippery|-Sticky)(-Hist|-Pd)-v0`.

## Height-scanner policy

`Isaac-Ant-Rough-All-Scan-v0` adds a `terrain_scanner` ray caster (1.6 x 1.0 m grid, 0.2 m resolution,
yaw-aligned, centred 0.3 m ahead of the torso) and a 54-dim `height_scan` observation term
(`mdp/observations.py`). The policy observation becomes 114-dim; actions, rewards and terminations are
unchanged. Policies trained with this task must be evaluated with a `*-Scan-v0` task so that the sensor exists.

## Train

```bash
cd ~/IsaacLab_RS
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ant-Rough-v0 --headless --seed 42 --run_name ant_rough
```

Checkpoints are written to `logs/rsl_rl/ant_rough/<timestamp>_ant_rough/`.

## Evaluate

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
    --task Isaac-Ant-Unseen-v0 --seed 24 --num_envs 64 --headless \
    --checkpoint logs/rsl_rl/ant_rough/<RUN_FOLDER>/model_1499.pt
```

Any task from the table can be substituted for `--task`, and the baseline checkpoint from
`logs/rsl_rl/ant/` can be evaluated with the same command for comparison.
