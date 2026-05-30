# Python TD3 Experiments

This folder starts the Python-based replication path for the paper's DC motor study.

## What is included

- `dc_motor_env.py`
  A Gymnasium environment that follows the paper's nominal plant, nonlinear perturbations, and four benchmark scenarios.
- `experiment1_train_td3_many_seeds.py`
  A multi-seed TD3 runner for Experiment 1 from your notes.
- `experiment2_add_controllers.py`
  A controller-comparison runner for PID, LQR, LQI, linear MPC, TD3, and optional DDPG/SAC baselines.

## Experiment 1 goal

Train TD3 multiple times with different random seeds, then evaluate each trained model on:

- `nominal`
- `ra_plus_50`
- `j_plus_50`
- `friction`
- `saturation`
- `combined_stress`

Across all four scenarios:

- `step_nominal`
- `step_load_disturbance`
- `ramp`
- `sine`

Metrics saved per run:

- `IAE`
- `ISE`
- `SSE`
- `ControlEnergy`
- `RiseTime`
- `SettlingTime`
- `Overshoot`

Graphs generated automatically after the run:

- `td3_learning_curves.png`
- `mean_ramp_iae_error_bars.png`
- `td3_ramp_iae_boxplot.png`
- `seed_mean_metrics_table.png`

## Run a quick smoke test

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --seeds 1 `
  --timesteps 2000
```

## Run the full repeated-seed experiment

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --seeds 10 `
  --timesteps 50000
```

Use a fresh output directory if the observation or reward function changes, because old models are not comparable with new-environment runs.

## Run the hard-nonlinearity energy experiment

This variant focuses training on saturation and combined-stress cases, with more step/sine exposure and stronger penalties for sustained near-limit voltage.

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --seeds 10 `
  --timesteps 50000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --output-dir "python_td3_experiments\outputs\experiment2_hard_energy_10seeds_200k"
```

## Run the balanced energy/error experiment

This variant softens the voltage penalty from Experiment 2 and mainly penalizes high voltage when error remains large. It keeps the hard nonlinear training distribution.

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --seeds 10 `
  --timesteps 50000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --output-dir "python_td3_experiments\outputs\experiment3_balanced_energy_10seeds_50k"
```

## Run the final condition-aware experiment

This final variant uses condition-aware reward weights: it allows more voltage in `combined_stress` when error is large, penalizes ineffective high voltage, and increases late/final tracking pressure in the hard nonlinear cases.

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --seeds 10 `
  --timesteps 100000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --output-dir "python_td3_experiments\outputs\experiment4_condition_aware_10seeds_100k"
```

## Try SAC instead of TD3

Use the same training runner with `--algorithm SAC`. Start with one seed before launching 10 seeds, because we first need to see whether SAC can reduce the TD3 tracking weakness against LQR/MPC.

Single-seed SAC probe:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --algorithm SAC `
  --seeds 1 `
  --timesteps 100000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --output-dir "python_td3_experiments\outputs\sac_hard_1seed_100k"
```

Full 10-seed SAC repeatability run:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --algorithm SAC `
  --seeds 10 `
  --timesteps 100000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --reward-profile balanced `
  --output-dir "python_td3_experiments\outputs\sac_balanced_10seeds_100k"
```

Compare the trained SAC model against classical controllers:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment2_add_controllers.py" `
  --output-dir "python_td3_experiments\outputs\sac_hard_1seed_100k_controller_compare" `
  --sac-model "python_td3_experiments\outputs\sac_hard_1seed_100k\seed_00\sac_model.zip"
```

After the 10-seed SAC run finishes, generate the final SAC vs TD3 vs classical mean +/- std comparison:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\compare_sac_td3_classical.py" `
  --sac-dir "python_td3_experiments\outputs\sac_balanced_10seeds_100k" `
  --td3-dir "python_td3_experiments\outputs\experiment4_condition_aware_10seeds_100k" `
  --classical-csv "python_td3_experiments\outputs\experiment2_more_controllers_full_rl\controller_metrics_long.csv" `
  --output-dir "python_td3_experiments\outputs\sac_vs_td3_classical_comparison"
```

## Best current RL setup: residual SAC over LQR

This is the strongest setup so far. SAC learns a residual correction on top of nominal LQR instead of learning the full voltage command from scratch. It also warm-starts the replay buffer with MPC expert behavior, uses a nominal-to-hard curriculum, and selects the best validation checkpoint.

Train one seed:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment1_train_td3_many_seeds.py" `
  --algorithm SAC `
  --control-mode residual_lqr `
  --residual-voltage-limit 8 `
  --expert-warmstart-steps 20000 `
  --expert-controller mpc `
  --curriculum nominal_to_hard `
  --seeds 1 `
  --timesteps 100000 `
  --chunk-timesteps 50000 `
  --print-every-steps 10000 `
  --training-condition-mode hard `
  --training-scenario-mode hard_tracking `
  --reward-profile balanced `
  --output-dir "python_td3_experiments\outputs\residual_sac_curriculum_mpcwarm_1seed_100k"
```

Select the best checkpoint:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\select_best_rl_checkpoint.py" `
  --run-dir "python_td3_experiments\outputs\residual_sac_curriculum_mpcwarm_1seed_100k" `
  --algorithm SAC `
  --control-mode residual_lqr `
  --residual-voltage-limit 8
```

Compare the best-selected checkpoint against TD3 and classical controllers:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\compare_sac_td3_classical.py" `
  --sac-dir "python_td3_experiments\outputs\residual_sac_curriculum_mpcwarm_1seed_100k\best_selected" `
  --td3-dir "python_td3_experiments\outputs\experiment4_condition_aware_10seeds_100k" `
  --classical-csv "python_td3_experiments\outputs\experiment2_more_controllers_full_rl\controller_metrics_long.csv" `
  --output-dir "python_td3_experiments\outputs\residual_sac_curriculum_mpcwarm_1seed_100k_best_comparison"
```

## Run the controller-comparison experiment

This evaluates all controllers on the same six plant conditions and four reference scenarios using the same +/-24 V voltage limit. Classical controllers are designed once on the nominal model and reused without retuning.

Fast classical + existing TD3 comparison:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment2_add_controllers.py" `
  --output-dir "python_td3_experiments\outputs\experiment2_more_controllers_with_td3"
```

Full comparison with freshly trained DDPG and SAC baselines:

```powershell
& "C:\Users\Nathnael Biresaw\AppData\Local\Programs\Python\Python311\python.exe" `
  "python_td3_experiments\experiment2_add_controllers.py" `
  --output-dir "python_td3_experiments\outputs\experiment2_more_controllers_full_rl" `
  --include-ddpg `
  --include-sac `
  --rl-timesteps 50000
```

Main outputs:

- `controller_metrics_long.csv`
- `controller_summary.csv`
- `controller_winner_table.csv`
- `controller_comparison_heatmap.png`
- `mean_iae_by_controller.png`
- `mean_control_energy_by_controller.png`
- `controller_winner_table.png`
