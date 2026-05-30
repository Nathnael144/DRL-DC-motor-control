# DRL-DC-Motor-Control

This repository now includes a Python experiment pipeline for benchmarking deep RL against classical controllers for DC motor speed control.

## Added Python experiment package

The new work lives in [`python_td3_experiments`](python_td3_experiments).

Main scripts:

- `dc_motor_env.py`
- `experiment1_train_td3_many_seeds.py`
- `experiment2_add_controllers.py`
- `compare_sac_td3_classical.py`
- `select_best_rl_checkpoint.py`
- `generate_residual_sac_report_graphs.py`

## Final residual SAC result

Best setup:

- residual SAC over nominal LQR
- MPC replay warm-start
- nominal-to-hard curriculum
- best-checkpoint validation selection

Final 10-seed comparison outputs are in:

- `python_td3_experiments/outputs/residual_sac_curriculum_mpcwarm_10seeds_100k_best_comparison`
- `python_td3_experiments/outputs/residual_sac_curriculum_mpcwarm_10seeds_100k_report_graphs`

High-level result:

- Residual SAC mean IAE: `15.7019 +- 0.0170`
- LQR mean IAE: `15.7137`
- MPC mean IAE: `15.7381`

This means the selected residual SAC configuration slightly outperformed standalone LQR and MPC on mean IAE and steady-state error across the benchmark set.
