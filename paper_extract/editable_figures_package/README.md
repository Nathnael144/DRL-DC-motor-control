Editable figure package for the submitted paper

This folder collects the figure files used in `robust_speed_control_td3_lqr_converted.tex` together with the main source files needed to edit or regenerate them.

Folder structure

- `figures/`
  The exported figure files used in the paper.
- `figure_sources/`
  The LaTeX and Python source files used to generate the figures and define the benchmark scenarios.
- `csv_sources/`
  The relevant CSV inputs and summary tables used to generate the paper figures.
- `notes/`
  Editable notes that answer the reviewers' request for missing operating-condition details.

Main reviewer clarification

The benchmark scenarios were defined directly in the simulation environment. The load-disturbance case uses a constant `100 rad/s` reference with a `0.02 N*m` load torque step applied at `0.2 s` and maintained until the end of the `0.5 s` episode. The sinusoidal reference is `100 + 20 sin(2*pi*2*t) rad/s`, i.e., mean `100 rad/s`, amplitude `20 rad/s`, frequency `2 Hz`. The ramp reference is `min(200 t, 120) rad/s`, and the nominal step reference is `100 rad/s`.

Figure source map

- `methodology_flow_diagram.pdf`
  Source: `figure_sources/methodology_flow_diagram.tex`
- `controller_comparison_heatmap.png`
  Source: `figure_sources/experiment2_add_controllers.py`
- `mean_iae_by_controller.png`
  Source: `figure_sources/experiment2_add_controllers.py`
- `mean_control_energy_by_controller.png`
  Source: `figure_sources/experiment2_add_controllers.py`
- `sac_learning_curves.png`
  Source: `figure_sources/experiment1_train_td3_many_seeds.py`
- `td3_learning_curves.png`
  Source: `figure_sources/experiment1_train_td3_many_seeds.py`
- `ddpg_learning_curves.png`
  Source: `figure_sources/experiment1_train_td3_many_seeds.py`
- `overall_mean_iae_with_seed_std.png`
  Source: `figure_sources/compare_sac_td3_classical.py`
- `overall_mean_sse_with_seed_std.png`
  Source: `figure_sources/compare_sac_td3_classical.py`
- `hard_cases_mean_iae_with_seed_std.png`
  Source: `figure_sources/compare_sac_td3_classical.py`
- `hard_cases_mean_energy_with_seed_std.png`
  Source: `figure_sources/compare_sac_td3_classical.py`
- `winner_count_bars.png`
  Source: `figure_sources/generate_residual_sac_report_graphs.py`
- `training_learning_curves_all_seeds.png`
  Source: `figure_sources/generate_residual_sac_report_graphs.py`
- `training_reward_with_curriculum_stages.png`
  Source: `figure_sources/generate_residual_sac_report_graphs.py`
- `overall_backbone_iae.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `overall_backbone_sse.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `hard_backbone_iae.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `hard_backbone_sse.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `backbone_winner_counts.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `sac_mpc_learning_curves.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`
- `sac_hinf_learning_curves.png`
  Source: backbone comparison output included here; related benchmark definitions are in `figure_sources/dc_motor_env.py` and the paper source in `figure_sources/robust_speed_control_td3_lqr_converted.tex`

Useful files for direct editing

- `figure_sources/robust_speed_control_td3_lqr_converted.tex`
  Main paper source showing where each figure is used.
- `figure_sources/dc_motor_env.py`
  Exact scenario definitions and operating-condition values.

CSV source folders

- `csv_sources/controller_comparison/`
  Data behind the direct/controller comparison figures.
- `csv_sources/residual_sac_hinf_learning/`
  Residual SAC H-infinity training summaries and checkpoint-selection tables.
- `csv_sources/td3_learning/`
  TD3 learning summaries used for the TD3 curve figure context.
- `csv_sources/ddpg_learning/`
  DDPG learning summaries used for the DDPG curve figure context.
- `csv_sources/cross_controller_comparison/`
  Combined metrics and summary tables for the overall and hard-case comparison figures.
- `csv_sources/residual_sac_report/`
  Training-monitor exports and winner-count/report tables used by the residual SAC report figures.
- `csv_sources/backbone_comparison/`
  Summary tables and winner tables behind the backbone comparison figures.
