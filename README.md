# PC-VS and RBF-KAN for Optimal Sensor Placement

Representative research code for **“A Framework Integrating Physically Constrained Virtual Sensing and Kolmogorov–Arnold Network Surrogate for Response Reconstruction and Optimal Sensor Placement.”**

The work evaluates layouts by the reconstruction error of unmeasured structural responses. A PC-VS evaluator is used with genetic-algorithm (GA) search; two distinct RBF-KAN surrogates are developed for response reconstruction (10DOF) and layout ranking (20DOF).

## Included code

| File | Main purpose |
| --- | --- |
| `pc_virtual_sensing_enum10.py` | PC-VS reconstruction and 5DOF exhaustive enumeration |
| `ga_pc_vs_10dof_stage4.py` | 10DOF PC-VS and GA placement search |
| `ga_pc_vs_20dof_stage5.py` | 20DOF PC-VS and GA placement search |
| `stage6_v3_residual_prior_fix_inputdim.py` | 10DOF RBF-KAN time-history correction model |
| `rank20_REGENERATE_kan_consistency_FINAL.py` | 20DOF RBF-KAN layout-ranking model |

The source scripts preserve their original scientific implementation. Two absolute local-path examples were removed **from comments only** in the 20DOF ranking script. No equations, loss terms, network parameters, or optimization settings were changed in this packaging step.

## Data and representative results

The CSV files at the repository root record case splits and input locations. Small result extracts are included under the corresponding `*_results` directories; these are **previously generated outputs**, not newly rerun benchmarks. The 20DOF known-layout reference table is provided in `ranking_consistency_20dof_REGENERATED_FINAL/`.

**This is a curated code release, not a complete, stand-alone reproduction environment.** The original earthquake records, simulated displacement histories, Stage6 teacher-label caches, trained checkpoints, full reviewer experiments (E1–E5), and full preprocessing/workflow utilities are **not** included. Some scripts require those files and a CUDA-enabled PyTorch environment. Do not interpret the presence of representative source and result files as independent verification of every manuscript table.

## Environment and use

Install Python and a compatible PyTorch build, then:

```bash
python -m pip install -r requirements.txt
```

Most scripts require an NVIDIA CUDA GPU (`FORCE_GPU = True` by default). Run scripts from the repository root **after** providing the missing inputs at the relative locations listed in `case_list*.csv`. For example:

```bash
python pc_virtual_sensing_enum10.py
python ga_pc_vs_10dof_stage4.py
python ga_pc_vs_20dof_stage5.py
python stage6_v3_residual_prior_fix_inputdim.py
python rank20_REGENERATE_kan_consistency_FINAL.py
```

These are original compute-intensive scripts, **not** a quick demo. Outputs may be written to directories already used for the included result snapshots; back up the repository before running. No runtime reproduction of the reported manuscript metrics has been performed for this release.

The five E1–E5 review experiments and their final numerical output files are outside the scope of this **representative-code** release. This must be reflected accurately in any Code Availability or reviewer-response statement.

Original ground-motion records are not redistributed here because their reuse and redistribution conditions have not been verified.

See [中文说明](README.zh-CN.md) for a brief Chinese guide.
