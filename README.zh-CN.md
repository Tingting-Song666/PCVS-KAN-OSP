# 论文代表性代码（精简公开版）

本仓库仅整理与论文核心方法直接对应的 **5 个原始 Python 程序**，包括 PC-VS、GA 传感器布设、10DOF 时程 RBF-KAN 和 20DOF 布设排序 RBF-KAN。随附少量原有数值结果和数据划分清单，便于了解方法的主要实现。

- `pc_virtual_sensing_enum10.py`：5DOF PC-VS 与全枚举。
- `ga_pc_vs_10dof_stage4.py`：10DOF PC-VS + GA。
- `ga_pc_vs_20dof_stage5.py`：20DOF PC-VS + GA。
- `stage6_v3_residual_prior_fix_inputdim.py`：10DOF 响应重构 RBF-KAN。
- `rank20_REGENERATE_kan_consistency_FINAL.py`：20DOF 布设排序 RBF-KAN。

**使用边界：** 这是用于展示研究方法的精简代码，不是完整的实验复现包。未包含原始地震动、响应矩阵、完整 PC-VS 标签、训练权重以及 E1–E5 的完整验证程序；因此不能直接据此复现论文的全部数值表格。大多数程序默认需要 CUDA，运行前需要自行准备相应数据和环境。

原代码科学实现保持不变，仅对排序脚本注释中的两处本地绝对路径示例进行了脱敏；不改变计算逻辑。结果 CSV 为已有实验结果，不是本次重新运行所得。仓库不包含私有训练缓存和未经确认可再分发的原始地震动数据。
