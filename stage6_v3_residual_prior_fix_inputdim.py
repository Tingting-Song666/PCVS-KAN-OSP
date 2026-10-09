# -*- coding: utf-8 -*-
"""
stage6_pikan_fkan_fast_reconstructor.py

Stage 6：基于物理约束虚拟传感标签的 PIKAN/FKAN 快速重构模型
================================================================================
研究主线：
    以未监测自由度响应重构精度为目标，构建 GA-PIKAN/FKAN 联合优化框架，
    在固定传感器数量约束下搜索传感器布设方案，并通过结构动力学物理残差
    约束提高重构结果的物理一致性。

当前脚本定位：
    前面阶段已经验证：物理约束虚拟传感 PC-VS 可以作为可靠 evaluator，
    GA 可以在 10DOF/20DOF 中搜索优良传感器布设。

    Stage 6 不再让 FKAN/PIKAN 盲目单 case 训练，而是把 PC-VS 的重构结果
    作为高质量伪标签，训练一个快速重构网络：

        输入：sensor mask + noisy sensor response + interpolation prior + time + ground acceleration
        输出：prior 到 PC-VS teacher 的残差修正；最终响应 = prior + residual。
        传感器 DOF 采用硬约束，网络主要学习未监测 DOF 的物理修正量。
        标签：PC-VS 物理约束虚拟传感重构结果。
        辅助：训练时加入有限差分结构动力学残差，保持物理一致性。

运行方式：
    放在工程根目录运行：
        python stage6_pikan_fkan_fast_reconstructor.py

    它会自动：
        1) 读取 case_list.csv / case_list(2).csv；
        2) 自动生成/读取 10DOF 响应缓存；
        3) 读取 10DOF Top20 布设排名，若缺失则使用内置 fallback；
        4) 如无 PC-VS 标签缓存，则先生成 Stage6 标签；
        5) 训练 PIKAN/FKAN-style 快速重构模型；
        6) 在 train/val/test 上评估：
           - 与 PC-VS teacher 的拟合误差；
           - 与 clean true response 的最终重构误差；
           - 有限差分物理残差。

重要说明：
    - 第一次运行会生成 PC-VS 伪标签，耗时较长；后续会断点续跑并复用缓存。
    - 为了先验证 Stage6 路线，默认使用 Enum120/Top20 中前 12 个布局。
      如果你要完整 Top20，把 LABEL_TOPK_LAYOUTS 改成 20。

输出目录：
    stage6_pikan_fkan_fast_reconstructor_results/

主要输出：
    label_cache/stage6_label_manifest.csv
    checkpoints/best_stage6_model.pt
    train_log.csv
    eval_per_sample.csv
    eval_aggregate_by_split.csv
    eval_aggregate_by_mask.csv
    eval_aggregate_by_noise.csv
    stage6_summary.txt
"""

import os
import re
import ast
import math
import time as pytime
import itertools
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Safe L-BFGS
# =============================================================================
# PyTorch 2.11 的 torch.optim.LBFGS 在 _gather_flat_grad() 中使用 p.grad.view(-1)。
# 当 PC-VS 标签优化变量的梯度不是 contiguous 时，会触发：
# RuntimeError: view size is not compatible ... Use .reshape(...) instead.
# 这里仅重写 flat grad 收集逻辑，把 view(-1) 改为 reshape(-1)，不改变 L-BFGS 算法本身。
class SafeLBFGS(torch.optim.LBFGS):
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new_zeros(p.numel())
            elif p.grad.is_sparse:
                view = p.grad.to_dense().reshape(-1)
            else:
                view = p.grad.reshape(-1)
            if torch.is_complex(view):
                view = torch.view_as_real(view).reshape(-1)
            views.append(view)
        return torch.cat(views, 0)

# 绘图不是本阶段重点，默认关闭。如果需要，可后续另写可视化脚本。


# =============================================================================
# 0. 全局配置
# =============================================================================

# -------------------------
# 设备
# -------------------------
FORCE_GPU = True
GPU_ID = 0

if FORCE_GPU:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 没有检测到 CUDA。若想临时用 CPU，把 FORCE_GPU 改成 False。"
        )
    DEVICE = torch.device(f"cuda:{GPU_ID}")
else:
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print("=" * 100)
print("Stage 6-v3 | Residual-to-prior PIKAN/FKAN surrogate trained from PC-VS teacher")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Selected DEVICE : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU name        : {torch.cuda.get_device_name(DEVICE)}")
print("=" * 100)

# -------------------------
# 路径
# -------------------------
CASE_LIST_CANDIDATES = ["case_list(2).csv", "case_list.csv"]
RESP10_ROOT = "responses_10dof"
CASE10_CSV = "case_list_10dof.csv"
CASE10_SUMMARY_CSV = "case_summary_10dof.csv"
OUT_DIR = "stage6_v3_residual_prior_results"
LABEL_DIR = os.path.join(OUT_DIR, "label_cache")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "checkpoints")

# 优先复用 Stage6-v2 已经生成的 Top20 高质量 teacher labels，避免重复优化 1520 个 PC-VS 标签。
# 若该文件不存在，脚本会自动在本 v3 输出目录中重新生成 label cache。
REUSE_STAGE6_V2_LABELS = True
STAGE6_V2_MANIFEST_CANDIDATES = [
    os.path.join("stage6_v2_top20_teacherfit_results", "label_cache", "stage6_label_manifest.csv"),
    os.path.join("stage6_pikan_fkan_fast_reconstructor_results", "label_cache", "stage6_label_manifest.csv"),
]

TOP20_RANKING_CANDIDATES = [
    os.path.join("pc_vs_10dof_top20_noise_robust_results", "top20_noise_overall_mask_ranking.csv"),
    os.path.join("pc_vs_10dof_enum120_verify_results", "enum120_mask_ranking.csv"),
    "top20_noise_overall_mask_ranking.csv",
    "enum120_mask_ranking.csv",
]

# 如果找不到 ranking csv，使用这组 10DOF Top20 fallback。
FALLBACK_TOP20_MASKS = [
    "DOF3_DOF5_DOF9",    # 10DOF 噪声鲁棒最优
    "DOF3_DOF6_DOF10",   # 10DOF clean 全局最优
    "DOF2_DOF5_DOF9",
    "DOF1_DOF5_DOF9",
    "DOF3_DOF5_DOF10",
    "DOF3_DOF6_DOF9",
    "DOF2_DOF5_DOF10",
    "DOF2_DOF4_DOF9",
    "DOF2_DOF7_DOF9",
    "DOF2_DOF6_DOF9",
    "DOF1_DOF5_DOF10",
    "DOF2_DOF6_DOF10",
    "DOF3_DOF6_DOF8",
    "DOF2_DOF6_DOF8",
    "DOF3_DOF7_DOF9",
    "DOF1_DOF4_DOF9",
    "DOF3_DOF5_DOF8",
    "DOF2_DOF4_DOF8",
    "DOF3_DOF4_DOF9",
    "DOF1_DOF6_DOF10",
]

# -------------------------
# 结构参数：Stage6 先固定 10DOF
# -------------------------
N_DOF = 10
N_SENSOR = 3
M_VAL = 20000.0
K_VAL = 15000000.0
ALPHA_RAYLEIGH = 0.5
BETA_RAYLEIGH = 0.001

# 地震波读取，与 generate_data.py 保持一致
EQ_SKIPROWS = 1
EQ_TIME_COL = 0
EQ_ACC_COL = 1
EQ_SCALE_FACTOR = 0.1

# Newmark-beta
NEWMARK_GAMMA = 0.5
NEWMARK_BETA = 0.25

# -------------------------
# Stage6 标签生成配置
# -------------------------
# Stage6-v2：使用 Top20 标签，覆盖 clean 最优与噪声鲁棒优良区域。
LABEL_TOPK_LAYOUTS = 20
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05]

# 若想先快速试跑：MAX_CASES_PER_SPLIT = 1, LABEL_TOPK_LAYOUTS = 4, NOISE_LEVELS = [0.0, 0.05]
MAX_CASES_PER_SPLIT: Optional[int] = None

# PC-VS 标签生成的下采样点数与优化强度
LABEL_MAX_POINTS = 1000
LABEL_ADAM_STEPS = 900
LABEL_ADAM_LR = 2.5e-2
LABEL_USE_LBFGS = True
LABEL_LBFGS_MAX_ITER = 60

# PC-VS 标签生成损失权重。与前面 evaluator 逻辑保持一致。
W_PHYSICS_LABEL = 1.0
W_PRIOR_LABEL = 5e-2
W_SMOOTH_LABEL = 2e-3
W_SPATIAL_LABEL = 5e-2
W_ENERGY_LABEL = 2e-2
W_IC_LABEL = 5e-1
ENERGY_RATIO_LIMIT = 4.0

# 标签断点续跑
BUILD_LABEL_CACHE = True
RESUME_LABEL_CACHE = True
SAVE_LABEL_INCREMENTAL = True

# -------------------------
# Stage6 网络训练配置
# -------------------------
BASE_SEED = 2026
NOISE_SEED_BASE = 880000

WINDOW_LEN = 384
WINDOWS_PER_SEQUENCE_PER_EPOCH = 4
BATCH_SIZE = 10
NUM_WORKERS = 0

MODEL_HIDDEN = 192
MODEL_GRID_SIZE = 10
MODEL_NUM_BLOCKS = 6
MODEL_DROPOUT = 0.03

EPOCHS = 260
LR = 1.5e-3
WEIGHT_DECAY = 5e-6
GRAD_CLIP = 1.0
PATIENCE = 45

# 训练 loss 权重
# teacher_loss_unmon 是主目标；physics 是物理一致性辅助约束。
L_TEACHER_UNMON = 2.0
L_TEACHER_ALL = 0.05
L_PHYSICS = 2e-4
L_SMOOTH = 5e-5

# Stage6-v2 重点：先逼近 PC-VS teacher，再谈快速 fitness。
# 对 teacher 本身较差的样本降低权重，避免网络过度学习弱伪标签。
USE_TEACHER_QUALITY_WEIGHT = True
TEACHER_QUALITY_REF_NRMSE = 0.12
TEACHER_QUALITY_MIN_WEIGHT = 0.35

# 评价时是否保存部分预测
SAVE_EVAL_PREDICTIONS = True
MAX_SAVE_EVAL_PRED_PER_SPLIT = 8


# =============================================================================
# 1. 基础工具
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(s))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_mask_name(mask_name: str) -> Tuple[int, ...]:
    nums = re.findall(r"DOF(\d+)", str(mask_name))
    return tuple(sorted([int(x) - 1 for x in nums]))


def sensor_tuple_to_name(sensor_tuple: Tuple[int, ...]) -> str:
    return "_".join([f"DOF{i + 1}" for i in sensor_tuple])


def sensor_tuple_to_str(sensor_tuple: Tuple[int, ...]) -> str:
    return "_".join([str(i + 1) for i in sensor_tuple])


def noise_tag(noise_level: float) -> str:
    return f"noise{int(round(noise_level * 100)):02d}pct"


def robust_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-18:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def calc_metrics(u_true: np.ndarray, u_pred: np.ndarray, sensor_indices: List[int]) -> Dict:
    sensor_set = set(sensor_indices)
    recon_indices = [i for i in range(u_true.shape[1]) if i not in sensor_set]
    n_time = u_true.shape[0]

    err = u_pred - u_true

    if len(recon_indices) > 0:
        err_un = err[:, recon_indices]
        true_un = u_true[:, recon_indices]
        rmse_un = float(np.sqrt(np.mean(err_un ** 2)))
        rms_true_un = float(np.sqrt(np.mean(true_un ** 2)))
        nrmse_un = rmse_un / (rms_true_un + 1e-18)
    else:
        rmse_un = np.nan
        nrmse_un = np.nan

    if len(sensor_indices) > 0:
        err_m = err[:, sensor_indices]
        true_m = u_true[:, sensor_indices]
        rmse_m = float(np.sqrt(np.mean(err_m ** 2)))
        rms_true_m = float(np.sqrt(np.mean(true_m ** 2)))
        nrmse_m = rmse_m / (rms_true_m + 1e-18)
    else:
        rmse_m = np.nan
        nrmse_m = np.nan

    r2_per_dof = []
    rmse_per_dof = []
    for i in range(u_true.shape[1]):
        r2_per_dof.append(compute_r2(u_true[:, i], u_pred[:, i]))
        rmse_per_dof.append(float(np.sqrt(np.mean((u_pred[:, i] - u_true[:, i]) ** 2))))

    recon_r2 = [r2_per_dof[i] for i in recon_indices]
    sensor_r2 = [r2_per_dof[i] for i in sensor_indices]
    recon_rmse = [rmse_per_dof[i] for i in recon_indices]

    def nanmean_safe(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanmean(vals))

    def nanmin_safe(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanmin(vals))

    def nanmax_safe(vals):
        vals = np.asarray(vals, dtype=np.float64)
        if vals.size == 0 or np.all(np.isnan(vals)):
            return np.nan
        return float(np.nanmax(vals))

    return {
        "rmse_unmonitored": rmse_un,
        "nrmse_unmonitored": nrmse_un,
        "rmse_monitored": rmse_m,
        "nrmse_monitored": nrmse_m,
        "r2_per_dof": r2_per_dof,
        "rmse_per_dof": rmse_per_dof,
        "mean_reconstructed_R2": nanmean_safe(recon_r2),
        "worst_reconstructed_R2": nanmin_safe(recon_r2),
        "worst_reconstructed_RMSE": nanmax_safe(recon_rmse),
        "mean_sensor_R2": nanmean_safe(sensor_r2),
        "worst_sensor_R2": nanmin_safe(sensor_r2),
    }


# =============================================================================
# 2. 结构动力学与数据生成
# =============================================================================

def build_shear_building_mck(n_dof: int, m_val: float, k_val: float,
                             alpha_rayleigh: float, beta_rayleigh: float):
    M = np.diag([m_val] * n_dof).astype(np.float64)
    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    for i in range(n_dof):
        if i == 0:
            K[i, i] = 2.0 * k_val
            if n_dof > 1:
                K[i, i + 1] = -k_val
        elif i == n_dof - 1:
            K[i, i] = k_val
            K[i, i - 1] = -k_val
        else:
            K[i, i] = 2.0 * k_val
            K[i, i - 1] = -k_val
            K[i, i + 1] = -k_val
    C = alpha_rayleigh * M + beta_rayleigh * K
    return M, K, C


def load_eq_file(eq_file: str):
    data = np.loadtxt(eq_file, skiprows=EQ_SKIPROWS)
    eq_time = data[:, EQ_TIME_COL].astype(np.float64)
    eq_acc = data[:, EQ_ACC_COL].astype(np.float64) * EQ_SCALE_FACTOR
    eq_time = eq_time - eq_time[0]
    return eq_time, eq_acc


def resample_to_uniform_time(t_raw: np.ndarray, ag_raw: np.ndarray):
    dt_list = np.diff(t_raw)
    if np.any(dt_list <= 0):
        raise ValueError("地震波时间列不是严格递增。")
    dt = float(np.min(dt_list))
    t_uniform = np.arange(t_raw[0], t_raw[-1] + 0.5 * dt, dt)
    ag_uniform = np.interp(t_uniform, t_raw, ag_raw)
    t_uniform = t_uniform - t_uniform[0]
    return t_uniform, ag_uniform, dt


def newmark_beta_linear(M, C, K, ag, dt, gamma=0.5, beta=0.25):
    n_dof = M.shape[0]
    nt = len(ag)
    r = np.ones((n_dof, 1), dtype=np.float64)

    u = np.zeros((nt, n_dof), dtype=np.float64)
    v = np.zeros((nt, n_dof), dtype=np.float64)
    a = np.zeros((nt, n_dof), dtype=np.float64)

    p0 = -M @ r * ag[0]
    a[0, :] = np.linalg.solve(M, (p0[:, 0] - C @ v[0, :] - K @ u[0, :]))

    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    K_eff = K + a0 * M + a1 * C

    for i in range(nt - 1):
        p_next = -M @ r * ag[i + 1]
        rhs = (
            p_next[:, 0]
            + M @ (a0 * u[i, :] + a2 * v[i, :] + a3 * a[i, :])
            + C @ (a1 * u[i, :] + a4 * v[i, :] + a5 * a[i, :])
        )
        u[i + 1, :] = np.linalg.solve(K_eff, rhs)
        a[i + 1, :] = a0 * (u[i + 1, :] - u[i, :]) - a2 * v[i, :] - a3 * a[i, :]
        v[i + 1, :] = v[i, :] + dt * ((1.0 - gamma) * a[i, :] + gamma * a[i + 1, :])
    return u, v, a


def save_response_csv(out_csv: str, t: np.ndarray, u: np.ndarray):
    df = pd.DataFrame({"Time": t})
    for i in range(u.shape[1]):
        df[f"DOF{i + 1}"] = u[:, i]
    df.to_csv(out_csv, index=False)


def find_case_list() -> str:
    """
    选择基础 case_list。

    修正点：
    1) 如果目录里同时存在单条旧 case_list.csv 和新的多条 case_list(2).csv，自动选择 case 数更多的文件。
    2) Stage6 生成 10DOF 响应只需要 case_id/split/eq_file，不强制要求基础 case_list 有 response_file。
    """
    valid = []
    for p in CASE_LIST_CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"⚠️ 跳过无法读取的 case_list 候选: {p} | {e}")
            continue
        required = {"case_id", "split", "eq_file"}
        if not required.issubset(set(df.columns)):
            print(f"⚠️ 跳过 {p}: 缺少必要列 {required - set(df.columns)}，当前列={list(df.columns)}")
            continue
        valid.append((len(df), p))
    if not valid:
        raise FileNotFoundError("找不到可用的 case_list。需要至少包含 case_id, split, eq_file 三列。")
    valid.sort(key=lambda x: x[0], reverse=True)
    n, p = valid[0]
    print(f"✅ 使用基础 case_list: {p} | n_cases={n}")
    return p


def _normalize_case10_dataframe(df_case10: pd.DataFrame) -> pd.DataFrame:
    """兼容不同旧版本 case_list_10dof.csv 的列名。"""
    df = df_case10.copy()
    alias_map = {
        "response10_file": "response_file",
        "response_10dof_file": "response_file",
        "resp10_file": "response_file",
        "resp_file": "response_file",
        "response_path": "response_file",
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def _case10_cache_is_valid(df_case10: pd.DataFrame) -> bool:
    df = _normalize_case10_dataframe(df_case10)
    required = {"case_id", "split", "eq_file", "response_file"}
    if not required.issubset(set(df.columns)):
        print("⚠️ 已存在的 case_list_10dof.csv 列不完整，将自动重建。")
        print(f"   当前列: {list(df_case10.columns)}")
        print(f"   缺少列: {required - set(df.columns)}")
        return False
    paths = [str(x) for x in df["response_file"].tolist()]
    exists_count = sum(os.path.exists(x) for x in paths)
    if len(paths) > 0 and exists_count < len(paths):
        print(f"⚠️ case_list_10dof.csv 中 response_file 有缺失: {exists_count}/{len(paths)} 存在，将自动重建。")
        return False
    return True


def build_or_load_10dof_cases(M_np, C_np, K_np) -> List[Dict]:
    """
    自动生成/读取 10DOF 响应缓存。

    修正点：
    - 如果已有 case_list_10dof.csv 但缺少 response_file 列，不再 KeyError，而是自动重建。
    - 如果基础 case_list 有新旧多个版本，自动选择 case 数最多的版本。
    - 基础 case_list 只要求 case_id/split/eq_file；response_file 不是必须项。
    """
    rebuild_cache = True
    df_case10 = None

    if os.path.exists(CASE10_CSV):
        try:
            df_tmp = pd.read_csv(CASE10_CSV)
            if _case10_cache_is_valid(df_tmp):
                df_case10 = _normalize_case10_dataframe(df_tmp)
                rebuild_cache = False
                print(f"✅ 检测到有效 10DOF case 缓存: {CASE10_CSV} | n_cases={len(df_case10)}")
            else:
                rebuild_cache = True
        except Exception as e:
            print(f"⚠️ 读取 {CASE10_CSV} 失败，将自动重建: {e}")
            rebuild_cache = True

    if rebuild_cache:
        case_list_path = find_case_list()
        df_base = pd.read_csv(case_list_path)
        required = {"case_id", "split", "eq_file"}
        if not required.issubset(set(df_base.columns)):
            raise ValueError(f"{case_list_path} 缺少必要列: {required - set(df_base.columns)}")

        ensure_dir(RESP10_ROOT)
        records = []
        summaries = []

        print(f"开始根据 {case_list_path} 生成/重建 10DOF 响应缓存...")
        for _, row in df_base.iterrows():
            case_id = str(row["case_id"])
            split = str(row["split"])
            eq_file = str(row["eq_file"])
            if not os.path.exists(eq_file):
                raise FileNotFoundError(f"找不到地震波: {eq_file}")

            eq_time_raw, eq_acc_raw = load_eq_file(eq_file)
            t, ag, dt = resample_to_uniform_time(eq_time_raw, eq_acc_raw)
            u, v, a = newmark_beta_linear(M_np, C_np, K_np, ag, dt,
                                           gamma=NEWMARK_GAMMA, beta=NEWMARK_BETA)

            resp_dir = os.path.join(RESP10_ROOT, split)
            ensure_dir(resp_dir)
            base_name = safe_name(os.path.splitext(os.path.basename(eq_file))[0])
            resp_file = os.path.join(resp_dir, f"response10_{base_name}.csv")
            save_response_csv(resp_file, t, u)

            records.append({
                "case_id": case_id,
                "split": split,
                "eq_file": eq_file.replace("\\", "/"),
                "response_file": resp_file.replace("\\", "/"),
            })
            summaries.append({
                "case_id": case_id,
                "split": split,
                "eq_file": eq_file.replace("\\", "/"),
                "response_file": resp_file.replace("\\", "/"),
                "n_time_steps": len(t),
                "dt": float(dt),
                "duration": float(t[-1] - t[0]),
                "ag_max_abs": float(np.max(np.abs(ag))),
                **{f"DOF{i+1}_max_abs_disp": float(np.max(np.abs(u[:, i]))) for i in range(N_DOF)}
            })
            print(f"✅ 10DOF response saved: {resp_file} | nt={len(t)} | case={case_id}")

        df_case10 = pd.DataFrame(records)
        df_case10.to_csv(CASE10_CSV, index=False)
        pd.DataFrame(summaries).to_csv(CASE10_SUMMARY_CSV, index=False)
        print(f"✅ 已生成/覆盖 {CASE10_CSV}, {CASE10_SUMMARY_CSV}")

    df_case10 = _normalize_case10_dataframe(df_case10)
    cases = []
    for idx, row in df_case10.iterrows():
        case_id = str(row["case_id"])
        split = str(row["split"])
        eq_file = str(row["eq_file"])
        response_file = str(row["response_file"])
        if not os.path.exists(eq_file):
            raise FileNotFoundError(f"找不到 eq_file: {eq_file}")
        if not os.path.exists(response_file):
            raise FileNotFoundError(f"找不到 response_file: {response_file}")

        eq_time_np, eq_acc_np = load_eq_file(eq_file)
        resp_df = pd.read_csv(response_file)
        t_np = resp_df["Time"].values.astype(np.float64)
        t_np = t_np - t_np[0]
        dof_cols = [f"DOF{i+1}" for i in range(N_DOF)]
        missing = [c for c in dof_cols if c not in resp_df.columns]
        if missing:
            raise ValueError(f"{response_file} 缺少 10DOF 响应列: {missing}")
        u_np = resp_df[dof_cols].values.astype(np.float64)

        cases.append({
            "case_index": len(cases),
            "case_id": case_id,
            "split": split,
            "quake_group": infer_quake_group(case_id, eq_file),
            "eq_file": eq_file,
            "response_file": response_file,
            "eq_time_np": eq_time_np.astype(np.float64),
            "eq_acc_np": eq_acc_np.astype(np.float64),
            "t_np_raw": t_np.astype(np.float64),
            "u_np_raw": u_np.astype(np.float64),
        })

    print("=" * 100)
    print(f"10DOF cases loaded: {len(cases)}")
    print(pd.Series([c["split"] for c in cases]).value_counts().to_string())
    print("=" * 100)
    return cases


def infer_quake_group(case_id: str, eq_file: str) -> str:
    s = (str(case_id) + " " + str(eq_file)).lower()
    if "example" in s:
        return "example"
    if "far" in s or "far-field" in s or "far_f" in s:
        return "far"
    if "pulse" in s and ("no" not in s and "nopulse" not in s and "no_pulse" not in s):
        return "nearP"
    if "no pulse" in s or "nopulse" in s or "no_pulse" in s or "near-field record set - no" in s:
        return "nearNP"
    if "near" in s:
        return "near"
    return "unknown"


def downsample_time_response(t_np, u_np, max_points: int):
    n = len(t_np)
    if max_points is None or n <= max_points:
        return t_np.astype(np.float64), u_np.astype(np.float64)
    idx = np.linspace(0, n - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return t_np[idx].astype(np.float64), u_np[idx].astype(np.float64)


# =============================================================================
# 3. 读取 Top 布设
# =============================================================================

def load_top_layouts() -> List[Tuple[str, Tuple[int, ...]]]:
    mask_names = []
    source = "fallback"
    for p in TOP20_RANKING_CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if "mask_name" in df.columns:
                mask_names = list(df["mask_name"].astype(str).values)
                source = p
                break
    if not mask_names:
        mask_names = FALLBACK_TOP20_MASKS
    # 去重，保序
    seen = set()
    layouts = []
    for name in mask_names:
        if name in seen:
            continue
        seen.add(name)
        tup = parse_mask_name(name)
        if len(tup) == N_SENSOR and all(0 <= i < N_DOF for i in tup):
            layouts.append((sensor_tuple_to_name(tup), tup))
        if len(layouts) >= LABEL_TOPK_LAYOUTS:
            break
    if len(layouts) == 0:
        raise RuntimeError("没有读到合法 Top layouts。")
    print(f"✅ Loaded {len(layouts)} layouts from {source}")
    for k, (name, tup) in enumerate(layouts[:10], 1):
        print(f"   {k:02d}. {name} floors={[i+1 for i in tup]}")
    return layouts


# =============================================================================
# 4. 物理约束虚拟传感标签生成器
# =============================================================================

def make_noisy_sensor(u_true: np.ndarray, sensor_indices: List[int], noise_level: float, seed: int) -> Tuple[np.ndarray, Dict]:
    u_sensor_clean = u_true[:, sensor_indices].copy()
    if noise_level <= 0:
        return u_sensor_clean, {
            "sensor_noise_rmse": 0.0,
            "sensor_noise_nrmse": 0.0,
            "sensor_noise_max_abs": 0.0,
        }
    rng = np.random.default_rng(seed)
    std_per_sensor = np.std(u_sensor_clean, axis=0, keepdims=True)
    std_per_sensor = np.maximum(std_per_sensor, 1e-12)
    noise = rng.normal(loc=0.0, scale=noise_level * std_per_sensor, size=u_sensor_clean.shape)
    noisy = u_sensor_clean + noise
    noise_rmse = float(np.sqrt(np.mean(noise ** 2)))
    clean_rms = float(np.sqrt(np.mean(u_sensor_clean ** 2)))
    return noisy, {
        "sensor_noise_rmse": noise_rmse,
        "sensor_noise_nrmse": noise_rmse / (clean_rms + 1e-18),
        "sensor_noise_max_abs": float(np.max(np.abs(noise))),
    }


def spatial_interpolation_prior(u_sensor: np.ndarray, sensor_indices: List[int], n_dof: int) -> np.ndarray:
    """用传感器响应在楼层方向做线性插值/外推，作为未知 DOF 初值和弱先验。"""
    floors = np.arange(n_dof, dtype=np.float64)
    sidx = np.asarray(sensor_indices, dtype=np.float64)
    u_prior = np.zeros((u_sensor.shape[0], n_dof), dtype=np.float64)
    for it in range(u_sensor.shape[0]):
        # np.interp 对范围外使用端点值，适合做稳健初值。
        u_prior[it, :] = np.interp(floors, sidx, u_sensor[it, :])
    return u_prior


class DirectVirtualSensingOptimizer:
    def __init__(self,
                 t_np: np.ndarray,
                 u_true_np: np.ndarray,
                 eq_time_np: np.ndarray,
                 eq_acc_np: np.ndarray,
                 sensor_indices: List[int],
                 M_np: np.ndarray,
                 C_np: np.ndarray,
                 K_np: np.ndarray,
                 sensor_noise_level: float,
                 noise_seed: int,
                 adam_steps: int = LABEL_ADAM_STEPS,
                 adam_lr: float = LABEL_ADAM_LR,
                 use_lbfgs: bool = LABEL_USE_LBFGS,
                 lbfgs_max_iter: int = LABEL_LBFGS_MAX_ITER):
        self.t_np = t_np.astype(np.float64)
        self.u_true_np = u_true_np.astype(np.float64)
        self.sensor_indices = list(sensor_indices)
        self.sensor_set = set(sensor_indices)
        self.recon_indices = [i for i in range(N_DOF) if i not in self.sensor_set]
        self.noise_level = float(sensor_noise_level)
        self.noise_seed = int(noise_seed)
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.use_lbfgs = bool(use_lbfgs)
        self.lbfgs_max_iter = int(lbfgs_max_iter)

        self.u_sensor_np, self.noise_info = make_noisy_sensor(
            self.u_true_np, self.sensor_indices, self.noise_level, self.noise_seed
        )
        self.u_prior_full_np = spatial_interpolation_prior(self.u_sensor_np, self.sensor_indices, N_DOF)
        self.u_prior_unknown_np = self.u_prior_full_np[:, self.recon_indices]

        # scale 用传感器 RMS，符合部署时只知道传感器数据的设定。
        self.u_scale = float(np.sqrt(np.mean(self.u_sensor_np ** 2)))
        self.u_scale = max(self.u_scale, 1e-10)

        self.ag_np = np.interp(self.t_np, eq_time_np - eq_time_np[0], eq_acc_np).astype(np.float64)
        self.ag_scale = float(np.std(self.ag_np))
        self.ag_scale = max(self.ag_scale, 1e-10)

        self.dt = float(np.median(np.diff(self.t_np)))
        self.dt = max(self.dt, 1e-8)

        self.M = torch.tensor(M_np, dtype=torch.float32, device=DEVICE)
        self.C = torch.tensor(C_np, dtype=torch.float32, device=DEVICE)
        self.K = torch.tensor(K_np, dtype=torch.float32, device=DEVICE)
        self.force_vec = (self.M @ torch.ones((N_DOF, 1), dtype=torch.float32, device=DEVICE)).view(-1)
        self.u_sensor = torch.tensor(self.u_sensor_np, dtype=torch.float32, device=DEVICE)
        self.u_prior_unknown = torch.tensor(self.u_prior_unknown_np, dtype=torch.float32, device=DEVICE)
        self.ag = torch.tensor(self.ag_np, dtype=torch.float32, device=DEVICE).view(-1, 1)

        # 未知量参数化为 normalized correction，避免不同地震波幅值差异导致优化不稳。
        self.z = nn.Parameter(torch.zeros_like(self.u_prior_unknown, dtype=torch.float32, device=DEVICE))

        absK = torch.abs(self.K)
        absM = torch.abs(self.M)
        scale_vec = torch.ones((N_DOF,), dtype=torch.float32, device=DEVICE) * float(self.u_scale)
        eq_scale_vec = (absM @ torch.ones((N_DOF, 1), dtype=torch.float32, device=DEVICE)).view(-1) * float(self.ag_scale)
        self.res_scale = (absK @ scale_vec.view(-1, 1)).view(-1) + eq_scale_vec + 1.0
        self.history = []

    def assemble_u(self):
        u_unknown = self.u_prior_unknown + self.z * float(self.u_scale)
        u_full = torch.zeros((self.u_sensor.shape[0], N_DOF), dtype=torch.float32, device=DEVICE)
        u_full[:, self.sensor_indices] = self.u_sensor
        u_full[:, self.recon_indices] = u_unknown
        return u_full

    def finite_diff_physics_loss(self, u_full):
        if u_full.shape[0] < 5:
            return torch.tensor(0.0, device=DEVICE)
        dt = float(self.dt)
        u_mid = u_full[1:-1]
        u_dot = (u_full[2:] - u_full[:-2]) / (2.0 * dt)
        u_ddot = (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / (dt * dt)
        ag_mid = self.ag[1:-1]
        residual = (
            u_ddot @ self.M.T
            + u_dot @ self.C.T
            + u_mid @ self.K.T
            + ag_mid @ self.force_vec.view(1, -1)
        )
        residual_norm = residual / self.res_scale.view(1, -1)
        return torch.mean(residual_norm ** 2)

    def loss_parts(self):
        u_full = self.assemble_u()
        physics = self.finite_diff_physics_loss(u_full)
        u_unknown = u_full[:, self.recon_indices]
        prior = torch.mean(((u_unknown - self.u_prior_unknown) / float(self.u_scale)) ** 2)

        if u_unknown.shape[0] >= 3:
            d2 = u_unknown[2:] - 2.0 * u_unknown[1:-1] + u_unknown[:-2]
            smooth = torch.mean((d2 / float(self.u_scale)) ** 2)
        else:
            smooth = torch.tensor(0.0, device=DEVICE)

        # 空间平滑：沿 DOF 方向抑制不合理锯齿，但权重较小。
        spatial = torch.mean(((u_full[:, 1:] - u_full[:, :-1]) / float(self.u_scale)) ** 2)

        # 能量约束：未监测 RMS 不应远大于传感器 RMS。
        sensor_rms = torch.sqrt(torch.mean(self.u_sensor ** 2) + 1e-18)
        unknown_rms = torch.sqrt(torch.mean(u_unknown ** 2) + 1e-18)
        energy = F.relu(unknown_rms / (sensor_rms + 1e-12) - ENERGY_RATIO_LIMIT) ** 2

        # 初始条件：响应从静止附近开始；如果地震波初值不是 0，此项会自动很弱。
        u0 = u_full[0]
        if u_full.shape[0] >= 2:
            v0 = (u_full[1] - u_full[0]) / float(self.dt)
        else:
            v0 = torch.zeros_like(u0)
        ic = torch.mean((u0 / float(self.u_scale)) ** 2) + 0.01 * torch.mean((v0 / float(self.u_scale)) ** 2)

        total = (
            W_PHYSICS_LABEL * physics
            + W_PRIOR_LABEL * prior
            + W_SMOOTH_LABEL * smooth
            + W_SPATIAL_LABEL * spatial
            + W_ENERGY_LABEL * energy
            + W_IC_LABEL * ic
        )
        return {
            "total": total,
            "physics": physics,
            "prior": prior,
            "smooth": smooth,
            "spatial": spatial,
            "energy": energy,
            "ic": ic,
        }

    def optimize(self, verbose=False):
        opt = torch.optim.Adam([self.z], lr=self.adam_lr)
        for step in range(self.adam_steps):
            opt.zero_grad(set_to_none=True)
            parts = self.loss_parts()
            loss = parts["total"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"PC-VS label generation non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([self.z], 5.0)
            opt.step()
            if step == 0 or (step + 1) % 200 == 0 or step == self.adam_steps - 1:
                self.history.append({
                    "phase": "adam",
                    "step": step + 1,
                    **{k: float(v.detach().cpu().item()) for k, v in parts.items()}
                })
                if verbose:
                    print(f"  [label Adam] {step+1}/{self.adam_steps} total={parts['total'].item():.3e} p={parts['physics'].item():.3e}")

        if self.use_lbfgs and self.lbfgs_max_iter > 0:
            opt2 = SafeLBFGS([self.z], lr=1.0, max_iter=self.lbfgs_max_iter,
                                     line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-9, tolerance_change=1e-12)
            def closure():
                opt2.zero_grad(set_to_none=True)
                parts = self.loss_parts()
                loss = parts["total"]
                loss.backward()
                return loss
            opt2.step(closure)
            parts = self.loss_parts()
            self.history.append({
                "phase": "lbfgs",
                "step": self.lbfgs_max_iter,
                **{k: float(v.detach().cpu().item()) for k, v in parts.items()}
            })

        with torch.no_grad():
            u_pred = self.assemble_u().detach().cpu().numpy().astype(np.float32)
            parts = self.loss_parts()
            final_parts = {k: float(v.detach().cpu().item()) for k, v in parts.items()}
        return {
            "u_pred": u_pred,
            "loss_parts": final_parts,
            "history": self.history,
            "u_sensor_np": self.u_sensor_np.astype(np.float32),
            "u_prior_full_np": self.u_prior_full_np.astype(np.float32),
            "ag_np": self.ag_np.astype(np.float32),
            "u_scale": float(self.u_scale),
            "ag_scale": float(self.ag_scale),
            "dt": float(self.dt),
            "noise_info": self.noise_info,
        }


def build_label_cache(cases: List[Dict], layouts: List[Tuple[str, Tuple[int, ...]]], M_np, C_np, K_np) -> str:
    ensure_dir(LABEL_DIR)
    manifest_path = os.path.join(LABEL_DIR, "stage6_label_manifest.csv")

    existing_records = []
    existing_keys = set()
    if RESUME_LABEL_CACHE and os.path.exists(manifest_path):
        df_old = pd.read_csv(manifest_path)
        existing_records = df_old.to_dict("records")
        for _, row in df_old.iterrows():
            existing_keys.add((str(row["case_id"]), float(row["noise_level"]), str(row["mask_name"])))
        print(f"✅ Found existing label manifest: {manifest_path} | existing={len(existing_records)}")

    records = list(existing_records)
    hist_rows = []
    total = len(cases) * len(NOISE_LEVELS) * len(layouts)
    done = len(existing_keys)
    print(f"Stage6 label cache tasks: total={total}, existing={done}, remaining={total-done}")

    t0_all = pytime.time()
    task_id = 0
    for case in cases:
        t_ds, u_ds = downsample_time_response(case["t_np_raw"], case["u_np_raw"], LABEL_MAX_POINTS)
        for noise_level in NOISE_LEVELS:
            for layout_name, sensor_tuple in layouts:
                task_id += 1
                key = (case["case_id"], float(noise_level), layout_name)
                if key in existing_keys:
                    continue
                sensor_indices = list(sensor_tuple)
                recon_indices = [i for i in range(N_DOF) if i not in set(sensor_indices)]
                combo_id = sum([(i + 1) * (10 ** k) for k, i in enumerate(sensor_indices)])
                noise_int = int(round(noise_level * 10000))
                noise_seed = NOISE_SEED_BASE + case["case_index"] * 100000 + combo_id * 1000 + noise_int
                set_all_seeds(BASE_SEED + case["case_index"] * 100000 + combo_id * 100 + noise_int)

                label_file = os.path.join(
                    LABEL_DIR,
                    f"{safe_name(case['case_id'])}_{noise_tag(noise_level)}_{layout_name}.npz"
                )
                print(f"\n[{task_id}/{total}] label: case={case['case_id']} split={case['split']} noise={noise_level:.2%} mask={layout_name}")

                opt = DirectVirtualSensingOptimizer(
                    t_np=t_ds,
                    u_true_np=u_ds,
                    eq_time_np=case["eq_time_np"],
                    eq_acc_np=case["eq_acc_np"],
                    sensor_indices=sensor_indices,
                    M_np=M_np,
                    C_np=C_np,
                    K_np=K_np,
                    sensor_noise_level=noise_level,
                    noise_seed=noise_seed,
                    adam_steps=LABEL_ADAM_STEPS,
                    adam_lr=LABEL_ADAM_LR,
                    use_lbfgs=LABEL_USE_LBFGS,
                    lbfgs_max_iter=LABEL_LBFGS_MAX_ITER,
                )
                t_start = pytime.time()
                result = opt.optimize(verbose=False)
                elapsed = pytime.time() - t_start
                u_teacher = result["u_pred"]
                teacher_metrics = calc_metrics(u_ds, u_teacher, sensor_indices)

                np.savez_compressed(
                    label_file,
                    time=t_ds.astype(np.float32),
                    u_true=u_ds.astype(np.float32),
                    u_teacher=u_teacher.astype(np.float32),
                    u_sensor=result["u_sensor_np"].astype(np.float32),
                    u_prior=result["u_prior_full_np"].astype(np.float32),
                    ag=result["ag_np"].astype(np.float32),
                    sensor_indices=np.array(sensor_indices, dtype=np.int64),
                    recon_indices=np.array(recon_indices, dtype=np.int64),
                    u_scale=np.array([result["u_scale"]], dtype=np.float32),
                    ag_scale=np.array([result["ag_scale"]], dtype=np.float32),
                    dt=np.array([result["dt"]], dtype=np.float32),
                )

                rec = {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "quake_group": case["quake_group"],
                    "noise_level": float(noise_level),
                    "noise_pct": float(noise_level * 100.0),
                    "noise_tag": noise_tag(noise_level),
                    "mask_name": layout_name,
                    "sensor_indices_0based": str(sensor_indices),
                    "sensor_floors_1based": str([i + 1 for i in sensor_indices]),
                    "reconstructed_indices_0based": str(recon_indices),
                    "reconstructed_floors_1based": str([i + 1 for i in recon_indices]),
                    "label_npz": label_file.replace("\\", "/"),
                    "n_time": len(t_ds),
                    "duration": float(t_ds[-1] - t_ds[0]),
                    "u_scale": result["u_scale"],
                    "ag_scale": result["ag_scale"],
                    "dt": result["dt"],
                    "teacher_nrmse_unmonitored_vs_true": teacher_metrics["nrmse_unmonitored"],
                    "teacher_worst_reconstructed_R2_vs_true": teacher_metrics["worst_reconstructed_R2"],
                    "teacher_mean_reconstructed_R2_vs_true": teacher_metrics["mean_reconstructed_R2"],
                    "teacher_true_p_fd": result["loss_parts"]["physics"],
                    "sensor_noise_nrmse": result["noise_info"]["sensor_noise_nrmse"],
                    "label_elapsed_sec": elapsed,
                }
                records.append(rec)
                existing_keys.add(key)

                for h in result["history"]:
                    hist_rows.append({
                        "case_id": case["case_id"],
                        "noise_level": float(noise_level),
                        "mask_name": layout_name,
                        **h,
                    })

                print(
                    f"  saved label | teacher unmon NRMSE={teacher_metrics['nrmse_unmonitored']:.4e} "
                    f"worstR2={teacher_metrics['worst_reconstructed_R2']:.4f} "
                    f"p={result['loss_parts']['physics']:.3e} | {elapsed:.1f}s"
                )

                if SAVE_LABEL_INCREMENTAL:
                    pd.DataFrame(records).to_csv(manifest_path, index=False)
                    if hist_rows:
                        hist_path = os.path.join(LABEL_DIR, "stage6_label_optimizer_history.csv")
                        if os.path.exists(hist_path):
                            old_hist = pd.read_csv(hist_path)
                            pd.concat([old_hist, pd.DataFrame(hist_rows)], ignore_index=True).to_csv(hist_path, index=False)
                        else:
                            pd.DataFrame(hist_rows).to_csv(hist_path, index=False)
                        hist_rows = []

    pd.DataFrame(records).to_csv(manifest_path, index=False)
    if hist_rows:
        hist_path = os.path.join(LABEL_DIR, "stage6_label_optimizer_history.csv")
        if os.path.exists(hist_path):
            old_hist = pd.read_csv(hist_path)
            pd.concat([old_hist, pd.DataFrame(hist_rows)], ignore_index=True).to_csv(hist_path, index=False)
        else:
            pd.DataFrame(hist_rows).to_csv(hist_path, index=False)

    print(f"\n✅ Stage6 label cache ready: {manifest_path} | records={len(records)} | elapsed={(pytime.time()-t0_all)/60:.1f} min")
    return manifest_path


# =============================================================================
# 5. Stage6 Dataset
# =============================================================================

class Stage6LabelStore:
    def __init__(self, manifest_csv: str):
        self.df = pd.read_csv(manifest_csv)
        if len(self.df) == 0:
            raise RuntimeError("label manifest 为空。")
        # 确保文件存在
        missing = [p for p in self.df["label_npz"].astype(str).values if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"有 {len(missing)} 个 label_npz 不存在。例: {missing[:3]}")
        self.cache: Dict[int, Dict] = {}

    def __len__(self):
        return len(self.df)

    def get(self, idx: int) -> Dict:
        if idx in self.cache:
            return self.cache[idx]
        row = self.df.iloc[idx]
        data = np.load(str(row["label_npz"]), allow_pickle=False)
        item = {
            "idx": idx,
            "case_id": str(row["case_id"]),
            "split": str(row["split"]),
            "quake_group": str(row["quake_group"]),
            "noise_level": float(row["noise_level"]),
            "mask_name": str(row["mask_name"]),
            "time": data["time"].astype(np.float32),
            "u_true": data["u_true"].astype(np.float32),
            "u_teacher": data["u_teacher"].astype(np.float32),
            "u_sensor": data["u_sensor"].astype(np.float32),
            "ag": data["ag"].astype(np.float32),
            "sensor_indices": data["sensor_indices"].astype(np.int64).tolist(),
            "recon_indices": data["recon_indices"].astype(np.int64).tolist(),
            "u_scale": float(data["u_scale"][0]),
            "ag_scale": float(data["ag_scale"][0]),
            "dt": float(data["dt"][0]),
            "teacher_nrmse_unmonitored_vs_true": float(row.get("teacher_nrmse_unmonitored_vs_true", np.nan)),
            "teacher_worst_R2_vs_true": float(row.get("teacher_worst_R2_vs_true", np.nan)),
        }
        self.cache[idx] = item
        return item


def build_feature_arrays(item: Dict):
    time = item["time"]
    T = len(time)
    u_teacher = item["u_teacher"]
    u_true = item["u_true"]
    ag = item["ag"]
    sensor_indices = item["sensor_indices"]
    sensor_set = set(sensor_indices)
    mask = np.zeros((N_DOF,), dtype=np.float32)
    mask[sensor_indices] = 1.0

    # sensor_full 采用 teacher 中的传感器通道；它等于 noisy sensor 硬约束值。
    sensor_full = np.zeros((T, N_DOF), dtype=np.float32)
    sensor_full[:, sensor_indices] = u_teacher[:, sensor_indices]

    u_scale = max(float(item["u_scale"]), 1e-10)
    ag_scale = max(float(item["ag_scale"]), 1e-10)

    # Stage6-v3 关键：先用传感器响应构造空间插值先验，再让网络只学习 teacher-prior 的残差。
    # 这样网络不再从零预测完整响应，而是学习 PC-VS 对传统插值先验的物理修正。
    u_prior = spatial_interpolation_prior(sensor_full[:, sensor_indices], sensor_indices, N_DOF).astype(np.float32)

    sensor_norm = sensor_full / u_scale
    prior_norm = u_prior / u_scale
    teacher_norm = u_teacher / u_scale
    true_norm = u_true / u_scale
    ag_norm = (ag / ag_scale).reshape(T, 1).astype(np.float32)

    t_norm = ((time - time[0]) / (time[-1] - time[0] + 1e-12)).reshape(T, 1).astype(np.float32)
    t_sin = np.sin(2.0 * np.pi * t_norm).astype(np.float32)
    t_cos = np.cos(2.0 * np.pi * t_norm).astype(np.float32)
    mask_broadcast = np.repeat(mask.reshape(1, -1), T, axis=0)

    # input: time_norm, sin, cos, ag_norm, sensor_norm(10), prior_norm(10), mask(10)
    x = np.concatenate([t_norm, t_sin, t_cos, ag_norm, sensor_norm, prior_norm, mask_broadcast], axis=1).astype(np.float32)

    teacher_q = 1.0
    if USE_TEACHER_QUALITY_WEIGHT:
        tn = float(item.get("teacher_nrmse_unmonitored_vs_true", np.nan))
        if np.isfinite(tn):
            teacher_q = 1.0 / (1.0 + (tn / TEACHER_QUALITY_REF_NRMSE) ** 2)
            teacher_q = max(float(TEACHER_QUALITY_MIN_WEIGHT), min(1.0, teacher_q))

    return {
        "x": x,
        "teacher_norm": teacher_norm.astype(np.float32),
        "true_norm": true_norm.astype(np.float32),
        "sensor_norm": sensor_norm.astype(np.float32),
        "prior_norm": prior_norm.astype(np.float32),
        "mask": mask.astype(np.float32),
        "ag_norm": ag_norm.astype(np.float32),
        "u_scale": np.array([u_scale], dtype=np.float32),
        "ag_scale": np.array([ag_scale], dtype=np.float32),
        "dt": np.array([item["dt"]], dtype=np.float32),
        "teacher_quality_weight": np.array([teacher_q], dtype=np.float32),
    }


class Stage6WindowDataset(Dataset):
    def __init__(self, store: Stage6LabelStore, indices: List[int], window_len: int, windows_per_sequence: int, seed: int = 0):
        self.store = store
        self.indices = list(indices)
        self.window_len = int(window_len)
        self.windows_per_sequence = int(windows_per_sequence)
        self.rng = np.random.default_rng(seed)
        self.length = len(self.indices) * self.windows_per_sequence

    def __len__(self):
        return self.length

    def __getitem__(self, n):
        seq_idx = self.indices[n % len(self.indices)]
        item = self.store.get(seq_idx)
        arrays = build_feature_arrays(item)
        T = arrays["x"].shape[0]
        L = min(self.window_len, T)
        if T == L:
            start = 0
        else:
            start = int(self.rng.integers(0, T - L + 1))
        end = start + L

        # 若最后不足 window_len，padding 到固定长度。通常不会发生，因为 T>=900。
        def slc(a):
            out = a[start:end]
            if out.shape[0] < self.window_len:
                pad = np.repeat(out[-1:], self.window_len - out.shape[0], axis=0)
                out = np.concatenate([out, pad], axis=0)
            return out.astype(np.float32)

        return {
            "x": torch.from_numpy(slc(arrays["x"])),
            "teacher_norm": torch.from_numpy(slc(arrays["teacher_norm"])),
            "true_norm": torch.from_numpy(slc(arrays["true_norm"])),
            "sensor_norm": torch.from_numpy(slc(arrays["sensor_norm"])),
            "prior_norm": torch.from_numpy(slc(arrays["prior_norm"])),
            "mask": torch.from_numpy(arrays["mask"]),
            "ag_norm": torch.from_numpy(slc(arrays["ag_norm"])),
            "u_scale": torch.from_numpy(arrays["u_scale"]),
            "ag_scale": torch.from_numpy(arrays["ag_scale"]),
            "dt": torch.from_numpy(arrays["dt"]),
            "teacher_quality_weight": torch.from_numpy(arrays["teacher_quality_weight"]),
        }


# =============================================================================
# 6. PIKAN/FKAN-style 模型
# =============================================================================

class FastKANLinear(nn.Module):
    """轻量 FastKAN-style RBF layer，用作点态非线性映射。

    不是依赖外部 AIStructDynSolve 的 FKAN，而是自包含实现，便于直接运行。
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 8, grid_min: float = -2.0, grid_max: float = 2.0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        grid = torch.linspace(grid_min, grid_max, grid_size)
        self.register_buffer("grid", grid)
        self.base = nn.Linear(in_features, out_features)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.02)
        self.spline_bias = nn.Parameter(torch.zeros(out_features))
        self.h = float((grid_max - grid_min) / max(grid_size - 1, 1))

    def forward(self, x):
        # x: [..., in_features]
        base_out = self.base(F.silu(x))
        # RBF: [..., in_features, grid_size]
        xg = x.unsqueeze(-1) - self.grid.view(*([1] * (x.ndim - 1)), 1, self.grid_size)
        rbf = torch.exp(-0.5 * (xg / (self.h + 1e-6)) ** 2)
        spline = torch.einsum("...ig,oig->...o", rbf, self.spline_weight) + self.spline_bias
        return base_out + spline


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.dropout(y)
        return F.gelu(x + y)


class PIKANFKANFastReconstructor(nn.Module):
    def __init__(self, input_dim: int, n_dof: int, hidden: int = 128, grid_size: int = 8,
                 num_blocks: int = 4, dropout: float = 0.05):
        super().__init__()
        self.input_kan = FastKANLinear(input_dim, hidden, grid_size=grid_size)
        blocks = []
        dilations = [1, 2, 4, 8, 1, 2]
        for i in range(num_blocks):
            blocks.append(TemporalResidualBlock(hidden, kernel_size=5, dilation=dilations[i % len(dilations)], dropout=dropout))
        self.temporal = nn.Sequential(*blocks)
        self.out_kan = FastKANLinear(hidden, n_dof, grid_size=grid_size)

    def forward(self, x, mask, sensor_norm, prior_norm):
        # x: [B,T,F], mask: [B,D], sensor_norm/prior_norm: [B,T,D]
        h = self.input_kan(x)              # [B,T,H]
        h = h.transpose(1, 2).contiguous() # [B,H,T]
        h = self.temporal(h)
        h = h.transpose(1, 2).contiguous() # [B,T,H]
        delta = self.out_kan(h)            # [B,T,D], residual in normalized displacement domain
        mask2 = mask.unsqueeze(1)
        # Stage6-v3: reconstructed DOF = prior + learned residual; sensor DOF hard constrained.
        raw = prior_norm + delta
        out = mask2 * sensor_norm + (1.0 - mask2) * raw
        return out


# =============================================================================
# 7. 训练 loss 与评估
# =============================================================================

def batch_physics_loss(pred_norm, ag_norm, mask, M_t, C_t, K_t, dt, u_scale, ag_scale):
    # pred_norm: [B,T,D], ag_norm: [B,T,1]
    B, T, D = pred_norm.shape
    if T < 5:
        return torch.tensor(0.0, device=pred_norm.device)
    # 还原物理量
    u = pred_norm * u_scale.view(B, 1, 1)
    ag = ag_norm * ag_scale.view(B, 1, 1)
    dtv = torch.clamp(dt.view(B, 1, 1), min=1e-8)
    u_mid = u[:, 1:-1, :]
    u_dot = (u[:, 2:, :] - u[:, :-2, :]) / (2.0 * dtv)
    u_ddot = (u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]) / (dtv * dtv)
    ag_mid = ag[:, 1:-1, :]

    # residual [B,T-2,D]
    residual = (
        torch.einsum("btd,ed->bte", u_ddot, M_t)
        + torch.einsum("btd,ed->bte", u_dot, C_t)
        + torch.einsum("btd,ed->bte", u_mid, K_t)
        + ag_mid * (M_t @ torch.ones((D, 1), device=pred_norm.device)).view(1, 1, D)
    )
    # normalize scale per sample and dof
    absK = torch.abs(K_t)
    absM = torch.abs(M_t)
    scale_vec = torch.ones((B, D), device=pred_norm.device) * u_scale.view(B, 1)
    eq_vec = (absM @ torch.ones((D, 1), device=pred_norm.device)).view(1, D) * ag_scale.view(B, 1)
    res_scale = torch.einsum("ed,bd->be", absK, scale_vec) + eq_vec + 1.0
    residual_norm = residual / res_scale.view(B, 1, D)
    return torch.mean(residual_norm ** 2)


def training_loss(batch, model, M_t, C_t, K_t):
    x = batch["x"].to(DEVICE)
    teacher = batch["teacher_norm"].to(DEVICE)
    sensor = batch["sensor_norm"].to(DEVICE)
    prior = batch["prior_norm"].to(DEVICE)
    mask = batch["mask"].to(DEVICE)
    ag = batch["ag_norm"].to(DEVICE)
    u_scale = batch["u_scale"].to(DEVICE).view(-1)
    ag_scale = batch["ag_scale"].to(DEVICE).view(-1)
    dt = batch["dt"].to(DEVICE).view(-1)

    pred = model(x, mask, sensor, prior)
    recon_mask = 1.0 - mask.unsqueeze(1)

    # Stage6-v2: teacher fitting 是主目标；按样本质量加权，避免弱 teacher 主导训练。
    q = batch.get("teacher_quality_weight", torch.ones((pred.shape[0], 1), device=DEVICE)).to(DEVICE).view(-1)
    q = torch.clamp(q, min=TEACHER_QUALITY_MIN_WEIGHT, max=1.0)
    err2 = (pred - teacher) ** 2
    # residual-to-prior 辅助诊断/轻微正则：学习 teacher-prior 的修正量，而不是无约束输出。
    # 这里不把 residual 正则设太强，避免限制模型逼近 teacher。
    residual_pred = (pred - prior) * recon_mask
    residual_teacher = (teacher - prior) * recon_mask
    residual_fit = torch.mean((residual_pred - residual_teacher) ** 2)
    recon_dof_count = torch.sum(1.0 - mask, dim=1).clamp(min=1.0)
    per_b_unmon = torch.sum(err2 * recon_mask, dim=(1, 2)) / (recon_dof_count * pred.shape[1] + 1e-9)
    per_b_all = torch.mean(err2, dim=(1, 2))
    teacher_unmon = torch.sum(q * per_b_unmon) / (torch.sum(q) + 1e-9)
    teacher_all = torch.sum(q * per_b_all) / (torch.sum(q) + 1e-9)

    p_loss = batch_physics_loss(pred, ag, mask, M_t, C_t, K_t, dt, u_scale, ag_scale)
    if pred.shape[1] >= 3:
        d2 = pred[:, 2:, :] - 2.0 * pred[:, 1:-1, :] + pred[:, :-2, :]
        smooth = torch.mean((d2 * recon_mask[:, :, :]) ** 2)
    else:
        smooth = torch.tensor(0.0, device=DEVICE)

    total = (
        L_TEACHER_UNMON * teacher_unmon
        + L_TEACHER_ALL * teacher_all
        + 0.25 * residual_fit
        + L_PHYSICS * p_loss
        + L_SMOOTH * smooth
    )
    return total, {
        "loss": float(total.detach().cpu().item()),
        "teacher_unmon": float(teacher_unmon.detach().cpu().item()),
        "teacher_all": float(teacher_all.detach().cpu().item()),
        "residual_fit": float(residual_fit.detach().cpu().item()),
        "physics": float(p_loss.detach().cpu().item()),
        "smooth": float(smooth.detach().cpu().item()),
        "teacher_q_mean": float(q.detach().mean().cpu().item()),
    }


@torch.no_grad()
def predict_full_sequence(model, item: Dict):
    arrays = build_feature_arrays(item)
    x = torch.from_numpy(arrays["x"]).unsqueeze(0).to(DEVICE)
    sensor = torch.from_numpy(arrays["sensor_norm"]).unsqueeze(0).to(DEVICE)
    prior = torch.from_numpy(arrays["prior_norm"]).unsqueeze(0).to(DEVICE)
    mask = torch.from_numpy(arrays["mask"]).unsqueeze(0).to(DEVICE)
    pred_norm = model(x, mask, sensor, prior).squeeze(0).detach().cpu().numpy()
    pred = pred_norm * float(arrays["u_scale"][0])
    return pred.astype(np.float32)


def finite_diff_true_p_numpy(u_full: np.ndarray, ag: np.ndarray, dt: float, M_np, C_np, K_np) -> float:
    if len(u_full) < 5:
        return np.nan
    u_mid = u_full[1:-1]
    u_dot = (u_full[2:] - u_full[:-2]) / (2.0 * dt)
    u_ddot = (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / (dt * dt)
    force_vec = (M_np @ np.ones((N_DOF, 1))).reshape(-1)
    residual = u_ddot @ M_np.T + u_dot @ C_np.T + u_mid @ K_np.T + ag[1:-1, None] @ force_vec.reshape(1, -1)
    scale_u = max(float(np.sqrt(np.mean(u_full ** 2))), 1e-10)
    scale_ag = max(float(np.std(ag)), 1e-10)
    res_scale = (np.abs(K_np) @ (np.ones(N_DOF) * scale_u).reshape(-1, 1)).reshape(-1) + (np.abs(M_np) @ np.ones((N_DOF, 1))).reshape(-1) * scale_ag + 1.0
    residual_norm = residual / res_scale.reshape(1, -1)
    return float(np.mean(residual_norm ** 2))


def evaluate_model(model, store: Stage6LabelStore, indices: List[int], M_np, C_np, K_np, split_name: str):
    model.eval()
    rows = []
    dof_rows = []
    pred_save_count = 0
    pred_dir = os.path.join(OUT_DIR, "eval_predictions")
    ensure_dir(pred_dir)

    for idx in indices:
        item = store.get(idx)
        pred = predict_full_sequence(model, item)
        teacher = item["u_teacher"]
        true = item["u_true"]
        sensor_indices = item["sensor_indices"]
        recon_indices = item["recon_indices"]

        m_teacher = calc_metrics(teacher, pred, sensor_indices)
        m_true = calc_metrics(true, pred, sensor_indices)
        m_teacher_vs_true = calc_metrics(true, teacher, sensor_indices)
        p_model = finite_diff_true_p_numpy(pred, item["ag"], item["dt"], M_np, C_np, K_np)

        pred_path = ""
        if SAVE_EVAL_PREDICTIONS and pred_save_count < MAX_SAVE_EVAL_PRED_PER_SPLIT:
            pred_path = os.path.join(pred_dir, f"{safe_name(item['case_id'])}_{noise_tag(item['noise_level'])}_{item['mask_name']}_model_pred.csv")
            dfp = pd.DataFrame({"Time": item["time"]})
            for j in range(N_DOF):
                dfp[f"DOF{j+1}_True"] = true[:, j]
                dfp[f"DOF{j+1}_TeacherPCVS"] = teacher[:, j]
                dfp[f"DOF{j+1}_ModelPred"] = pred[:, j]
                dfp[f"DOF{j+1}_Role"] = "sensor" if j in set(sensor_indices) else "reconstructed"
            dfp.to_csv(pred_path, index=False)
            pred_save_count += 1

        row = {
            "idx": idx,
            "case_id": item["case_id"],
            "split": item["split"],
            "quake_group": item["quake_group"],
            "noise_level": item["noise_level"],
            "noise_pct": item["noise_level"] * 100.0,
            "mask_name": item["mask_name"],
            "sensor_floors_1based": str([i + 1 for i in sensor_indices]),
            "reconstructed_floors_1based": str([i + 1 for i in recon_indices]),
            "model_vs_teacher_unmon_nrmse": m_teacher["nrmse_unmonitored"],
            "model_vs_teacher_worst_R2": m_teacher["worst_reconstructed_R2"],
            "model_vs_true_unmon_nrmse": m_true["nrmse_unmonitored"],
            "model_vs_true_worst_R2": m_true["worst_reconstructed_R2"],
            "model_vs_true_mean_R2": m_true["mean_reconstructed_R2"],
            "teacher_vs_true_unmon_nrmse": m_teacher_vs_true["nrmse_unmonitored"],
            "teacher_vs_true_worst_R2": m_teacher_vs_true["worst_reconstructed_R2"],
            "model_true_p_fd": p_model,
            "prediction_csv": pred_path,
        }
        rows.append(row)

        for j in range(N_DOF):
            dof_rows.append({
                "idx": idx,
                "case_id": item["case_id"],
                "split": item["split"],
                "noise_level": item["noise_level"],
                "mask_name": item["mask_name"],
                "DOF": j + 1,
                "role": "sensor" if j in set(sensor_indices) else "reconstructed",
                "model_vs_true_R2": m_true["r2_per_dof"][j],
                "model_vs_true_RMSE": m_true["rmse_per_dof"][j],
                "teacher_vs_true_R2": m_teacher_vs_true["r2_per_dof"][j],
                "teacher_vs_true_RMSE": m_teacher_vs_true["rmse_per_dof"][j],
            })
    return pd.DataFrame(rows), pd.DataFrame(dof_rows)


def aggregate_eval(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    out = df.groupby(group_cols, dropna=False).agg(
        n_samples=("idx", "count"),
        mean_model_vs_true_nrmse=("model_vs_true_unmon_nrmse", "mean"),
        worst_model_vs_true_nrmse=("model_vs_true_unmon_nrmse", "max"),
        mean_model_vs_true_worst_R2=("model_vs_true_worst_R2", "mean"),
        worst_model_vs_true_worst_R2=("model_vs_true_worst_R2", "min"),
        mean_model_vs_teacher_nrmse=("model_vs_teacher_unmon_nrmse", "mean"),
        mean_teacher_vs_true_nrmse=("teacher_vs_true_unmon_nrmse", "mean"),
        mean_model_true_p_fd=("model_true_p_fd", "mean"),
    ).reset_index()
    out = out.sort_values(["mean_model_vs_true_nrmse", "worst_model_vs_true_nrmse"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# =============================================================================
# 8. 训练主流程
# =============================================================================

def split_indices(store: Stage6LabelStore):
    df = store.df
    train_idx = df.index[df["split"].astype(str).str.lower() == "train"].tolist()
    val_idx = df.index[df["split"].astype(str).str.lower() == "val"].tolist()
    test_idx = df.index[df["split"].astype(str).str.lower() == "test"].tolist()
    if len(train_idx) == 0:
        train_idx = df.index.tolist()
    if len(val_idx) == 0:
        val_idx = train_idx[:max(1, len(train_idx)//5)]
    return train_idx, val_idx, test_idx


def train_stage6_model(manifest_csv: str, M_np, C_np, K_np):
    ensure_dir(CHECKPOINT_DIR)
    store = Stage6LabelStore(manifest_csv)
    train_idx, val_idx, test_idx = split_indices(store)
    print("=" * 100)
    print("Stage6 training split")
    print(f"train samples: {len(train_idx)} | val samples: {len(val_idx)} | test samples: {len(test_idx)}")
    print("=" * 100)

    train_ds = Stage6WindowDataset(store, train_idx, WINDOW_LEN, WINDOWS_PER_SEQUENCE_PER_EPOCH, seed=BASE_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)

    # input_dim = time_norm, sin, cos, ag_norm, sensor_norm(D), prior_norm(D), mask(D)
    # Stage6-v3 的特征向量实际包含 4 个标量时间/地震动特征 + 3 组 DOF 通道，
    # 即 1+2+1 + sensor_norm(D) + prior_norm(D) + mask(D)。
    # 旧版少算了 prior_norm(D)，导致输入为 34 维但模型按 24 维初始化。
    input_dim = 1 + 2 + 1 + N_DOF + N_DOF + N_DOF
    model = PIKANFKANFastReconstructor(
        input_dim=input_dim,
        n_dof=N_DOF,
        hidden=MODEL_HIDDEN,
        grid_size=MODEL_GRID_SIZE,
        num_blocks=MODEL_NUM_BLOCKS,
        dropout=MODEL_DROPOUT,
    ).to(DEVICE)

    M_t = torch.tensor(M_np, dtype=torch.float32, device=DEVICE)
    C_t = torch.tensor(C_np, dtype=torch.float32, device=DEVICE)
    K_t = torch.tensor(K_np, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5)

    best_val = float("inf")
    best_path = os.path.join(CHECKPOINT_DIR, "best_stage6_model.pt")
    log_rows = []
    bad_count = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = pytime.time()
        sums = {"loss": 0.0, "teacher_unmon": 0.0, "teacher_all": 0.0, "physics": 0.0, "smooth": 0.0}
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, parts = training_loss(batch, model, M_t, C_t, K_t)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Stage6 training non-finite loss at epoch {epoch}")
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            for k in sums:
                sums[k] += parts[k]
            n_batches += 1

        train_parts = {k: sums[k] / max(n_batches, 1) for k in sums}

        # 每若干 epoch 做全 val 评估；为了稳，epoch 1 和每 5 epoch 都评估。
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            val_df, _ = evaluate_model(model, store, val_idx, M_np, C_np, K_np, "val")
            val_metric = float(val_df["model_vs_teacher_unmon_nrmse"].mean())
            val_true_metric = float(val_df["model_vs_true_unmon_nrmse"].mean())
            val_teacher_metric = val_metric
        else:
            val_metric = best_val
            val_teacher_metric = np.nan

        scheduler.step(val_metric)
        elapsed = pytime.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch,
            "lr": lr_now,
            "elapsed_sec": elapsed,
            **{f"train_{k}": v for k, v in train_parts.items()},
            "val_model_vs_true_unmon_nrmse": val_true_metric,
            "val_model_vs_teacher_unmon_nrmse": val_teacher_metric,
        }
        log_rows.append(log_row)
        pd.DataFrame(log_rows).to_csv(os.path.join(OUT_DIR, "train_log.csv"), index=False)

        print(
            f"epoch {epoch:04d}/{EPOCHS} | "
            f"train_loss={train_parts['loss']:.4e} | train_teacher_unmon={train_parts['teacher_unmon']:.4e} | "
            f"train_p={train_parts['physics']:.3e} | q={train_parts.get('teacher_q_mean', 1.0):.2f} | val_true_nrmse={val_metric:.4e} | "
            f"val_teacher_nrmse={val_teacher_metric:.4e} | val_true_nrmse={val_true_metric:.4e} | lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        if val_metric < best_val - 1e-6:
            best_val = val_metric
            bad_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "N_DOF": N_DOF,
                    "N_SENSOR": N_SENSOR,
                    "input_dim": input_dim,
                    "hidden": MODEL_HIDDEN,
                    "grid_size": MODEL_GRID_SIZE,
                    "num_blocks": MODEL_NUM_BLOCKS,
                    "dropout": MODEL_DROPOUT,
                },
                "best_val_model_vs_teacher_unmon_nrmse": best_val,
            }, best_path)
            print(f"  ✅ saved best model: {best_path}")
        else:
            bad_count += 1
            if bad_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}, best_val={best_val:.4e}")
                break

    # 加载 best 做最终评估
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_eval_rows = []
    all_dof_rows = []
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(indices) == 0:
            continue
        df_eval, df_dof = evaluate_model(model, store, indices, M_np, C_np, K_np, split_name)
        all_eval_rows.append(df_eval)
        all_dof_rows.append(df_dof)
        print(f"[{split_name}] model_vs_true mean unmon NRMSE = {df_eval['model_vs_true_unmon_nrmse'].mean():.4e}")

    df_all = pd.concat(all_eval_rows, ignore_index=True)
    df_dof_all = pd.concat(all_dof_rows, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "eval_per_sample.csv"), index=False)
    df_dof_all.to_csv(os.path.join(OUT_DIR, "eval_per_dof.csv"), index=False)

    agg_split = aggregate_eval(df_all, ["split"])
    agg_mask = aggregate_eval(df_all, ["mask_name"])
    agg_noise = aggregate_eval(df_all, ["noise_level"])
    agg_split_mask = aggregate_eval(df_all, ["split", "mask_name"])
    agg_noise_mask = aggregate_eval(df_all, ["noise_level", "mask_name"])

    agg_split.to_csv(os.path.join(OUT_DIR, "eval_aggregate_by_split.csv"), index=False)
    agg_mask.to_csv(os.path.join(OUT_DIR, "eval_aggregate_by_mask.csv"), index=False)
    agg_noise.to_csv(os.path.join(OUT_DIR, "eval_aggregate_by_noise.csv"), index=False)
    agg_split_mask.to_csv(os.path.join(OUT_DIR, "eval_aggregate_by_split_mask.csv"), index=False)
    agg_noise_mask.to_csv(os.path.join(OUT_DIR, "eval_aggregate_by_noise_mask.csv"), index=False)

    # summary
    overall_mean = float(df_all["model_vs_true_unmon_nrmse"].mean())
    overall_worst = float(df_all["model_vs_true_unmon_nrmse"].max())
    teacher_mean = float(df_all["teacher_vs_true_unmon_nrmse"].mean())
    mimic_mean = float(df_all["model_vs_teacher_unmon_nrmse"].mean())
    best_mask = agg_mask.iloc[0]["mask_name"] if len(agg_mask) else "NA"

    # 额外输出：排除 example1 后的统计，用来判断该单条自有地震波是否为 OOD/outlier。
    df_no_example = df_all[~df_all["case_id"].astype(str).str.contains("example1", case=False, na=False)].copy()
    if len(df_no_example) > 0:
        noex_model_true = float(df_no_example["model_vs_true_unmon_nrmse"].mean())
        noex_teacher_true = float(df_no_example["teacher_vs_true_unmon_nrmse"].mean())
        noex_model_teacher = float(df_no_example["model_vs_teacher_unmon_nrmse"].mean())
        df_no_example.to_csv(os.path.join(OUT_DIR, "eval_per_sample_exclude_example1.csv"), index=False)
    else:
        noex_model_true = np.nan
        noex_teacher_true = np.nan
        noex_model_teacher = np.nan

    summary_path = os.path.join(OUT_DIR, "stage6_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Stage 6-v2 Top20 teacher-first PIKAN/FKAN-style fast reconstructor summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"label_manifest: {manifest_csv}\n")
        f.write(f"best_model: {best_path}\n")
        f.write(f"overall model_vs_true mean unmon NRMSE: {overall_mean:.6e}\n")
        f.write(f"overall model_vs_true worst unmon NRMSE: {overall_worst:.6e}\n")
        f.write(f"overall teacher_vs_true mean unmon NRMSE: {teacher_mean:.6e}\n")
        f.write(f"overall model_vs_teacher mean unmon NRMSE: {mimic_mean:.6e}\n")
        f.write(f"exclude_example1 model_vs_true mean unmon NRMSE: {noex_model_true:.6e}\n")
        f.write(f"exclude_example1 teacher_vs_true mean unmon NRMSE: {noex_teacher_true:.6e}\n")
        f.write(f"exclude_example1 model_vs_teacher mean unmon NRMSE: {noex_model_teacher:.6e}\n")
        f.write(f"best mask by model eval aggregate: {best_mask}\n")
        f.write("\n主线说明：Stage6-v3 residual-to-prior 的目标不是重新寻找最优布设，而是更准确地学习 PC-VS teacher，\n")
        f.write("形成可用于后续 GA fitness 预筛/快速重构的 PIKAN/FKAN-style 加速器。\n")
        f.write("若 model_vs_teacher 仍明显偏大，则该模型只作为 proof-of-concept，不直接替代 PC-VS。\n")

    print("=" * 100)
    print("Stage6-v3 final summary")
    print(f"overall model_vs_true mean unmon NRMSE : {overall_mean:.6e}")
    print(f"overall model_vs_true worst unmon NRMSE: {overall_worst:.6e}")
    print(f"overall teacher_vs_true mean unmon NRMSE: {teacher_mean:.6e}")
    print(f"overall model_vs_teacher mean unmon NRMSE: {mimic_mean:.6e}")
    print(f"exclude example1 model_vs_true mean unmon NRMSE: {noex_model_true:.6e}")
    print(f"exclude example1 model_vs_teacher mean unmon NRMSE: {noex_model_teacher:.6e}")
    print(f"best mask by model aggregate: {best_mask}")
    print(f"summary saved: {summary_path}")
    print("=" * 100)


# =============================================================================
# 9. 主程序
# =============================================================================

def main():
    ensure_dir(OUT_DIR)
    ensure_dir(LABEL_DIR)
    ensure_dir(CHECKPOINT_DIR)
    set_all_seeds(BASE_SEED)

    M_np, K_np, C_np = build_shear_building_mck(
        N_DOF, M_VAL, K_VAL, ALPHA_RAYLEIGH, BETA_RAYLEIGH
    )

    cases = build_or_load_10dof_cases(M_np, C_np, K_np)
    layouts = load_top_layouts()

    manifest_path = os.path.join(LABEL_DIR, "stage6_label_manifest.csv")

    # Stage6-v3 只改变 student 结构与训练目标，teacher label 可以直接复用 v2 的高质量 Top20 cache。
    reused = False
    if REUSE_STAGE6_V2_LABELS:
        for cand_manifest in STAGE6_V2_MANIFEST_CANDIDATES:
            if os.path.exists(cand_manifest):
                print(f"检测到可复用的 Stage6 label manifest: {cand_manifest}")
                manifest_path = cand_manifest
                reused = True
                break

    if not reused:
        if BUILD_LABEL_CACHE:
            manifest_path = build_label_cache(cases, layouts, M_np, C_np, K_np)
        elif not os.path.exists(manifest_path):
            raise FileNotFoundError(f"BUILD_LABEL_CACHE=False，且找不到 {manifest_path}")

    train_stage6_model(manifest_path, M_np, C_np, K_np)


if __name__ == "__main__":
    main()
