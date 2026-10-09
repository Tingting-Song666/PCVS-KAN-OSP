# -*- coding: utf-8 -*-
"""
ga_pc_vs_20dof_stage5_ga.py

阶段 5：20 自由度结构 + 固定 4 传感器 + GA 多随机种子搜索 + Top-K 复评 + 噪声鲁棒性 + 局部邻域补充验证
============================================================================
研究主题：
    以未监测自由度响应重构精度为目标，构建 GA-PIKAN/FKAN 联合优化框架，
    在固定传感器数量约束下搜索传感器布设方案，并通过结构动力学物理残差约束
    提高重构结果的物理一致性。

当前脚本定位：
    这是阶段 5 的 GA 搜索与验证版本。
    先不把 PIKAN/FKAN 强行接回训练，而是把已经验证有效的
    Physics-Constrained Virtual Sensing 作为 GA 的 fitness evaluator。

    对 20 自由度剪切型结构：
        1) 读取现有多地震波 case_list.csv 中的 eq_file；
        2) 内部用 Newmark-beta 生成 20-DOF 响应，并缓存到 responses_10dof/；
        3) 固定 N_SENSOR 个传感器，用 GA 搜索候选布设；
        4) 对每个布设，用“传感器硬约束 + 未监测 DOF 直接优化 + 结构动力学残差”评价；
        5) 在 GA 粗筛后，对 Top-K 布设在 train/val/test 全部地震波上做高精度复评；
        6) 输出总体最优布设、分 split 排名、每 case 指标和 GA 搜索记录。

运行方式：
    放在包含 case_list.csv、eq_data/ 的工程根目录运行：
        python ga_pc_vs_20dof_stage5_ga.py

输出目录：
    ga_pc_vs_20dof_stage5_results/

建议：
    先用默认配置跑通。默认是 N_DOF=10, N_SENSOR=3。
    20DOF+3传感器组合数 C(10,3)=120，已经开始适合 GA。
"""

import os
import ast
import time as pytime
import itertools
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 0. 全局配置
# =============================================================================

# -------------------------
# 设备设置
# -------------------------
FORCE_GPU = True
GPU_ID = 0

if FORCE_GPU:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "当前 PyTorch 没有检测到 CUDA，无法使用 GPU。\n"
            "如果你想临时用 CPU 跑，把 FORCE_GPU 改成 False。"
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
print("Stage 4 | 20DOF + GA sensor placement + physics-constrained virtual sensing")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Selected DEVICE : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU name        : {torch.cuda.get_device_name(DEVICE)}")
print("=" * 100)

# -------------------------
# 文件路径
# -------------------------
CASE_LIST_CANDIDATES = ["case_list.csv", "case_list(2).csv"]
OUT_DIR = "ga_pc_vs_20dof_stage5_results"
RESP10_ROOT = "responses_20dof"
CASE10_CSV = "case_list_20dof.csv"
CASE10_SUMMARY_CSV = "case_summary_20dof.csv"

# -------------------------
# 地震波读取设置，与 generate_data.py 保持一致
# -------------------------
EQ_SKIPROWS = 1
EQ_TIME_COL = 0
EQ_ACC_COL = 1
EQ_SCALE_FACTOR = 0.1

# -------------------------
# 结构参数
# -------------------------
N_DOF = 20
N_SENSOR = 4
M_VAL = 20000.0
K_VAL = 15000000.0
ALPHA_RAYLEIGH = 0.5
BETA_RAYLEIGH = 0.001
NEWMARK_GAMMA = 0.5
NEWMARK_BETA = 0.25

# -------------------------
# 运行范围
# -------------------------
SPLITS_TO_RUN = ["train", "val", "test"]
GA_TRAIN_SPLIT = "train"
# GA 粗筛默认只用部分 train case，保证搜索能跑起来；Top-K 复评会用全部 train/val/test。
GA_MAX_TRAIN_CASES = 6      # 想更严格可改 None；想更快可改 3
FINAL_MAX_CASES_PER_SPLIT: Optional[int] = None

# -------------------------
# GA 设置
# -------------------------
BASE_SEED = 2026
GA_POP_SIZE = 18
GA_NUM_GENERATIONS = 10
GA_ELITE_KEEP = 4
GA_MUTATION_RATE = 0.38
GA_TOURNAMENT_K = 3
GA_TOPK_FINAL_EVAL = 30

# fitness: score = mean_nrmse + ALPHA_WORST * worst_nrmse + BETA_PHYS * mean_true_p
ALPHA_WORST = 0.35
BETA_PHYS = 0.05

# 是否额外做全枚举对照。C(10,3)=120，较慢；默认 False。
RUN_FULL_ENUMERATION_BENCHMARK = False

# -------------------------
# 粗筛 evaluator 参数：快，服务 GA 搜索方向
# -------------------------
COARSE_MAX_POINTS = 420
COARSE_ADAM_STEPS = 220
COARSE_ADAM_LR = 3e-3
COARSE_USE_LBFGS = False
COARSE_LBFGS_MAX_ITER = 0

# -------------------------
# Top-K 复评 evaluator 参数：更稳，用于最终排序
# -------------------------
FINAL_MAX_POINTS = 700
FINAL_ADAM_STEPS = 550
FINAL_ADAM_LR = 2e-3
FINAL_USE_LBFGS = True
FINAL_LBFGS_MAX_ITER = 35

PRINT_OPT_EVERY = 999999  # 默认不打印每步优化，避免日志爆炸；调试可改 200

# -------------------------
# loss 权重：延续已验证的物理约束虚拟传感框架
# -------------------------
PHYSICS_WEIGHT = 1.0
PRIOR_WEIGHT = 1e-2
SMOOTH_WEIGHT = 1e-3
SPATIAL_SMOOTH_WEIGHT = 1e-4
ENERGY_WEIGHT = 1e-2
ENERGY_RATIO_LIMIT = 3.0
IC_WEIGHT = 1.0
USE_ZERO_INITIAL_DISPLACEMENT = True
USE_ZERO_INITIAL_VELOCITY = True

# -------------------------
# 输出与恢复
# -------------------------
CACHE_20DOF_RESPONSES = True
RESUME_GA_CACHE = True
RESUME_FINAL_EVAL = True
SAVE_INCREMENTAL = True
SAVE_FINAL_PREDICTIONS = False   # Top-K 复评是否保存预测 CSV，默认只保存指标
SAVE_FINAL_PLOTS = False         # 默认不保存图，避免 TopK*case 过多

# -------------------------
# Stage 5 专用设置：GA 多随机种子、局部邻域、噪声鲁棒性
# -------------------------
GA_RANDOM_SEEDS = [2026, 2027, 2028]
GA_TOPK_CLEAN_EVAL = 30
LOCAL_BASE_TOP_N = 10
LOCAL_NEIGHBOR_RADIUS = 2
LOCAL_MAX_NEW_CANDIDATES = 80
ROBUST_TOPK_CANDIDATES = 20
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05]
NOISE_BASE_SEED = 7001

# 局部邻域与噪声鲁棒性复评使用同一套 evaluator，略低于 enum 全复评成本，但保持稳定。
LOCAL_MAX_POINTS = FINAL_MAX_POINTS
LOCAL_ADAM_STEPS = FINAL_ADAM_STEPS
LOCAL_ADAM_LR = FINAL_ADAM_LR
LOCAL_USE_LBFGS = FINAL_USE_LBFGS
LOCAL_LBFGS_MAX_ITER = FINAL_LBFGS_MAX_ITER

ROBUST_MAX_POINTS = FINAL_MAX_POINTS
ROBUST_ADAM_STEPS = FINAL_ADAM_STEPS
ROBUST_ADAM_LR = FINAL_ADAM_LR
ROBUST_USE_LBFGS = FINAL_USE_LBFGS
ROBUST_LBFGS_MAX_ITER = FINAL_LBFGS_MAX_ITER


# =============================================================================
# 1. 工具函数
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_case_list_csv() -> str:
    for p in CASE_LIST_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"没有找到 case_list.csv。已尝试: {CASE_LIST_CANDIDATES}\n"
        "请把本脚本放在已有多地震波 case_list.csv 所在目录运行。"
    )


def safe_name(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def build_shear_building_mck(
    n_dof: int,
    m_val: float,
    k_val: float,
    alpha_rayleigh: float,
    beta_rayleigh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def load_eq_file(eq_file: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(eq_file, skiprows=EQ_SKIPROWS)
    t_raw = data[:, EQ_TIME_COL].astype(np.float64)
    ag_raw = data[:, EQ_ACC_COL].astype(np.float64) * EQ_SCALE_FACTOR
    t_raw = t_raw - t_raw[0]
    return t_raw, ag_raw


def resample_to_uniform_time(t_raw: np.ndarray, ag_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    dt_list = np.diff(t_raw)
    if np.any(dt_list <= 0):
        raise ValueError("地震波时间列不是严格递增，请检查 eq_file。")
    dt = float(np.min(dt_list))
    t_uniform = np.arange(t_raw[0], t_raw[-1] + 0.5 * dt, dt)
    ag_uniform = np.interp(t_uniform, t_raw, ag_raw)
    t_uniform = t_uniform - t_uniform[0]
    return t_uniform.astype(np.float64), ag_uniform.astype(np.float64), dt


def newmark_beta_linear(
    M: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    ag: np.ndarray,
    dt: float,
    gamma: float = 0.5,
    beta: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """求解 M u¨ + C u˙ + K u = -M r ag(t)，返回相对位移/速度/加速度。"""
    n_dof = M.shape[0]
    nt = len(ag)
    r = np.ones((n_dof, 1), dtype=np.float64)

    u = np.zeros((nt, n_dof), dtype=np.float64)
    v = np.zeros((nt, n_dof), dtype=np.float64)
    a = np.zeros((nt, n_dof), dtype=np.float64)

    p0 = -M @ r * ag[0]
    a[0, :] = np.linalg.solve(M, p0[:, 0] - C @ v[0, :] - K @ u[0, :])

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


def save_response_csv(out_csv: str, t: np.ndarray, u: np.ndarray) -> None:
    ensure_dir(os.path.dirname(out_csv))
    df = pd.DataFrame({"Time": t})
    for i in range(u.shape[1]):
        df[f"DOF{i + 1}"] = u[:, i]
    df.to_csv(out_csv, index=False)


def infer_quake_group(case_id: str, eq_file: str) -> str:
    s = f"{case_id} {eq_file}".lower()
    if "example" in s:
        return "example"
    if "nearp" in s or "near_p" in s or "pulse" in s:
        return "near_pulse"
    if "nearnp" in s or "near_np" in s or "no_pulse" in s or "no pulse" in s:
        return "near_no_pulse"
    if "far" in s:
        return "far_field"
    return "unknown"


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def safe_mean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def safe_min(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else float("nan")


def safe_max(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else float("nan")


def make_sensor_name(sensor_indices: List[int]) -> str:
    return "_".join([f"DOF{i + 1}" for i in sensor_indices])


def sensor_tuple_to_str(tup: Tuple[int, ...]) -> str:
    return "[" + ",".join(str(int(x)) for x in tup) + "]"


def parse_sensor_tuple(s: str) -> Tuple[int, ...]:
    if isinstance(s, (tuple, list)):
        return tuple(int(x) for x in s)
    try:
        obj = ast.literal_eval(str(s))
        return tuple(int(x) for x in obj)
    except Exception:
        s2 = str(s).strip().strip("[]")
        return tuple(int(x) for x in s2.split(",") if x.strip() != "")


def select_time_indices(n_time: int, max_points: int) -> np.ndarray:
    if n_time <= max_points:
        return np.arange(n_time, dtype=int)
    stride = int(np.ceil(n_time / max_points))
    idx = np.arange(0, n_time, stride, dtype=int)
    if idx[-1] != n_time - 1:
        idx = np.r_[idx, n_time - 1]
    return np.unique(idx)


# =============================================================================
# 2. 20DOF 响应生成与读取
# =============================================================================

def load_or_generate_10dof_cases(case_list_csv: str, M_np: np.ndarray, C_np: np.ndarray, K_np: np.ndarray) -> List[Dict]:
    df_case = pd.read_csv(case_list_csv)
    required_cols = ["case_id", "split", "eq_file"]
    for c in required_cols:
        if c not in df_case.columns:
            raise ValueError(f"{case_list_csv} 缺少必要列: {c}")

    cases: List[Dict] = []
    case10_records = []
    case10_summaries = []

    for idx, row in df_case.iterrows():
        case_id = str(row["case_id"])
        split = str(row["split"]).lower()
        eq_file = str(row["eq_file"])
        if not os.path.exists(eq_file):
            raise FileNotFoundError(f"找不到地震波文件: {eq_file}")

        resp_dir = os.path.join(RESP10_ROOT, split)
        ensure_dir(resp_dir)
        resp_file = os.path.join(resp_dir, f"response20_{safe_name(case_id)}.csv")

        if CACHE_20DOF_RESPONSES and os.path.exists(resp_file):
            resp_df = pd.read_csv(resp_file)
            t_np = resp_df["Time"].values.astype(np.float64)
            dof_cols = [f"DOF{i + 1}" for i in range(N_DOF)]
            u_np = resp_df[dof_cols].values.astype(np.float64)
            t_raw, ag_raw = load_eq_file(eq_file)
            eq_time, eq_acc, _ = resample_to_uniform_time(t_raw, ag_raw)
        else:
            t_raw, ag_raw = load_eq_file(eq_file)
            eq_time, eq_acc, dt = resample_to_uniform_time(t_raw, ag_raw)
            u_np, v_np, a_np = newmark_beta_linear(
                M=M_np, C=C_np, K=K_np, ag=eq_acc, dt=dt,
                gamma=NEWMARK_GAMMA, beta=NEWMARK_BETA,
            )
            t_np = eq_time.copy()
            if CACHE_20DOF_RESPONSES:
                save_response_csv(resp_file, t_np, u_np)

        t_np = t_np - t_np[0]
        duration = float(t_np[-1] - t_np[0]) if len(t_np) > 1 else 0.0
        quake_group = infer_quake_group(case_id, eq_file)

        cases.append({
            "case_index": len(cases),
            "case_id": case_id,
            "split": split,
            "quake_group": quake_group,
            "eq_file": eq_file,
            "response20_file": resp_file,
            "eq_time_np": eq_time.astype(np.float64),
            "eq_acc_np": eq_acc.astype(np.float64),
            "t_np": t_np.astype(np.float64),
            "u_np": u_np.astype(np.float64),
            "duration": duration,
            "n_time_raw": int(len(t_np)),
        })

        case10_records.append({
            "case_id": case_id,
            "split": split,
            "quake_group": quake_group,
            "eq_file": eq_file.replace("\\", "/"),
            "response20_file": resp_file.replace("\\", "/"),
        })
        summary = {
            "case_id": case_id,
            "split": split,
            "quake_group": quake_group,
            "eq_file": eq_file,
            "response20_file": resp_file,
            "n_time_steps": len(t_np),
            "duration": duration,
            "dt_median": float(np.median(np.diff(t_np))) if len(t_np) > 1 else np.nan,
            "ag_max_abs": float(np.max(np.abs(eq_acc))),
        }
        for j in range(N_DOF):
            summary[f"DOF{j + 1}_max_abs_disp"] = float(np.max(np.abs(u_np[:, j])))
        case10_summaries.append(summary)

    pd.DataFrame(case10_records).to_csv(CASE10_CSV, index=False)
    pd.DataFrame(case10_summaries).to_csv(CASE10_SUMMARY_CSV, index=False)
    return cases


# =============================================================================
# 3. 指标与插值先验
# =============================================================================

def interpolation_prior_from_sensors(
    u_sensor: np.ndarray,
    sensor_indices: List[int],
    recon_indices: List[int],
) -> np.ndarray:
    sensor_x = np.array(sensor_indices, dtype=np.float64)
    recon_x = np.array(recon_indices, dtype=np.float64)
    prior = np.zeros((u_sensor.shape[0], len(recon_indices)), dtype=np.float64)
    for it in range(u_sensor.shape[0]):
        prior[it, :] = np.interp(recon_x, sensor_x, u_sensor[it, :])
    return prior


def calc_metrics(u_true: np.ndarray, u_pred: np.ndarray, sensor_indices: List[int]) -> Dict:
    n_dof = u_true.shape[1]
    sensor_set = set(sensor_indices)
    recon_indices = [i for i in range(n_dof) if i not in sensor_set]
    n_time = u_true.shape[0]

    mask = np.zeros(n_dof, dtype=np.float64)
    mask[sensor_indices] = 1.0
    unmask = 1.0 - mask
    err = u_pred - u_true

    sensor_rmse = np.sqrt(np.sum((err * mask.reshape(1, -1)) ** 2) / (np.sum(mask) * n_time + 1e-18))
    sensor_rms_true = np.sqrt(np.sum((u_true * mask.reshape(1, -1)) ** 2) / (np.sum(mask) * n_time + 1e-18))
    sensor_nrmse = sensor_rmse / (sensor_rms_true + 1e-18)

    recon_rmse = np.sqrt(np.sum((err * unmask.reshape(1, -1)) ** 2) / (np.sum(unmask) * n_time + 1e-18))
    recon_rms_true = np.sqrt(np.sum((u_true * unmask.reshape(1, -1)) ** 2) / (np.sum(unmask) * n_time + 1e-18))
    recon_nrmse = recon_rmse / (recon_rms_true + 1e-18)

    r2_per_dof, rmse_per_dof = [], []
    for i in range(n_dof):
        r2_per_dof.append(compute_r2(u_true[:, i], u_pred[:, i]))
        rmse_per_dof.append(float(np.sqrt(np.mean((u_pred[:, i] - u_true[:, i]) ** 2))))

    sensor_r2 = [r2_per_dof[i] for i in sensor_indices]
    recon_r2 = [r2_per_dof[i] for i in recon_indices]
    recon_rmse_list = [rmse_per_dof[i] for i in recon_indices]

    return {
        "rmse_monitored": float(sensor_rmse),
        "nrmse_monitored": float(sensor_nrmse),
        "rmse_unmonitored": float(recon_rmse),
        "nrmse_unmonitored": float(recon_nrmse),
        "r2_per_dof": r2_per_dof,
        "rmse_per_dof": rmse_per_dof,
        "mean_sensor_R2": safe_mean(sensor_r2),
        "worst_sensor_R2": safe_min(sensor_r2),
        "mean_reconstructed_R2": safe_mean(recon_r2),
        "worst_reconstructed_R2": safe_min(recon_r2),
        "worst_reconstructed_RMSE": safe_max(recon_rmse_list),
    }


# =============================================================================
# 4. 物理约束直接虚拟传感优化器
# =============================================================================

class DirectVirtualSensingOptimizer:
    def __init__(
        self,
        t_np: np.ndarray,
        u_true_np: np.ndarray,
        eq_time_np: np.ndarray,
        eq_acc_np: np.ndarray,
        sensor_indices: List[int],
        M_np: np.ndarray,
        C_np: np.ndarray,
        K_np: np.ndarray,
        adam_steps: int,
        adam_lr: float,
        use_lbfgs: bool,
        lbfgs_max_iter: int,
    ):
        self.t_np = t_np.astype(np.float64)
        self.u_true_np = u_true_np.astype(np.float64)
        self.eq_time_np = eq_time_np.astype(np.float64)
        self.eq_acc_np = eq_acc_np.astype(np.float64)
        self.sensor_indices = list(sensor_indices)
        self.n_dof = self.u_true_np.shape[1]
        self.recon_indices = [i for i in range(self.n_dof) if i not in set(sensor_indices)]
        self.n_time = len(t_np)
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.use_lbfgs = bool(use_lbfgs)
        self.lbfgs_max_iter = int(lbfgs_max_iter)
        self.history: List[Dict] = []

        if self.n_time < 5:
            raise ValueError("时间点太少，无法进行有限差分物理残差。")
        dt_arr = np.diff(self.t_np)
        self.dt = float(np.median(dt_arr))

        self.ag_np = np.interp(self.t_np, self.eq_time_np, self.eq_acc_np).astype(np.float64)
        self.u_sensor_np = self.u_true_np[:, self.sensor_indices]
        self.u_prior_unknown_np = interpolation_prior_from_sensors(
            u_sensor=self.u_sensor_np,
            sensor_indices=self.sensor_indices,
            recon_indices=self.recon_indices,
        )

        self.u_sensor_t = torch.tensor(self.u_sensor_np, dtype=torch.float32, device=DEVICE)
        self.u_prior_unknown_t = torch.tensor(self.u_prior_unknown_np, dtype=torch.float32, device=DEVICE)
        self.ag_t = torch.tensor(self.ag_np, dtype=torch.float32, device=DEVICE).view(-1, 1)
        self.M_t = torch.tensor(M_np, dtype=torch.float32, device=DEVICE)
        self.C_t = torch.tensor(C_np, dtype=torch.float32, device=DEVICE)
        self.K_t = torch.tensor(K_np, dtype=torch.float32, device=DEVICE)
        self.sensor_indices_t = torch.tensor(self.sensor_indices, dtype=torch.long, device=DEVICE)
        self.recon_indices_t = torch.tensor(self.recon_indices, dtype=torch.long, device=DEVICE)

        ones = torch.ones((self.n_dof, 1), dtype=torch.float32, device=DEVICE)
        self.force_vec = (self.M_t @ ones).view(-1)
        self.u_unknown = torch.nn.Parameter(self.u_prior_unknown_t.clone())

        full_init = self._assemble_full_response(self.u_unknown.detach())
        u_scale = torch.std(full_init, dim=0)
        sensor_scale_mean = torch.mean(torch.std(self.u_sensor_t, dim=0))
        u_scale = torch.clamp(u_scale, min=torch.clamp(sensor_scale_mean * 0.05, min=1e-8))
        self.u_scale = u_scale.detach()

        ag_scale = torch.clamp(torch.std(self.ag_t).detach(), min=1e-8)
        duration = max(float(self.t_np[-1] - self.t_np[0]), self.dt * 10.0)
        disp_term = torch.abs(self.K_t) @ self.u_scale.view(-1, 1)
        vel_term = torch.abs(self.C_t) @ (self.u_scale / duration).view(-1, 1)
        acc_term = torch.abs(self.M_t) @ (self.u_scale / (duration ** 2)).view(-1, 1)
        eq_term = torch.abs(self.force_vec).view(-1, 1) * ag_scale
        res_scale = (disp_term + vel_term + acc_term + eq_term).view(-1)
        self.res_scale = torch.clamp(res_scale, min=1.0).detach()

        sensor_rms = torch.sqrt(torch.mean(self.u_sensor_t ** 2) + 1e-18).detach()
        self.sensor_rms = torch.clamp(sensor_rms, min=1e-8)

    def _assemble_full_response(self, u_unknown: torch.Tensor) -> torch.Tensor:
        u_full = torch.zeros((self.n_time, self.n_dof), dtype=torch.float32, device=DEVICE)
        u_full[:, self.sensor_indices_t] = self.u_sensor_t
        u_full[:, self.recon_indices_t] = u_unknown
        return u_full

    def finite_difference(self, u_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dt = self.dt
        u_mid = u_full[1:-1, :]
        u_dot = (u_full[2:, :] - u_full[:-2, :]) / (2.0 * dt)
        u_ddot = (u_full[2:, :] - 2.0 * u_full[1:-1, :] + u_full[:-2, :]) / (dt ** 2)
        ag_mid = self.ag_t[1:-1, :]
        return u_mid, u_dot, u_ddot, ag_mid

    def physics_loss(self, u_full: torch.Tensor) -> torch.Tensor:
        u_mid, u_dot, u_ddot, ag_mid = self.finite_difference(u_full)
        residual = (
            u_ddot @ self.M_t.T
            + u_dot @ self.C_t.T
            + u_mid @ self.K_t.T
            + ag_mid @ self.force_vec.view(1, -1)
        )
        residual_norm = residual / self.res_scale.view(1, -1)
        return torch.mean(residual_norm ** 2)

    def prior_loss(self, u_unknown: torch.Tensor) -> torch.Tensor:
        scale = self.u_scale[self.recon_indices_t].view(1, -1)
        return torch.mean(((u_unknown - self.u_prior_unknown_t) / scale) ** 2)

    def smooth_loss(self, u_unknown: torch.Tensor) -> torch.Tensor:
        d2 = u_unknown[2:, :] - 2.0 * u_unknown[1:-1, :] + u_unknown[:-2, :]
        scale = self.u_scale[self.recon_indices_t].view(1, -1)
        return torch.mean((d2 / scale) ** 2)

    def spatial_smooth_loss(self, u_full: torch.Tensor) -> torch.Tensor:
        if self.n_dof < 3:
            return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        d2_space = u_full[:, 2:] - 2.0 * u_full[:, 1:-1] + u_full[:, :-2]
        scale = torch.clamp(torch.mean(self.u_scale), min=1e-8)
        return torch.mean((d2_space / scale) ** 2)

    def energy_loss(self, u_unknown: torch.Tensor) -> torch.Tensor:
        unk_rms_each = torch.sqrt(torch.mean(u_unknown ** 2, dim=0) + 1e-18)
        ratio = unk_rms_each / (self.sensor_rms + 1e-18)
        penalty = torch.relu(ratio / ENERGY_RATIO_LIMIT - 1.0)
        return torch.mean(penalty ** 2)

    def ic_loss(self, u_full: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        if USE_ZERO_INITIAL_DISPLACEMENT:
            loss = loss + torch.mean((u_full[0, :] / self.u_scale) ** 2)
        if USE_ZERO_INITIAL_VELOCITY and self.n_time >= 2:
            v0 = (u_full[1, :] - u_full[0, :]) / max(self.dt, 1e-12)
            v_scale = torch.clamp(self.u_scale / max(self.dt * 10.0, 1e-12), min=1e-8)
            loss = loss + torch.mean((v0 / v_scale) ** 2)
        return loss

    def total_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        u_full = self._assemble_full_response(self.u_unknown)
        p = self.physics_loss(u_full)
        prior = self.prior_loss(self.u_unknown)
        sm = self.smooth_loss(self.u_unknown)
        sp = self.spatial_smooth_loss(u_full)
        en = self.energy_loss(self.u_unknown)
        ic = self.ic_loss(u_full)
        total = (
            PHYSICS_WEIGHT * p
            + PRIOR_WEIGHT * prior
            + SMOOTH_WEIGHT * sm
            + SPATIAL_SMOOTH_WEIGHT * sp
            + ENERGY_WEIGHT * en
            + IC_WEIGHT * ic
        )
        parts = {
            "total": float(total.detach().cpu().item()),
            "physics": float(p.detach().cpu().item()),
            "prior": float(prior.detach().cpu().item()),
            "smooth": float(sm.detach().cpu().item()),
            "spatial_smooth": float(sp.detach().cpu().item()),
            "energy": float(en.detach().cpu().item()),
            "ic": float(ic.detach().cpu().item()),
        }
        return total, parts

    def optimize(self, layout_name: str, verbose: bool = False) -> Dict:
        opt = torch.optim.Adam([self.u_unknown], lr=self.adam_lr)
        t0 = pytime.time()
        for step in range(1, self.adam_steps + 1):
            opt.zero_grad(set_to_none=True)
            loss, parts = self.total_loss()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{layout_name} Adam step={step} 出现非有限 loss: {loss.item()}")
            loss.backward()
            opt.step()
            if step == 1 or step == self.adam_steps or (PRINT_OPT_EVERY > 0 and step % PRINT_OPT_EVERY == 0):
                elapsed = pytime.time() - t0
                self.history.append({"stage": "Adam", "step": step, **parts, "elapsed_sec": elapsed})
                if verbose:
                    print(f"[{layout_name} Adam] {step}/{self.adam_steps} total={parts['total']:.3e} p={parts['physics']:.3e}")

        if self.use_lbfgs and self.lbfgs_max_iter > 0:
            lbfgs = torch.optim.LBFGS(
                [self.u_unknown], lr=1.0, max_iter=self.lbfgs_max_iter,
                tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe",
            )
            calls = {"n": 0}
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _ = self.total_loss()
                loss.backward()
                calls["n"] += 1
                return loss
            lbfgs.step(closure)
            loss, parts = self.total_loss()
            elapsed = pytime.time() - t0
            self.history.append({"stage": "LBFGS", "step": calls["n"], **parts, "elapsed_sec": elapsed})

        with torch.no_grad():
            u_full = self._assemble_full_response(self.u_unknown).detach().cpu().numpy()
            _, final_parts = self.total_loss()
        return {"u_pred": u_full, "loss_parts": final_parts, "history": list(self.history)}


# =============================================================================
# 5. 单 case / 单 layout 评价
# =============================================================================

def run_one_case_one_layout(
    case: Dict,
    sensor_indices: List[int],
    M_np: np.ndarray,
    C_np: np.ndarray,
    K_np: np.ndarray,
    max_points: int,
    adam_steps: int,
    adam_lr: float,
    use_lbfgs: bool,
    lbfgs_max_iter: int,
    combo_id: int = 0,
    save_outputs: bool = False,
    out_prefix: str = "",
) -> Tuple[Dict, List[Dict], List[Dict]]:
    idx = select_time_indices(len(case["t_np"]), max_points)
    t_np = case["t_np"][idx]
    u_true = case["u_np"][idx, :]
    sensor_indices = sorted([int(x) for x in sensor_indices])
    recon_indices = [i for i in range(N_DOF) if i not in set(sensor_indices)]
    layout_name = make_sensor_name(sensor_indices)
    set_all_seeds(BASE_SEED + case["case_index"] * 1000 + combo_id)

    opt = DirectVirtualSensingOptimizer(
        t_np=t_np, u_true_np=u_true,
        eq_time_np=case["eq_time_np"], eq_acc_np=case["eq_acc_np"],
        sensor_indices=sensor_indices,
        M_np=M_np, C_np=C_np, K_np=K_np,
        adam_steps=adam_steps, adam_lr=adam_lr,
        use_lbfgs=use_lbfgs, lbfgs_max_iter=lbfgs_max_iter,
    )

    # prior baseline
    u_prior_full = np.zeros_like(u_true)
    u_prior_full[:, sensor_indices] = u_true[:, sensor_indices]
    u_prior_full[:, recon_indices] = opt.u_prior_unknown_np
    prior_metrics = calc_metrics(u_true, u_prior_full, sensor_indices)

    result = opt.optimize(f"{case['case_id']}|{layout_name}", verbose=False)
    u_pred = result["u_pred"]
    loss_parts = result["loss_parts"]
    metrics = calc_metrics(u_true, u_pred, sensor_indices)
    improve_ratio = metrics["nrmse_unmonitored"] / (prior_metrics["nrmse_unmonitored"] + 1e-18)

    pred_path = ""
    if save_outputs and SAVE_FINAL_PREDICTIONS and out_prefix:
        pred_dir = os.path.join(OUT_DIR, "final_predictions")
        ensure_dir(pred_dir)
        pred_path = os.path.join(pred_dir, f"{out_prefix}_{layout_name}_prediction.csv")
        df = pd.DataFrame({"Time": t_np})
        for j in range(N_DOF):
            df[f"DOF{j + 1}_True"] = u_true[:, j]
            df[f"DOF{j + 1}_Recon"] = u_pred[:, j]
            df[f"DOF{j + 1}_Role"] = "sensor" if j in set(sensor_indices) else "reconstructed"
        df.to_csv(pred_path, index=False)

    summary = {
        "case_id": case["case_id"],
        "split": case["split"],
        "quake_group": case["quake_group"],
        "mask_name": layout_name,
        "sensor_tuple": sensor_tuple_to_str(tuple(sensor_indices)),
        "sensor_indices_0based": str(sensor_indices),
        "sensor_floors_1based": str([i + 1 for i in sensor_indices]),
        "reconstructed_indices_0based": str(recon_indices),
        "reconstructed_floors_1based": str([i + 1 for i in recon_indices]),
        "n_time_raw": case["n_time_raw"],
        "n_time_used": len(t_np),
        "duration": case["duration"],
        "prior_nrmse_unmonitored": prior_metrics["nrmse_unmonitored"],
        "prior_worst_reconstructed_R2": prior_metrics["worst_reconstructed_R2"],
        "improve_ratio_vs_prior": improve_ratio,
        "rmse_unmonitored": metrics["rmse_unmonitored"],
        "nrmse_unmonitored": metrics["nrmse_unmonitored"],
        "rmse_monitored": metrics["rmse_monitored"],
        "nrmse_monitored": metrics["nrmse_monitored"],
        "mean_reconstructed_R2": metrics["mean_reconstructed_R2"],
        "worst_reconstructed_R2": metrics["worst_reconstructed_R2"],
        "worst_reconstructed_RMSE": metrics["worst_reconstructed_RMSE"],
        "mean_sensor_R2": metrics["mean_sensor_R2"],
        "worst_sensor_R2": metrics["worst_sensor_R2"],
        "true_p_fd": loss_parts["physics"],
        "prior_loss": loss_parts["prior"],
        "smooth_loss": loss_parts["smooth"],
        "spatial_smooth_loss": loss_parts["spatial_smooth"],
        "energy_loss": loss_parts["energy"],
        "ic_loss": loss_parts["ic"],
        "prediction_csv": pred_path,
    }

    dof_rows = []
    for j in range(N_DOF):
        dof_rows.append({
            "case_id": case["case_id"],
            "split": case["split"],
            "quake_group": case["quake_group"],
            "mask_name": layout_name,
            "sensor_tuple": sensor_tuple_to_str(tuple(sensor_indices)),
            "DOF": j + 1,
            "role": "sensor" if j in set(sensor_indices) else "reconstructed",
            "R2": metrics["r2_per_dof"][j],
            "RMSE": metrics["rmse_per_dof"][j],
        })

    hist_rows = []
    for h in result["history"]:
        hist_rows.append({
            "case_id": case["case_id"],
            "split": case["split"],
            "quake_group": case["quake_group"],
            "mask_name": layout_name,
            "sensor_tuple": sensor_tuple_to_str(tuple(sensor_indices)),
            **h,
        })

    del opt
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return summary, dof_rows, hist_rows


# =============================================================================
# 6. 聚合与保存
# =============================================================================

def aggregate_rankings(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False).agg(
        n_cases=("case_id", "nunique"),
        mean_unmon_nrmse=("nrmse_unmonitored", "mean"),
        median_unmon_nrmse=("nrmse_unmonitored", "median"),
        worst_unmon_nrmse=("nrmse_unmonitored", "max"),
        std_unmon_nrmse=("nrmse_unmonitored", "std"),
        mean_prior_nrmse=("prior_nrmse_unmonitored", "mean"),
        mean_improve_ratio=("improve_ratio_vs_prior", "mean"),
        mean_worst_R2=("worst_reconstructed_R2", "mean"),
        worst_worst_R2=("worst_reconstructed_R2", "min"),
        mean_recon_R2=("mean_reconstructed_R2", "mean"),
        mean_true_p=("true_p_fd", "mean"),
        worst_true_p=("true_p_fd", "max"),
        mean_rmse_unmonitored=("rmse_unmonitored", "mean"),
    ).reset_index()
    agg = agg.sort_values(
        by=["mean_unmon_nrmse", "worst_unmon_nrmse", "worst_worst_R2", "mean_true_p"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    agg.insert(len(group_cols), "rank", np.arange(1, len(agg) + 1))
    return agg


def save_df(rows: List[Dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


# =============================================================================
# 7. GA 搜索
# =============================================================================

class GeneticSensorSearch:
    def __init__(self, cases_train: List[Dict], M_np: np.ndarray, C_np: np.ndarray, K_np: np.ndarray):
        self.cases_train = list(cases_train)
        if GA_MAX_TRAIN_CASES is not None:
            self.cases_train = self.cases_train[:GA_MAX_TRAIN_CASES]
        self.M_np = M_np
        self.C_np = C_np
        self.K_np = K_np
        self.all_genes = list(range(N_DOF))
        self.cache: Dict[Tuple[int, ...], Dict] = {}
        self.generation_rows: List[Dict] = []
        self.cache_path = os.path.join(OUT_DIR, "ga_candidate_cache.csv")
        self.gen_path = os.path.join(OUT_DIR, "ga_generation_log.csv")
        if RESUME_GA_CACHE:
            self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_path):
            df = pd.read_csv(self.cache_path)
            for _, row in df.iterrows():
                tup = parse_sensor_tuple(row["sensor_tuple"])
                self.cache[tup] = row.to_dict()
            print(f"检测到 GA candidate cache: {self.cache_path}，已加载 {len(self.cache)} 个候选。")
        if os.path.exists(self.gen_path):
            try:
                self.generation_rows = pd.read_csv(self.gen_path).to_dict("records")
            except Exception:
                self.generation_rows = []

    def _save_cache(self) -> None:
        if self.cache:
            pd.DataFrame(list(self.cache.values())).to_csv(self.cache_path, index=False)
        if self.generation_rows:
            pd.DataFrame(self.generation_rows).to_csv(self.gen_path, index=False)

    def random_candidate(self) -> Tuple[int, ...]:
        return tuple(sorted(random.sample(self.all_genes, N_SENSOR)))

    def repair(self, cand: List[int]) -> Tuple[int, ...]:
        unique = []
        for x in cand:
            x = max(0, min(N_DOF - 1, int(x)))
            if x not in unique:
                unique.append(x)
        while len(unique) < N_SENSOR:
            x = random.choice(self.all_genes)
            if x not in unique:
                unique.append(x)
        if len(unique) > N_SENSOR:
            unique = random.sample(unique, N_SENSOR)
        return tuple(sorted(unique))

    def initial_population(self) -> List[Tuple[int, ...]]:
        seeds = []
        # 均匀覆盖结构高度的一些启发式候选，避免初代太随机。
        heuristic = [
            [0, N_DOF // 2, N_DOF - 1],
            [0, max(1, N_DOF // 3), max(2, 2 * N_DOF // 3)],
            [1, N_DOF // 2, N_DOF - 1],
            [0, N_DOF // 2 - 1, N_DOF - 2],
            [1, N_DOF // 2, N_DOF - 2],
        ]
        for h in heuristic:
            if len(h) >= N_SENSOR:
                seeds.append(self.repair(h[:N_SENSOR]))
        while len(seeds) < GA_POP_SIZE:
            seeds.append(self.random_candidate())
        # 去重后补齐
        pop = []
        for c in seeds:
            if c not in pop:
                pop.append(c)
        while len(pop) < GA_POP_SIZE:
            c = self.random_candidate()
            if c not in pop:
                pop.append(c)
        return pop[:GA_POP_SIZE]

    def evaluate_candidate(self, cand: Tuple[int, ...]) -> Dict:
        cand = tuple(sorted(cand))
        if cand in self.cache:
            return self.cache[cand]

        case_rows = []
        t0 = pytime.time()
        for ci, case in enumerate(self.cases_train):
            summary, _, _ = run_one_case_one_layout(
                case=case,
                sensor_indices=list(cand),
                M_np=self.M_np,
                C_np=self.C_np,
                K_np=self.K_np,
                max_points=COARSE_MAX_POINTS,
                adam_steps=COARSE_ADAM_STEPS,
                adam_lr=COARSE_ADAM_LR,
                use_lbfgs=COARSE_USE_LBFGS,
                lbfgs_max_iter=COARSE_LBFGS_MAX_ITER,
                combo_id=ci,
                save_outputs=False,
            )
            case_rows.append(summary)

        df = pd.DataFrame(case_rows)
        mean_nrmse = float(df["nrmse_unmonitored"].mean())
        worst_nrmse = float(df["nrmse_unmonitored"].max())
        mean_true_p = float(df["true_p_fd"].mean())
        mean_worst_r2 = float(df["worst_reconstructed_R2"].mean())
        worst_worst_r2 = float(df["worst_reconstructed_R2"].min())
        score = mean_nrmse + ALPHA_WORST * worst_nrmse + BETA_PHYS * mean_true_p
        fitness = 1.0 / (score + 1e-12)
        elapsed = pytime.time() - t0
        row = {
            "sensor_tuple": sensor_tuple_to_str(cand),
            "mask_name": make_sensor_name(list(cand)),
            "sensor_floors_1based": str([i + 1 for i in cand]),
            "n_train_cases": len(self.cases_train),
            "fitness": fitness,
            "score": score,
            "mean_unmon_nrmse": mean_nrmse,
            "worst_unmon_nrmse": worst_nrmse,
            "mean_true_p": mean_true_p,
            "mean_worst_R2": mean_worst_r2,
            "worst_worst_R2": worst_worst_r2,
            "elapsed_sec": elapsed,
        }
        self.cache[cand] = row
        self._save_cache()
        print(
            f"[GA eval] {make_sensor_name(list(cand)):<24s} | "
            f"meanNRMSE={mean_nrmse:.4e} worst={worst_nrmse:.4e} "
            f"fitness={fitness:.4e} time={elapsed:.1f}s"
        )
        return row

    def tournament_select(self, population: List[Tuple[int, ...]], fit_map: Dict[Tuple[int, ...], float]) -> Tuple[int, ...]:
        sample = random.sample(population, min(GA_TOURNAMENT_K, len(population)))
        sample = sorted(sample, key=lambda c: fit_map[c], reverse=True)
        return sample[0]

    def crossover(self, p1: Tuple[int, ...], p2: Tuple[int, ...]) -> Tuple[int, ...]:
        pool = list(set(p1).union(set(p2)))
        random.shuffle(pool)
        if len(pool) >= N_SENSOR:
            child = pool[:N_SENSOR]
        else:
            child = pool[:]
            while len(child) < N_SENSOR:
                x = random.choice(self.all_genes)
                if x not in child:
                    child.append(x)
        return self.repair(child)

    def mutate(self, cand: Tuple[int, ...]) -> Tuple[int, ...]:
        arr = list(cand)
        if random.random() < GA_MUTATION_RATE:
            pos = random.randrange(N_SENSOR)
            choices = [x for x in self.all_genes if x not in arr]
            if choices:
                arr[pos] = random.choice(choices)
        return self.repair(arr)

    def run(self) -> Tuple[Tuple[int, ...], pd.DataFrame]:
        set_all_seeds(BASE_SEED)
        population = self.initial_population()
        best_cand = None
        best_fit = -np.inf

        for gen in range(1, GA_NUM_GENERATIONS + 1):
            print("\n" + "=" * 100)
            print(f"GA generation {gen}/{GA_NUM_GENERATIONS}")
            print("=" * 100)

            fit_map = {}
            eval_rows = []
            for cand in population:
                row = self.evaluate_candidate(cand)
                fit_map[cand] = float(row["fitness"])
                eval_rows.append(row)

            ranked = sorted(population, key=lambda c: fit_map[c], reverse=True)
            gen_best = ranked[0]
            if fit_map[gen_best] > best_fit:
                best_fit = fit_map[gen_best]
                best_cand = gen_best

            best_row = self.cache[gen_best]
            print(
                f">>> Gen {gen} best: {best_row['mask_name']} | "
                f"fitness={best_row['fitness']:.4e} | meanNRMSE={best_row['mean_unmon_nrmse']:.4e}"
            )
            self.generation_rows.append({
                "generation": gen,
                "best_sensor_tuple": sensor_tuple_to_str(gen_best),
                "best_mask_name": best_row["mask_name"],
                "best_fitness": best_row["fitness"],
                "best_mean_unmon_nrmse": best_row["mean_unmon_nrmse"],
                "best_worst_unmon_nrmse": best_row["worst_unmon_nrmse"],
                "global_best_tuple": sensor_tuple_to_str(best_cand),
                "global_best_fitness": best_fit,
            })
            self._save_cache()

            # 产生下一代
            next_pop = ranked[:GA_ELITE_KEEP]
            while len(next_pop) < GA_POP_SIZE:
                p1 = self.tournament_select(population, fit_map)
                p2 = self.tournament_select(population, fit_map)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                if child not in next_pop:
                    next_pop.append(child)
            population = next_pop[:GA_POP_SIZE]

        cache_df = pd.DataFrame(list(self.cache.values()))
        cache_df = cache_df.sort_values("fitness", ascending=False).reset_index(drop=True)
        cache_df.to_csv(self.cache_path, index=False)
        return best_cand, cache_df


# =============================================================================
# 8. Top-K 复评
# =============================================================================

def final_evaluate_topk(
    candidates: List[Tuple[int, ...]],
    cases_all: List[Dict],
    M_np: np.ndarray,
    C_np: np.ndarray,
    K_np: np.ndarray,
) -> None:
    out_metrics = os.path.join(OUT_DIR, "topk_per_case_mask_metrics.csv")
    out_dof = os.path.join(OUT_DIR, "topk_per_dof_metrics.csv")
    out_hist = os.path.join(OUT_DIR, "topk_optimizer_history.csv")

    summary_rows: List[Dict] = []
    dof_rows: List[Dict] = []
    hist_rows: List[Dict] = []
    existing_keys = set()

    if RESUME_FINAL_EVAL and os.path.exists(out_metrics):
        old = pd.read_csv(out_metrics)
        summary_rows = old.to_dict("records")
        existing_keys = set(zip(old["case_id"].astype(str), old["mask_name"].astype(str)))
        if os.path.exists(out_dof):
            dof_rows = pd.read_csv(out_dof).to_dict("records")
        if os.path.exists(out_hist):
            hist_rows = pd.read_csv(out_hist).to_dict("records")
        print(f"检测到 Top-K 复评已有 {len(existing_keys)} 个 case-mask，将断点续跑。")

    cases_to_run = []
    for sp in SPLITS_TO_RUN:
        sp_cases = [c for c in cases_all if c["split"] == sp]
        if FINAL_MAX_CASES_PER_SPLIT is not None:
            sp_cases = sp_cases[:FINAL_MAX_CASES_PER_SPLIT]
        cases_to_run.extend(sp_cases)

    total_jobs = len(candidates) * len(cases_to_run)
    job = 0
    for cand_id, cand in enumerate(candidates, start=1):
        layout_name = make_sensor_name(list(cand))
        print("\n" + "#" * 100)
        print(f"Top-K 复评候选 {cand_id}/{len(candidates)}: {layout_name} sensors={[i+1 for i in cand]}")
        print("#" * 100)
        for case in cases_to_run:
            job += 1
            key = (str(case["case_id"]), layout_name)
            if key in existing_keys:
                print(f"[跳过] {job}/{total_jobs} | {case['case_id']} | {layout_name}")
                continue
            print(f"[复评] {job}/{total_jobs} | case={case['case_id']} split={case['split']} | mask={layout_name}")
            safe_prefix = f"{safe_name(case['case_id'])}_{layout_name}"
            summary, dr, hr = run_one_case_one_layout(
                case=case,
                sensor_indices=list(cand),
                M_np=M_np, C_np=C_np, K_np=K_np,
                max_points=FINAL_MAX_POINTS,
                adam_steps=FINAL_ADAM_STEPS,
                adam_lr=FINAL_ADAM_LR,
                use_lbfgs=FINAL_USE_LBFGS,
                lbfgs_max_iter=FINAL_LBFGS_MAX_ITER,
                combo_id=cand_id,
                save_outputs=True,
                out_prefix=safe_prefix,
            )
            summary_rows.append(summary)
            dof_rows.extend(dr)
            hist_rows.extend(hr)
            existing_keys.add(key)
            if SAVE_INCREMENTAL:
                pd.DataFrame(summary_rows).to_csv(out_metrics, index=False)
                pd.DataFrame(dof_rows).to_csv(out_dof, index=False)
                pd.DataFrame(hist_rows).to_csv(out_hist, index=False)
                save_final_rankings(summary_rows)

    pd.DataFrame(summary_rows).to_csv(out_metrics, index=False)
    pd.DataFrame(dof_rows).to_csv(out_dof, index=False)
    pd.DataFrame(hist_rows).to_csv(out_hist, index=False)
    save_final_rankings(summary_rows)


def save_final_rankings(summary_rows: List[Dict]) -> None:
    df = pd.DataFrame(summary_rows)
    if len(df) == 0:
        return
    aggregate_rankings(df, ["mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, "topk_multiquake_mask_ranking.csv"), index=False
    )
    aggregate_rankings(df, ["split", "mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, "topk_split_mask_ranking.csv"), index=False
    )
    aggregate_rankings(df, ["quake_group", "mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, "topk_quake_group_mask_ranking.csv"), index=False
    )
    best_case = df.sort_values(
        by=["case_id", "nrmse_unmonitored", "worst_reconstructed_R2", "true_p_fd"],
        ascending=[True, True, False, True],
    ).groupby("case_id", as_index=False).head(1)
    best_case.to_csv(os.path.join(OUT_DIR, "topk_case_best_mask.csv"), index=False)



# =============================================================================
# 9. Stage 5：多随机种子 GA + Top-K 复评 + 局部邻域补充 + 噪声鲁棒性
# =============================================================================

def stage5_initial_population(self) -> List[Tuple[int, ...]]:
    """替换原 GA 初始种群：针对 N_DOF=20, N_SENSOR=4 提供高度分散启发式种子。"""
    seeds: List[Tuple[int, ...]] = []
    anchor_sets = [
        np.linspace(0, N_DOF - 1, N_SENSOR).round().astype(int).tolist(),
        [1, 6, 12, 18],
        [2, 7, 13, 19],
        [2, 6, 11, 17],
        [3, 7, 12, 18],
        [0, 5, 10, 15],
        [4, 8, 13, 19],
        [2, 5, 9, 15],
        [3, 6, 10, 19],
    ]
    for h in anchor_sets:
        seeds.append(self.repair(h))
    while len(seeds) < GA_POP_SIZE:
        seeds.append(self.random_candidate())
    pop: List[Tuple[int, ...]] = []
    for c in seeds:
        if c not in pop:
            pop.append(c)
    while len(pop) < GA_POP_SIZE:
        c = self.random_candidate()
        if c not in pop:
            pop.append(c)
    return pop[:GA_POP_SIZE]


# monkey patch，让 GeneticSensorSearch 在 Stage 5 使用更适合 20DOF+4sensor 的初始种群。
GeneticSensorSearch.initial_population = stage5_initial_population


def make_noisy_case_for_sensors(case: Dict, sensor_indices: List[int], noise_level: float, seed: int) -> Tuple[Dict, float]:
    """
    生成带传感器噪声的 case。未监测 DOF 真值不变；传感器 DOF 作为观测被加噪，硬约束进入虚拟传感。
    指标里的未监测 NRMSE/R2 仍然是与 clean 未监测真值比较。
    """
    if noise_level <= 0.0:
        return dict(case), 0.0
    rng = np.random.default_rng(seed)
    u_clean = case["u_np"]
    u_noisy = u_clean.copy()
    noise_energy = []
    signal_energy = []
    for j in sensor_indices:
        sig = u_clean[:, j]
        scale = float(np.std(sig))
        if scale < 1e-12:
            scale = float(np.sqrt(np.mean(sig ** 2)) + 1e-12)
        eps = rng.normal(0.0, noise_level * scale, size=sig.shape)
        u_noisy[:, j] = sig + eps
        noise_energy.append(float(np.mean(eps ** 2)))
        signal_energy.append(float(np.mean(sig ** 2)) + 1e-18)
    sensor_noise_nrmse = float(np.sqrt(np.sum(noise_energy) / (np.sum(signal_energy) + 1e-18)))
    noisy_case = dict(case)
    noisy_case["u_np"] = u_noisy
    return noisy_case, sensor_noise_nrmse


def evaluate_candidates_generic(
    candidates: List[Tuple[int, ...]],
    cases_all: List[Dict],
    M_np: np.ndarray,
    C_np: np.ndarray,
    K_np: np.ndarray,
    prefix: str,
    max_points: int,
    adam_steps: int,
    adam_lr: float,
    use_lbfgs: bool,
    lbfgs_max_iter: int,
    noise_levels: Optional[List[float]] = None,
    save_outputs: bool = False,
) -> pd.DataFrame:
    """通用候选复评器：支持 clean 或多噪声水平。"""
    ensure_dir(OUT_DIR)
    if noise_levels is None:
        noise_levels = [0.0]

    out_metrics = os.path.join(OUT_DIR, f"{prefix}_per_case_noise_mask_metrics.csv")
    out_dof = os.path.join(OUT_DIR, f"{prefix}_per_dof_noise_metrics.csv")
    out_hist = os.path.join(OUT_DIR, f"{prefix}_optimizer_noise_history.csv")

    summary_rows: List[Dict] = []
    dof_rows: List[Dict] = []
    hist_rows: List[Dict] = []
    existing_keys = set()

    if RESUME_FINAL_EVAL and os.path.exists(out_metrics):
        old = pd.read_csv(out_metrics)
        summary_rows = old.to_dict("records")
        existing_keys = set(zip(
            old["case_id"].astype(str),
            old["mask_name"].astype(str),
            old["noise_level"].astype(float),
        ))
        if os.path.exists(out_dof):
            dof_rows = pd.read_csv(out_dof).to_dict("records")
        if os.path.exists(out_hist):
            hist_rows = pd.read_csv(out_hist).to_dict("records")
        print(f"检测到 {prefix} 已有 {len(existing_keys)} 个 case-mask-noise，将断点续跑。")

    cases_to_run = []
    for sp in SPLITS_TO_RUN:
        sp_cases = [c for c in cases_all if c["split"] == sp]
        if FINAL_MAX_CASES_PER_SPLIT is not None:
            sp_cases = sp_cases[:FINAL_MAX_CASES_PER_SPLIT]
        cases_to_run.extend(sp_cases)

    total_jobs = len(candidates) * len(cases_to_run) * len(noise_levels)
    job = 0
    for cand_id, cand in enumerate(candidates, start=1):
        cand = tuple(sorted(cand))
        layout_name = make_sensor_name(list(cand))
        print("\n" + "#" * 100)
        print(f"[{prefix}] 候选 {cand_id}/{len(candidates)}: {layout_name} floors={[i + 1 for i in cand]}")
        print("#" * 100)
        for noise_level in noise_levels:
            for case in cases_to_run:
                job += 1
                key = (str(case["case_id"]), layout_name, float(noise_level))
                if key in existing_keys:
                    print(f"[跳过] {job}/{total_jobs} | noise={noise_level:.1%} | {case['case_id']} | {layout_name}")
                    continue
                noise_seed = NOISE_BASE_SEED + case["case_index"] * 100000 + cand_id * 1000 + int(round(noise_level * 10000))
                case_eval, sensor_noise_nrmse = make_noisy_case_for_sensors(case, list(cand), noise_level, noise_seed)
                print(f"[复评] {job}/{total_jobs} | noise={noise_level:.1%} | case={case['case_id']} split={case['split']} | mask={layout_name}")
                safe_prefix = f"{prefix}_{safe_name(case['case_id'])}_noise{int(noise_level*100)}_{layout_name}"
                summary, dr, hr = run_one_case_one_layout(
                    case=case_eval,
                    sensor_indices=list(cand),
                    M_np=M_np,
                    C_np=C_np,
                    K_np=K_np,
                    max_points=max_points,
                    adam_steps=adam_steps,
                    adam_lr=adam_lr,
                    use_lbfgs=use_lbfgs,
                    lbfgs_max_iter=lbfgs_max_iter,
                    combo_id=cand_id + int(noise_level * 1000),
                    save_outputs=save_outputs,
                    out_prefix=safe_prefix,
                )
                summary["noise_level"] = float(noise_level)
                summary["sensor_noise_nrmse"] = float(sensor_noise_nrmse)
                summary_rows.append(summary)
                for r in dr:
                    r["noise_level"] = float(noise_level)
                for r in hr:
                    r["noise_level"] = float(noise_level)
                dof_rows.extend(dr)
                hist_rows.extend(hr)
                existing_keys.add(key)
                if SAVE_INCREMENTAL:
                    pd.DataFrame(summary_rows).to_csv(out_metrics, index=False)
                    pd.DataFrame(dof_rows).to_csv(out_dof, index=False)
                    pd.DataFrame(hist_rows).to_csv(out_hist, index=False)
                    save_stage5_rankings(summary_rows, prefix)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_metrics, index=False)
    pd.DataFrame(dof_rows).to_csv(out_dof, index=False)
    pd.DataFrame(hist_rows).to_csv(out_hist, index=False)
    save_stage5_rankings(summary_rows, prefix)
    return df


def aggregate_rankings_with_noise(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    agg = df.groupby(group_cols, dropna=False).agg(
        n_records=("case_id", "count"),
        n_cases=("case_id", "nunique"),
        n_noise_levels=("noise_level", "nunique"),
        mean_unmon_nrmse=("nrmse_unmonitored", "mean"),
        median_unmon_nrmse=("nrmse_unmonitored", "median"),
        worst_unmon_nrmse=("nrmse_unmonitored", "max"),
        std_unmon_nrmse=("nrmse_unmonitored", "std"),
        mean_worst_R2=("worst_reconstructed_R2", "mean"),
        worst_worst_R2=("worst_reconstructed_R2", "min"),
        mean_recon_R2=("mean_reconstructed_R2", "mean"),
        mean_true_p=("true_p_fd", "mean"),
        worst_true_p=("true_p_fd", "max"),
        mean_sensor_noise_nrmse=("sensor_noise_nrmse", "mean"),
    ).reset_index()
    agg = agg.sort_values(
        by=["mean_unmon_nrmse", "worst_unmon_nrmse", "worst_worst_R2", "mean_true_p"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    agg.insert(len(group_cols), "rank", np.arange(1, len(agg) + 1))
    return agg


def save_stage5_rankings(summary_rows: List[Dict], prefix: str) -> None:
    df = pd.DataFrame(summary_rows)
    if len(df) == 0:
        return
    # 如果没有 noise_level，补 0，统一接口。
    if "noise_level" not in df.columns:
        df["noise_level"] = 0.0
    if "sensor_noise_nrmse" not in df.columns:
        df["sensor_noise_nrmse"] = 0.0

    aggregate_rankings_with_noise(df, ["mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, f"{prefix}_overall_mask_ranking.csv"), index=False
    )
    aggregate_rankings_with_noise(df, ["noise_level", "mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, f"{prefix}_noise_level_mask_ranking.csv"), index=False
    )
    aggregate_rankings_with_noise(df, ["split", "noise_level", "mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, f"{prefix}_split_noise_mask_ranking.csv"), index=False
    )
    aggregate_rankings_with_noise(df, ["quake_group", "noise_level", "mask_name", "sensor_floors_1based"]).to_csv(
        os.path.join(OUT_DIR, f"{prefix}_group_noise_mask_ranking.csv"), index=False
    )
    best_case_noise = df.sort_values(
        by=["case_id", "noise_level", "nrmse_unmonitored", "worst_reconstructed_R2", "true_p_fd"],
        ascending=[True, True, True, False, True],
    ).groupby(["case_id", "noise_level"], as_index=False).head(1)
    best_case_noise.to_csv(os.path.join(OUT_DIR, f"{prefix}_case_noise_best_mask.csv"), index=False)


def run_multi_seed_ga(cases_train: List[Dict], M_np: np.ndarray, C_np: np.ndarray, K_np: np.ndarray) -> pd.DataFrame:
    global BASE_SEED
    print("\n" + "=" * 100)
    print("Stage 5 | GA 多随机种子搜索")
    print("=" * 100)
    all_best = []
    for seed in GA_RANDOM_SEEDS:
        BASE_SEED = int(seed)
        set_all_seeds(BASE_SEED)
        print("\n" + "-" * 100)
        print(f"启动 GA seed={BASE_SEED} | pop={GA_POP_SIZE}, generations={GA_NUM_GENERATIONS}")
        print("-" * 100)
        ga = GeneticSensorSearch(cases_train, M_np, C_np, K_np)
        best_cand, cache_df = ga.run()
        if best_cand is not None:
            all_best.append({"seed": seed, "best_tuple": sensor_tuple_to_str(best_cand), "best_mask_name": make_sensor_name(list(best_cand))})
    if all_best:
        pd.DataFrame(all_best).to_csv(os.path.join(OUT_DIR, "ga_multi_seed_best.csv"), index=False)
    cache_path = os.path.join(OUT_DIR, "ga_candidate_cache.csv")
    if not os.path.exists(cache_path):
        raise RuntimeError("GA candidate cache 不存在，GA 可能没有完成。")
    cache_df = pd.read_csv(cache_path)
    cache_df = cache_df.sort_values("fitness", ascending=False).reset_index(drop=True)
    cache_df.to_csv(cache_path, index=False)
    return cache_df


def take_top_candidates_from_ranking(path: str, top_n: int) -> List[Tuple[int, ...]]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    cands: List[Tuple[int, ...]] = []
    for _, row in df.head(top_n).iterrows():
        tup = parse_sensor_tuple(row["sensor_tuple"]) if "sensor_tuple" in row else None
        if tup is None:
            floors = ast.literal_eval(str(row["sensor_floors_1based"]))
            tup = tuple(sorted([int(f) - 1 for f in floors]))
        if tup not in cands:
            cands.append(tup)
    return cands


def local_neighbors(candidates: List[Tuple[int, ...]], radius: int = 2, max_new: int = 80) -> List[Tuple[int, ...]]:
    """围绕 Top 候选做局部邻域补充：每次替换一个传感器到邻近楼层。"""
    base = {tuple(sorted(c)) for c in candidates}
    neigh = set()
    for cand in base:
        cand_set = set(cand)
        for old in cand:
            for new in range(max(0, old - radius), min(N_DOF - 1, old + radius) + 1):
                if new in cand_set:
                    continue
                arr = sorted((cand_set - {old}) | {new})
                if len(arr) == N_SENSOR:
                    t = tuple(arr)
                    if t not in base:
                        neigh.add(t)
        # 额外允许替换到任意楼层，但只保留少量，避免局部被卡死。
        for old in cand:
            for new in range(N_DOF):
                if new in cand_set:
                    continue
                arr = sorted((cand_set - {old}) | {new})
                t = tuple(arr)
                if t not in base:
                    neigh.add(t)
    # 优先保留高度分散的邻居，减少集中布设浪费。
    def spread_score(t: Tuple[int, ...]) -> float:
        gaps = np.diff(np.array(t, dtype=float))
        return float(np.min(gaps) + 0.1 * np.sum(gaps))
    ordered = sorted(neigh, key=lambda x: spread_score(x), reverse=True)
    if max_new is not None and len(ordered) > max_new:
        ordered = ordered[:max_new]
    pd.DataFrame({
        "sensor_tuple": [sensor_tuple_to_str(c) for c in ordered],
        "mask_name": [make_sensor_name(list(c)) for c in ordered],
        "sensor_floors_1based": [str([i + 1 for i in c]) for c in ordered],
    }).to_csv(os.path.join(OUT_DIR, "local_neighbor_candidates.csv"), index=False)
    return ordered


def write_stage5_summary(clean_rank_path: str, local_rank_path: str, robust_rank_path: str) -> None:
    lines = []
    lines.append("Stage 5 Summary | 20DOF + 4 sensors + GA + Top-K + Local + Noise Robustness")
    lines.append("=" * 90)
    lines.append("研究主目标：以未监测自由度响应重构精度为目标，固定传感器数量，搜索最优布设；物理残差作为一致性约束。")
    for title, path in [
        ("GA Top-K clean ranking", clean_rank_path),
        ("Local neighborhood clean ranking", local_rank_path),
        ("Noise robust overall ranking", robust_rank_path),
    ]:
        lines.append("\n" + title)
        lines.append("-" * 90)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                b = df.iloc[0]
                lines.append(f"best mask: {b['mask_name']} | floors={b['sensor_floors_1based']}")
                lines.append(f"mean_unmon_nrmse={b['mean_unmon_nrmse']:.6e}")
                lines.append(f"worst_unmon_nrmse={b['worst_unmon_nrmse']:.6e}")
                lines.append(f"mean_worst_R2={b['mean_worst_R2']:.6f}, worst_worst_R2={b['worst_worst_R2']:.6f}")
                lines.append(f"mean_true_p={b['mean_true_p']:.6e}")
        else:
            lines.append("not found")
    summary_path = os.path.join(OUT_DIR, "stage5_summary.txt")
    PathLike = type(os.path)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nStage 5 summary saved: {summary_path}")


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(RESP10_ROOT)
    set_all_seeds(BASE_SEED)

    case_list_csv = find_case_list_csv()
    print(f"读取原始 case_list: {case_list_csv}")

    M_np, K_np, C_np = build_shear_building_mck(
        n_dof=N_DOF,
        m_val=M_VAL,
        k_val=K_VAL,
        alpha_rayleigh=ALPHA_RAYLEIGH,
        beta_rayleigh=BETA_RAYLEIGH,
    )
    n_combo = len(list(itertools.combinations(range(N_DOF), N_SENSOR)))
    print("\n" + "=" * 100)
    print("Stage 5 | 20DOF + 固定4传感器 + GA多随机种子 + Top-K复评 + 噪声鲁棒性 + 局部邻域补充")
    print("=" * 100)
    print(f"结构设置: N_DOF={N_DOF}, N_SENSOR={N_SENSOR}, 组合数 C({N_DOF},{N_SENSOR})={n_combo}")
    print(f"GA_RANDOM_SEEDS={GA_RANDOM_SEEDS}")

    cases_all = load_or_generate_10dof_cases(case_list_csv, M_np, C_np, K_np)
    print("\n20DOF case 统计:")
    print(f"全部: {len(cases_all)}")
    for sp in ["train", "val", "test"]:
        print(f"{sp:5s}: {sum(c['split'] == sp for c in cases_all)}")
    print(f"已生成/更新: {CASE10_CSV}, {CASE10_SUMMARY_CSV}")

    cases_train = [c for c in cases_all if c["split"] == GA_TRAIN_SPLIT]
    if len(cases_train) == 0:
        raise RuntimeError("没有 train case，无法进行 GA 搜索。")

    # 1) 多随机种子 GA 粗筛
    cache_df = run_multi_seed_ga(cases_train, M_np, C_np, K_np)
    print("\nGA candidate cache Top 20:")
    show_cols = ["mask_name", "sensor_floors_1based", "fitness", "mean_unmon_nrmse", "worst_unmon_nrmse", "mean_worst_R2", "mean_true_p"]
    print(cache_df[show_cols].head(20).to_string(index=False))

    # 2) GA Top-K clean 多地震波复评
    ga_top_candidates: List[Tuple[int, ...]] = []
    for _, row in cache_df.head(GA_TOPK_CLEAN_EVAL).iterrows():
        tup = parse_sensor_tuple(row["sensor_tuple"])
        if tup not in ga_top_candidates:
            ga_top_candidates.append(tup)

    print("\n" + "=" * 100)
    print(f"开始 GA Top-{len(ga_top_candidates)} clean 多地震波复评")
    print("=" * 100)
    evaluate_candidates_generic(
        candidates=ga_top_candidates,
        cases_all=cases_all,
        M_np=M_np,
        C_np=C_np,
        K_np=K_np,
        prefix="ga_topk_clean",
        max_points=FINAL_MAX_POINTS,
        adam_steps=FINAL_ADAM_STEPS,
        adam_lr=FINAL_ADAM_LR,
        use_lbfgs=FINAL_USE_LBFGS,
        lbfgs_max_iter=FINAL_LBFGS_MAX_ITER,
        noise_levels=[0.0],
        save_outputs=False,
    )
    clean_rank_path = os.path.join(OUT_DIR, "ga_topk_clean_overall_mask_ranking.csv")
    clean_rank = pd.read_csv(clean_rank_path)
    print("\nGA Top-K clean ranking Top 10:")
    print(clean_rank[["rank", "mask_name", "sensor_floors_1based", "mean_unmon_nrmse", "worst_unmon_nrmse", "mean_worst_R2", "mean_true_p"]].head(10).to_string(index=False))

    # 3) 局部邻域补充验证
    base_for_local: List[Tuple[int, ...]] = []
    for _, row in clean_rank.head(LOCAL_BASE_TOP_N).iterrows():
        floors = ast.literal_eval(str(row["sensor_floors_1based"]))
        tup = tuple(sorted([int(f) - 1 for f in floors]))
        base_for_local.append(tup)
    local_cands = local_neighbors(base_for_local, radius=LOCAL_NEIGHBOR_RADIUS, max_new=LOCAL_MAX_NEW_CANDIDATES)
    print("\n" + "=" * 100)
    print(f"开始局部邻域补充验证：base_top={LOCAL_BASE_TOP_N}, new_candidates={len(local_cands)}")
    print("=" * 100)
    evaluate_candidates_generic(
        candidates=local_cands,
        cases_all=cases_all,
        M_np=M_np,
        C_np=C_np,
        K_np=K_np,
        prefix="local_clean",
        max_points=LOCAL_MAX_POINTS,
        adam_steps=LOCAL_ADAM_STEPS,
        adam_lr=LOCAL_ADAM_LR,
        use_lbfgs=LOCAL_USE_LBFGS,
        lbfgs_max_iter=LOCAL_LBFGS_MAX_ITER,
        noise_levels=[0.0],
        save_outputs=False,
    )
    local_rank_path = os.path.join(OUT_DIR, "local_clean_overall_mask_ranking.csv")

    # 4) 合并 clean 结果，选择鲁棒性候选
    clean_metrics_files = [
        os.path.join(OUT_DIR, "ga_topk_clean_per_case_noise_mask_metrics.csv"),
        os.path.join(OUT_DIR, "local_clean_per_case_noise_mask_metrics.csv"),
    ]
    dfs = [pd.read_csv(p) for p in clean_metrics_files if os.path.exists(p)]
    combined_clean = pd.concat(dfs, ignore_index=True)
    combined_clean = combined_clean.drop_duplicates(subset=["case_id", "mask_name", "noise_level"], keep="first")
    combined_clean.to_csv(os.path.join(OUT_DIR, "combined_clean_per_case_mask_metrics.csv"), index=False)
    combined_clean_rank = aggregate_rankings_with_noise(combined_clean, ["mask_name", "sensor_floors_1based"])
    combined_rank_path = os.path.join(OUT_DIR, "combined_clean_mask_ranking.csv")
    combined_clean_rank.to_csv(combined_rank_path, index=False)
    print("\nCombined clean ranking Top 20:")
    print(combined_clean_rank[["rank", "mask_name", "sensor_floors_1based", "mean_unmon_nrmse", "worst_unmon_nrmse", "mean_worst_R2", "mean_true_p"]].head(20).to_string(index=False))

    robust_candidates: List[Tuple[int, ...]] = []
    for _, row in combined_clean_rank.head(ROBUST_TOPK_CANDIDATES).iterrows():
        floors = ast.literal_eval(str(row["sensor_floors_1based"]))
        tup = tuple(sorted([int(f) - 1 for f in floors]))
        if tup not in robust_candidates:
            robust_candidates.append(tup)

    # 保证 GA clean top1 也进入噪声鲁棒性复评。
    if len(clean_rank) > 0:
        floors = ast.literal_eval(str(clean_rank.iloc[0]["sensor_floors_1based"]))
        tup = tuple(sorted([int(f) - 1 for f in floors]))
        if tup not in robust_candidates:
            robust_candidates.append(tup)

    print("\n" + "=" * 100)
    print(f"开始噪声鲁棒性验证：candidates={len(robust_candidates)}, noise={NOISE_LEVELS}")
    print("=" * 100)
    evaluate_candidates_generic(
        candidates=robust_candidates,
        cases_all=cases_all,
        M_np=M_np,
        C_np=C_np,
        K_np=K_np,
        prefix="robust_noise",
        max_points=ROBUST_MAX_POINTS,
        adam_steps=ROBUST_ADAM_STEPS,
        adam_lr=ROBUST_ADAM_LR,
        use_lbfgs=ROBUST_USE_LBFGS,
        lbfgs_max_iter=ROBUST_LBFGS_MAX_ITER,
        noise_levels=NOISE_LEVELS,
        save_outputs=False,
    )
    robust_rank_path = os.path.join(OUT_DIR, "robust_noise_overall_mask_ranking.csv")

    if os.path.exists(robust_rank_path):
        robust_rank = pd.read_csv(robust_rank_path)
        print("\n" + "=" * 100)
        print("Stage 5 噪声鲁棒性总体排序 Top 20")
        print("=" * 100)
        cols = ["rank", "mask_name", "sensor_floors_1based", "n_records", "n_cases", "n_noise_levels", "mean_unmon_nrmse", "worst_unmon_nrmse", "mean_worst_R2", "worst_worst_R2", "mean_recon_R2", "mean_true_p"]
        print(robust_rank[cols].head(20).to_string(index=False))
        best = robust_rank.iloc[0]
        print("\n✅ Stage 5 当前 20DOF+4sensor 噪声鲁棒最优候选：")
        print(f"mask_name: {best['mask_name']}")
        print(f"sensor floors: {best['sensor_floors_1based']}")
        print(f"mean unmonitored NRMSE : {best['mean_unmon_nrmse']:.6e}")
        print(f"worst unmonitored NRMSE: {best['worst_unmon_nrmse']:.6e}")
        print(f"mean worst R2          : {best['mean_worst_R2']:.6f}")
        print(f"worst worst R2         : {best['worst_worst_R2']:.6f}")
        print(f"mean true_p_fd         : {best['mean_true_p']:.6e}")

    write_stage5_summary(clean_rank_path, local_rank_path, robust_rank_path)

    print("\n输出目录:", OUT_DIR)
    print("关键文件:")
    print("  ga_candidate_cache.csv")
    print("  ga_multi_seed_best.csv")
    print("  ga_topk_clean_overall_mask_ranking.csv")
    print("  local_neighbor_candidates.csv")
    print("  local_clean_overall_mask_ranking.csv")
    print("  combined_clean_mask_ranking.csv")
    print("  robust_noise_overall_mask_ranking.csv")
    print("  robust_noise_noise_level_mask_ranking.csv")
    print("  robust_noise_split_noise_mask_ranking.csv")
    print("  stage5_summary.txt")


if __name__ == "__main__":
    main()
