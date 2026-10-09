# -*- coding: utf-8 -*-
"""
pc_virtual_sensing_enum10.py

物理约束虚拟传感 / 直接重构基线代码
============================================================
研究目标：
    以未监测自由度响应重构精度为目标，在固定传感器数量约束下，
    枚举/评价不同传感器布设方案，并通过结构动力学物理残差提高重构物理一致性。

核心思想：
    1) 不再让网络去学习传感器 DOF；传感器 DOF 直接硬约束为观测响应。
    2) 未监测 DOF 的完整时间序列直接作为优化变量 torch.Parameter。
    3) 用结构动力学残差：M u_ddot + C u_dot + K u + M r a_g(t)=0 约束整体响应。
    4) 枚举 5 自由度、2 个传感器的全部 C(5,2)=10 种布设，按未监测 DOF 重构误差排序。

为什么这一步重要：
    这不是最终替代 PIKAN/FKAN，而是给 GA-PIKAN/FKAN 框架建立一个稳定、可验证的
    physics-constrained virtual sensing baseline。后续 PIKAN/FKAN 可作为该直接优化器的加速器或低维修正器。

输入文件：
    case_list.csv，要求至少包含：case_id, split, eq_file, response_file
    response_file 中要求包含 Time, DOF1, DOF2, ..., DOF5
    eq_file 默认第 0 列时间，第 1 列地震加速度，第一行为表头或跳过行。

输出目录：
    pc_virtual_sensing_enum10_results/
        summary.csv
        per_dof_metrics.csv
        stage_history.csv
        predictions/*.csv
        plots/*.png

运行方式：
    直接在 PyCharm / VSCode / 命令行运行本文件。
"""

import os
import time as pytime
import itertools
import random
from typing import Dict, List, Tuple

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

print("=" * 80)
print("Physics-Constrained Virtual Sensing: enumerate all sensor layouts")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
print(f"Selected DEVICE : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU name        : {torch.cuda.get_device_name(DEVICE)}")
print("=" * 80)

# -------------------------
# 文件路径
# -------------------------
CASE_LIST_CSV = "case_list.csv"
OUT_DIR = "pc_virtual_sensing_enum10_results"

# -------------------------
# 地震波读取设置
# -------------------------
EQ_SKIPROWS = 1
EQ_TIME_COL = 0
EQ_ACC_COL = 1
EQ_SCALE_FACTOR = 0.1

# -------------------------
# 结构参数
# -------------------------
N_DOF = 5
N_SENSOR = 2

M_VAL = 20000.0
K_VAL = 15000000.0
ALPHA_RAYLEIGH = 0.5
BETA_RAYLEIGH = 0.001

# -------------------------
# 运行范围
# -------------------------
# 当前建议：先跑第一个 train case；后续多地震波时可以改 True。
RUN_ALL_TRAIN_CASES = False
MAX_CASES_TO_RUN = 1

# 枚举所有 C(5,2)=10 种布设。
RUN_ALL_SENSOR_COMBINATIONS = True
# 如果只想先跑某几个，可把上面改 False，然后设置这里，1-based 楼层编号。
MANUAL_SENSOR_FLOORS_1BASED = [
    [1, 3],
]

# -------------------------
# 时间点下采样
# -------------------------
# 直接优化未监测时间序列，点数过大也会慢。先用 1200。
# 如果响应原始点数更少，则使用全部点。
MAX_TRAIN_POINTS = 1200

# -------------------------
# 优化参数
# -------------------------
BASE_SEED = 2026

ADAM_STEPS = 2500
ADAM_LR = 3e-3
PRINT_EVERY = 250

USE_LBFGS = True
LBFGS_MAX_ITER = 120

# 如果想快速试跑，把下面改小：
# ADAM_STEPS = 800
# USE_LBFGS = False

# -------------------------
# loss 权重
# -------------------------
# 核心是物理残差，传感器 DOF 已硬约束，不需要 data loss。
PHYSICS_WEIGHT = 1.0

# 未监测 DOF 的插值先验：防止欠定问题直接发散。
# prior 不是用未监测真值，而是由传感器楼层响应沿楼层方向插值得到。
PRIOR_WEIGHT = 1e-2

# 未监测 DOF 时间平滑，抑制高频乱振。
SMOOTH_WEIGHT = 1e-3

# 空间平滑，抑制楼层间响应形态过于锯齿。
SPATIAL_SMOOTH_WEIGHT = 1e-4

# 能量约束，未监测 DOF 的 RMS 不应远大于传感器响应 RMS。
ENERGY_WEIGHT = 1e-2
ENERGY_RATIO_LIMIT = 3.0

# 初始条件约束。若响应由静止初始条件生成，一般可保留。
IC_WEIGHT = 1.0
USE_ZERO_INITIAL_DISPLACEMENT = True
USE_ZERO_INITIAL_VELOCITY = True

# 是否保存图
SAVE_PLOTS = True
SHOW_PLOTS_IN_PYCHARM = False


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


def build_shear_building_mck(
    n_dof: int,
    m_val: float,
    k_val: float,
    alpha_rayleigh: float,
    beta_rayleigh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造 n_dof 剪切型结构 M, K, C。"""
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
    eq_time = data[:, EQ_TIME_COL].astype(np.float64)
    eq_acc = data[:, EQ_ACC_COL].astype(np.float64) * EQ_SCALE_FACTOR
    eq_time = eq_time - eq_time[0]
    return eq_time, eq_acc


def load_cases(case_list_csv: str) -> List[Dict]:
    if not os.path.exists(case_list_csv):
        raise FileNotFoundError(f"找不到 {case_list_csv}，请确认当前工作目录是否正确。")

    df_case = pd.read_csv(case_list_csv)
    required_cols = ["case_id", "split", "eq_file", "response_file"]
    for c in required_cols:
        if c not in df_case.columns:
            raise ValueError(f"{case_list_csv} 缺少必要列: {c}")

    cases = []
    for _, row in df_case.iterrows():
        case_id = str(row["case_id"])
        split = str(row["split"])
        eq_file = str(row["eq_file"])
        response_file = str(row["response_file"])

        if not os.path.exists(eq_file):
            raise FileNotFoundError(f"找不到地震波文件: {eq_file}")
        if not os.path.exists(response_file):
            raise FileNotFoundError(f"找不到响应文件: {response_file}")

        eq_time, eq_acc = load_eq_file(eq_file)
        resp_df = pd.read_csv(response_file)

        if "Time" not in resp_df.columns:
            raise ValueError(f"{response_file} 缺少 Time 列。")

        dof_cols = [f"DOF{i + 1}" for i in range(N_DOF)]
        for c in dof_cols:
            if c not in resp_df.columns:
                raise ValueError(f"{response_file} 缺少 {c} 列。")

        t_np = resp_df["Time"].values.astype(np.float64)
        t_np = t_np - t_np[0]
        u_np = resp_df[dof_cols].values.astype(np.float64)
        duration = float(t_np[-1])

        cases.append({
            "case_index": len(cases),
            "case_id": case_id,
            "split": split,
            "eq_file": eq_file,
            "response_file": response_file,
            "eq_time_np": eq_time.astype(np.float64),
            "eq_acc_np": eq_acc.astype(np.float64),
            "t_np": t_np,
            "u_np": u_np,
            "duration": duration,
        })

    return cases


def select_time_indices(n_time: int, max_points: int) -> np.ndarray:
    if n_time <= max_points:
        return np.arange(n_time, dtype=int)
    idx = np.linspace(0, n_time - 1, max_points).round().astype(int)
    idx = np.unique(idx)
    return idx


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-18:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def safe_mean(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_min(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.min(arr))


def make_sensor_name(sensor_indices: List[int]) -> str:
    return "_".join([f"DOF{i + 1}" for i in sensor_indices])


def interpolation_prior_from_sensors(
    u_sensor: np.ndarray,
    sensor_indices: List[int],
    recon_indices: List[int],
    n_dof: int,
) -> np.ndarray:
    """
    用传感器楼层响应沿楼层方向做插值，作为未监测 DOF 初值/弱先验。
    不使用未监测真值。
    """
    sensor_x = np.array(sensor_indices, dtype=np.float64)
    recon_x = np.array(recon_indices, dtype=np.float64)
    n_time = u_sensor.shape[0]
    prior = np.zeros((n_time, len(recon_indices)), dtype=np.float64)

    for it in range(n_time):
        sensor_y = u_sensor[it, :]
        # np.interp 对范围外采用边界值；对当前单调楼层索引很稳。
        prior[it, :] = np.interp(recon_x, sensor_x, sensor_y)

    return prior


def calc_metrics(u_true: np.ndarray, u_pred: np.ndarray, sensor_indices: List[int]) -> Dict:
    sensor_set = set(sensor_indices)
    recon_indices = [i for i in range(u_true.shape[1]) if i not in sensor_set]
    n_time = u_true.shape[0]

    mask = np.zeros(u_true.shape[1], dtype=np.float64)
    mask[sensor_indices] = 1.0
    unmask = 1.0 - mask

    err = u_pred - u_true

    sensor_rmse = np.sqrt(np.sum((err * mask.reshape(1, -1)) ** 2) / (np.sum(mask) * n_time + 1e-18))
    sensor_rms_true = np.sqrt(np.sum((u_true * mask.reshape(1, -1)) ** 2) / (np.sum(mask) * n_time + 1e-18))
    sensor_nrmse = sensor_rmse / (sensor_rms_true + 1e-18)

    recon_rmse = np.sqrt(np.sum((err * unmask.reshape(1, -1)) ** 2) / (np.sum(unmask) * n_time + 1e-18))
    recon_rms_true = np.sqrt(np.sum((u_true * unmask.reshape(1, -1)) ** 2) / (np.sum(unmask) * n_time + 1e-18))
    recon_nrmse = recon_rmse / (recon_rms_true + 1e-18)

    r2_per_dof = []
    rmse_per_dof = []
    for i in range(u_true.shape[1]):
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
        "worst_reconstructed_RMSE": float(np.nanmax(recon_rmse_list)) if len(recon_rmse_list) > 0 else float("nan"),
    }


# =============================================================================
# 2. 直接优化器：传感器硬约束 + 未监测 DOF 参数优化
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
    ):
        self.t_np = t_np.astype(np.float64)
        self.u_true_np = u_true_np.astype(np.float64)
        self.eq_time_np = eq_time_np.astype(np.float64)
        self.eq_acc_np = eq_acc_np.astype(np.float64)
        self.sensor_indices = list(sensor_indices)
        self.recon_indices = [i for i in range(N_DOF) if i not in set(sensor_indices)]

        self.n_time = len(t_np)
        if self.n_time < 5:
            raise ValueError("时间点太少，无法进行有限差分物理残差计算。")

        dt_arr = np.diff(self.t_np)
        self.dt = float(np.median(dt_arr))
        dt_rel = float(np.max(np.abs(dt_arr - self.dt)) / (abs(self.dt) + 1e-18))
        if dt_rel > 1e-3:
            print(f"⚠️ 时间步长不是严格均匀，dt 相对波动={dt_rel:.3e}。当前仍使用 median dt={self.dt:.6e}。")

        # 地震动插值到响应时间。
        self.ag_np = np.interp(self.t_np, self.eq_time_np, self.eq_acc_np).astype(np.float64)

        # 传感器观测硬约束。
        self.u_sensor_np = self.u_true_np[:, self.sensor_indices]
        self.u_prior_unknown_np = interpolation_prior_from_sensors(
            u_sensor=self.u_sensor_np,
            sensor_indices=self.sensor_indices,
            recon_indices=self.recon_indices,
            n_dof=N_DOF,
        )

        # torch tensors
        self.t_t = torch.tensor(self.t_np, dtype=torch.float32, device=DEVICE)
        self.u_true_t = torch.tensor(self.u_true_np, dtype=torch.float32, device=DEVICE)
        self.u_sensor_t = torch.tensor(self.u_sensor_np, dtype=torch.float32, device=DEVICE)
        self.u_prior_unknown_t = torch.tensor(self.u_prior_unknown_np, dtype=torch.float32, device=DEVICE)
        self.ag_t = torch.tensor(self.ag_np, dtype=torch.float32, device=DEVICE).view(-1, 1)

        self.M_t = torch.tensor(M_np, dtype=torch.float32, device=DEVICE)
        self.C_t = torch.tensor(C_np, dtype=torch.float32, device=DEVICE)
        self.K_t = torch.tensor(K_np, dtype=torch.float32, device=DEVICE)

        ones = torch.ones((N_DOF, 1), dtype=torch.float32, device=DEVICE)
        self.force_vec = (self.M_t @ ones).view(-1)

        self.sensor_indices_t = torch.tensor(self.sensor_indices, dtype=torch.long, device=DEVICE)
        self.recon_indices_t = torch.tensor(self.recon_indices, dtype=torch.long, device=DEVICE)

        # 优化变量：未监测 DOF 完整时间序列，初始化为传感器插值 prior。
        self.u_unknown = torch.nn.Parameter(self.u_prior_unknown_t.clone())

        # 归一化尺度：仅用传感器和插值先验，不使用未监测真值作为训练信息。
        full_init = self._assemble_full_response(self.u_unknown.detach())
        u_scale = torch.std(full_init, dim=0)
        sensor_scale_mean = torch.mean(torch.std(self.u_sensor_t, dim=0))
        u_scale = torch.clamp(u_scale, min=torch.clamp(sensor_scale_mean * 0.05, min=1e-8))
        self.u_scale = u_scale.detach()

        ag_scale = torch.std(self.ag_t).detach()
        ag_scale = torch.clamp(ag_scale, min=1e-8)

        # 物理残差归一化尺度。
        # 注意：这里用先验尺度估计，不用未监测真值。
        duration = float(self.t_np[-1] - self.t_np[0])
        duration = max(duration, self.dt * 10.0)
        disp_term = torch.abs(self.K_t) @ self.u_scale.view(-1, 1)
        vel_term = torch.abs(self.C_t) @ (self.u_scale / duration).view(-1, 1)
        acc_term = torch.abs(self.M_t) @ (self.u_scale / (duration ** 2)).view(-1, 1)
        eq_term = torch.abs(self.force_vec).view(-1, 1) * ag_scale
        res_scale = (disp_term + vel_term + acc_term + eq_term).view(-1)
        self.res_scale = torch.clamp(res_scale, min=1.0).detach()

        sensor_rms = torch.sqrt(torch.mean(self.u_sensor_t ** 2) + 1e-18).detach()
        self.sensor_rms = torch.clamp(sensor_rms, min=1e-8)

        self.history = []

    def _assemble_full_response(self, u_unknown: torch.Tensor) -> torch.Tensor:
        """组装完整响应，其中传感器 DOF 硬约束为观测响应。"""
        u_full = torch.zeros((self.n_time, N_DOF), dtype=torch.float32, device=DEVICE)
        u_full[:, self.sensor_indices_t] = self.u_sensor_t
        u_full[:, self.recon_indices_t] = u_unknown
        return u_full

    def finite_difference(self, u_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """中心差分，返回 interior 的 u, u_dot, u_ddot, ag。"""
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
        # 不除 dt，作为形状平滑约束，避免因 dt 很小导致数值过大。
        d2 = u_unknown[2:, :] - 2.0 * u_unknown[1:-1, :] + u_unknown[:-2, :]
        scale = self.u_scale[self.recon_indices_t].view(1, -1)
        return torch.mean((d2 / scale) ** 2)

    def spatial_smooth_loss(self, u_full: torch.Tensor) -> torch.Tensor:
        if N_DOF < 3:
            return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
        d2_space = u_full[:, 2:] - 2.0 * u_full[:, 1:-1] + u_full[:, :-2]
        scale = torch.mean(self.u_scale)
        scale = torch.clamp(scale, min=1e-8)
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
            v_scale = self.u_scale / max(self.dt * 10.0, 1e-12)
            v_scale = torch.clamp(v_scale, min=1e-8)
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

    def optimize(self, layout_name: str) -> Dict:
        optimizer = torch.optim.Adam([self.u_unknown], lr=ADAM_LR)
        t0 = pytime.time()

        for step in range(1, ADAM_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, parts = self.total_loss()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{layout_name} Adam step={step} 出现非有限 loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            if step == 1 or step % PRINT_EVERY == 0 or step == ADAM_STEPS:
                elapsed = pytime.time() - t0
                print(
                    f"[{layout_name} Adam] step {step:5d}/{ADAM_STEPS} | "
                    f"total={parts['total']:.4e} | p={parts['physics']:.4e} | "
                    f"prior={parts['prior']:.4e} | smooth={parts['smooth']:.4e} | "
                    f"spatial={parts['spatial_smooth']:.4e} | energy={parts['energy']:.4e} | "
                    f"ic={parts['ic']:.4e} | time={elapsed:.1f}s"
                )
                self.history.append({
                    "layout_name": layout_name,
                    "stage": "Adam",
                    "step": step,
                    **parts,
                    "elapsed_sec": elapsed,
                })

        if USE_LBFGS and LBFGS_MAX_ITER > 0:
            print(f"[{layout_name} L-BFGS] start max_iter={LBFGS_MAX_ITER}")
            lbfgs = torch.optim.LBFGS(
                [self.u_unknown],
                lr=1.0,
                max_iter=LBFGS_MAX_ITER,
                tolerance_grad=1e-9,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
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
            print(
                f"[{layout_name} L-BFGS] done calls={calls['n']} | "
                f"total={parts['total']:.4e} | p={parts['physics']:.4e} | "
                f"prior={parts['prior']:.4e} | smooth={parts['smooth']:.4e} | "
                f"spatial={parts['spatial_smooth']:.4e} | energy={parts['energy']:.4e} | "
                f"ic={parts['ic']:.4e} | time={elapsed:.1f}s"
            )
            self.history.append({
                "layout_name": layout_name,
                "stage": "LBFGS",
                "step": calls["n"],
                **parts,
                "elapsed_sec": elapsed,
            })

        with torch.no_grad():
            u_full = self._assemble_full_response(self.u_unknown).detach().cpu().numpy()
            _, final_parts = self.total_loss()

        return {
            "u_pred": u_full,
            "loss_parts": final_parts,
            "history": list(self.history),
        }


def save_prediction_csv(
    out_path: str,
    t_np: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    sensor_indices: List[int],
) -> None:
    sensor_set = set(sensor_indices)
    df = pd.DataFrame({"Time": t_np})
    for i in range(N_DOF):
        role = "sensor" if i in sensor_set else "reconstructed"
        df[f"DOF{i + 1}_True"] = u_true[:, i]
        df[f"DOF{i + 1}_Recon"] = u_pred[:, i]
        df[f"DOF{i + 1}_Role"] = role
    df.to_csv(out_path, index=False)


def save_plot(
    out_path: str,
    t_np: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    sensor_indices: List[int],
    title: str,
) -> None:
    sensor_set = set(sensor_indices)
    fig, axes = plt.subplots(N_DOF, 1, figsize=(12, 2.2 * N_DOF), sharex=True)
    if N_DOF == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        role = "sensor-hard" if i in sensor_set else "reconstructed"
        ax.plot(t_np, u_true[:, i], label="True", linewidth=1.2)
        ax.plot(t_np, u_pred[:, i], label="Recon", linewidth=1.0, linestyle="--")
        r2 = compute_r2(u_true[:, i], u_pred[:, i])
        rmse = np.sqrt(np.mean((u_pred[:, i] - u_true[:, i]) ** 2))
        ax.set_ylabel(f"DOF{i + 1}\n{role}")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.01, 0.85,
            f"R2={r2:.3f}, RMSE={rmse:.2e}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    if SHOW_PLOTS_IN_PYCHARM:
        plt.show()
    plt.close(fig)


# =============================================================================
# 3. 主流程
# =============================================================================

def run_one_case(case: Dict, M_np: np.ndarray, C_np: np.ndarray, K_np: np.ndarray) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    case_id = case["case_id"]
    print("\n" + "=" * 80)
    print(f"开始处理 case: {case_id} | split={case['split']}")
    print("=" * 80)

    idx = select_time_indices(len(case["t_np"]), MAX_TRAIN_POINTS)
    t_np = case["t_np"][idx]
    u_true = case["u_np"][idx, :]
    eq_time_np = case["eq_time_np"]
    eq_acc_np = case["eq_acc_np"]

    print(f"原始时间点数: {len(case['t_np'])} | 当前优化点数: {len(t_np)}")
    print(f"时间范围: {t_np[0]:.6e} ~ {t_np[-1]:.6e} | median dt={np.median(np.diff(t_np)):.6e}")

    if RUN_ALL_SENSOR_COMBINATIONS:
        sensor_combos = list(itertools.combinations(range(N_DOF), N_SENSOR))
    else:
        sensor_combos = [tuple([x - 1 for x in floors]) for floors in MANUAL_SENSOR_FLOORS_1BASED]

    summary_rows = []
    per_dof_rows = []
    history_rows = []

    pred_dir = os.path.join(OUT_DIR, "predictions")
    plot_dir = os.path.join(OUT_DIR, "plots")
    ensure_dir(pred_dir)
    ensure_dir(plot_dir)

    for combo_id, sensor_indices_tuple in enumerate(sensor_combos, start=1):
        sensor_indices = list(sensor_indices_tuple)
        recon_indices = [i for i in range(N_DOF) if i not in set(sensor_indices)]
        layout_name = make_sensor_name(sensor_indices)
        layout_seed = BASE_SEED + case["case_index"] * 1000 + combo_id
        set_all_seeds(layout_seed)

        print("\n" + "-" * 80)
        print(
            f"布设 {combo_id}/{len(sensor_combos)}: {layout_name} | "
            f"sensor floors={[i + 1 for i in sensor_indices]} | "
            f"reconstructed floors={[i + 1 for i in recon_indices]}"
        )
        print("-" * 80)

        optimizer = DirectVirtualSensingOptimizer(
            t_np=t_np,
            u_true_np=u_true,
            eq_time_np=eq_time_np,
            eq_acc_np=eq_acc_np,
            sensor_indices=sensor_indices,
            M_np=M_np,
            C_np=C_np,
            K_np=K_np,
        )

        # 插值先验 baseline，用于对比。
        u_prior_full = np.zeros_like(u_true)
        u_prior_full[:, sensor_indices] = u_true[:, sensor_indices]
        u_prior_full[:, recon_indices] = optimizer.u_prior_unknown_np
        prior_metrics = calc_metrics(u_true, u_prior_full, sensor_indices)
        print(
            f"[{layout_name} prior] unmon_NRMSE={prior_metrics['nrmse_unmonitored']:.4e} | "
            f"worst_R2={prior_metrics['worst_reconstructed_R2']:.4f} | "
            f"mean_R2={prior_metrics['mean_reconstructed_R2']:.4f}"
        )

        result = optimizer.optimize(layout_name)
        u_pred = result["u_pred"]
        loss_parts = result["loss_parts"]
        metrics = calc_metrics(u_true, u_pred, sensor_indices)

        print("\n--- 物理约束直接虚拟传感结果 ---")
        print(f"mask_name              : {layout_name}")
        print(f"sensor floors          : {[i + 1 for i in sensor_indices]}")
        print(f"reconstructed floors   : {[i + 1 for i in recon_indices]}")
        print(f"未监测点 RMSE          : {metrics['rmse_unmonitored']:.6e}")
        print(f"未监测点 NRMSE         : {metrics['nrmse_unmonitored']:.6e}")
        print(f"监测点 RMSE            : {metrics['rmse_monitored']:.6e}")
        print(f"监测点 NRMSE           : {metrics['nrmse_monitored']:.6e}")
        print(f"物理残差 true_p_fd     : {loss_parts['physics']:.6e}")
        print(f"prior loss             : {loss_parts['prior']:.6e}")
        print(f"smooth loss            : {loss_parts['smooth']:.6e}")
        print(f"spatial smooth loss    : {loss_parts['spatial_smooth']:.6e}")
        print(f"energy loss            : {loss_parts['energy']:.6e}")
        print(f"ic loss                : {loss_parts['ic']:.6e}")
        print(f"重构 DOF 平均 R2        : {metrics['mean_reconstructed_R2']:.6f}")
        print(f"重构 DOF 最差 R2        : {metrics['worst_reconstructed_R2']:.6f}")
        print(f"重构 DOF 最差 RMSE      : {metrics['worst_reconstructed_RMSE']:.6e}")

        for i in range(N_DOF):
            role = "监测点-硬约束" if i in set(sensor_indices) else "重构点"
            print(
                f"DOF{i + 1} [{role}] | "
                f"R2={metrics['r2_per_dof'][i]:.6f} | RMSE={metrics['rmse_per_dof'][i]:.6e}"
            )

        safe_case_id = str(case_id).replace("/", "_").replace("\\", "_")
        pred_path = os.path.join(pred_dir, f"{safe_case_id}_{layout_name}_prediction.csv")
        plot_path = os.path.join(plot_dir, f"{safe_case_id}_{layout_name}_plot.png")
        save_prediction_csv(pred_path, t_np, u_true, u_pred, sensor_indices)
        if SAVE_PLOTS:
            title = f"{case_id} | {layout_name} | unmon NRMSE={metrics['nrmse_unmonitored']:.3f}"
            save_plot(plot_path, t_np, u_true, u_pred, sensor_indices, title)

        summary_rows.append({
            "case_id": case_id,
            "split": case["split"],
            "mask_name": layout_name,
            "sensor_indices_0based": str(sensor_indices),
            "sensor_floors_1based": str([i + 1 for i in sensor_indices]),
            "reconstructed_indices_0based": str(recon_indices),
            "reconstructed_floors_1based": str([i + 1 for i in recon_indices]),
            "n_time_used": len(t_np),
            "prior_nrmse_unmonitored": prior_metrics["nrmse_unmonitored"],
            "prior_worst_reconstructed_R2": prior_metrics["worst_reconstructed_R2"],
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
            "plot_png": plot_path if SAVE_PLOTS else "",
        })

        for i in range(N_DOF):
            per_dof_rows.append({
                "case_id": case_id,
                "mask_name": layout_name,
                "DOF": i + 1,
                "role": "sensor" if i in set(sensor_indices) else "reconstructed",
                "R2": metrics["r2_per_dof"][i],
                "RMSE": metrics["rmse_per_dof"][i],
            })

        for h in result["history"]:
            row = {"case_id": case_id, **h}
            history_rows.append(row)

        # 清显存，避免多布设循环时占用逐渐增大。
        del optimizer
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return summary_rows, per_dof_rows, history_rows


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(os.path.join(OUT_DIR, "predictions"))
    ensure_dir(os.path.join(OUT_DIR, "plots"))

    set_all_seeds(BASE_SEED)

    M_np, K_np, C_np = build_shear_building_mck(
        n_dof=N_DOF,
        m_val=M_VAL,
        k_val=K_VAL,
        alpha_rayleigh=ALPHA_RAYLEIGH,
        beta_rayleigh=BETA_RAYLEIGH,
    )

    cases = load_cases(CASE_LIST_CSV)
    train_cases = [c for c in cases if c["split"].lower() == "train"]
    if len(train_cases) == 0:
        print("⚠️ case_list.csv 中没有 train case，将使用全部 case。")
        train_cases = cases

    if not RUN_ALL_TRAIN_CASES:
        cases_to_run = train_cases[:MAX_CASES_TO_RUN]
    else:
        cases_to_run = train_cases

    print("\n案例统计:")
    print(f"全部 case 数量 : {len(cases)}")
    print(f"train case 数量: {len(train_cases)}")
    print(f"本次运行 case  : {len(cases_to_run)}")
    print(f"N_DOF={N_DOF}, N_SENSOR={N_SENSOR}")
    print(f"RUN_ALL_SENSOR_COMBINATIONS={RUN_ALL_SENSOR_COMBINATIONS}")

    all_summary = []
    all_per_dof = []
    all_history = []

    t_all0 = pytime.time()
    for case in cases_to_run:
        s_rows, d_rows, h_rows = run_one_case(case, M_np, C_np, K_np)
        all_summary.extend(s_rows)
        all_per_dof.extend(d_rows)
        all_history.extend(h_rows)

    df_summary = pd.DataFrame(all_summary)
    if len(df_summary) > 0:
        df_summary = df_summary.sort_values(
            by=["nrmse_unmonitored", "worst_reconstructed_R2", "true_p_fd"],
            ascending=[True, False, True],
        )
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    df_summary.to_csv(summary_path, index=False)

    df_per_dof = pd.DataFrame(all_per_dof)
    per_dof_path = os.path.join(OUT_DIR, "per_dof_metrics.csv")
    df_per_dof.to_csv(per_dof_path, index=False)

    df_history = pd.DataFrame(all_history)
    history_path = os.path.join(OUT_DIR, "stage_history.csv")
    df_history.to_csv(history_path, index=False)

    elapsed_all = pytime.time() - t_all0

    print("\n" + "=" * 80)
    print("全部物理约束虚拟传感枚举完成")
    print("=" * 80)
    print(f"总耗时: {elapsed_all:.1f}s")
    print(f"summary 保存到: {summary_path}")
    print(f"per-DOF 保存到: {per_dof_path}")
    print(f"history 保存到: {history_path}")

    if len(df_summary) > 0:
        show_cols = [
            "mask_name",
            "sensor_floors_1based",
            "reconstructed_floors_1based",
            "prior_nrmse_unmonitored",
            "nrmse_unmonitored",
            "rmse_unmonitored",
            "worst_reconstructed_R2",
            "mean_reconstructed_R2",
            "true_p_fd",
            "smooth_loss",
            "energy_loss",
        ]
        print("\n当前最优布设排序 Top:")
        print(df_summary[show_cols].head(10).to_string(index=False))

        best = df_summary.iloc[0]
        print("\n✅ 当前最佳布设:")
        print(f"mask_name: {best['mask_name']}")
        print(f"sensor floors: {best['sensor_floors_1based']}")
        print(f"unmonitored NRMSE: {best['nrmse_unmonitored']:.6e}")
        print(f"worst reconstructed R2: {best['worst_reconstructed_R2']:.6f}")
        print(f"true_p_fd: {best['true_p_fd']:.6e}")


if __name__ == "__main__":
    main()
