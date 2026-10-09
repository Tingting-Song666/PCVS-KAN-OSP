# -*- coding: utf-8 -*-
r"""
FINAL：20DOF ranking consistency 补实验（不再反复报错版）

运行：
    cd "<local-project-directory>"
    .\.venv\Scripts\python.exe .\rank20_REGENERATE_kan_consistency_FINAL.py

这个脚本完成：
1. 从 <local-project-directory>/ga_pc_vs_20dof_stage5_results 自动读取已有 20DOF、4传感器 PC-VS teacher/review 结果；
2. 严格过滤：只保留 20DOF + 4 个传感器候选，避免混入 10DOF/3传感器；
3. 用这些 PC-VS teacher 标签重新训练一个 20DOF KAN/RBF surrogate；
4. 用 KAN/RBF surrogate 快速评价全部 C(20,4)=4845 个候选；
5. 在已有 PC-VS teacher/review 集合上计算：
   - Spearman
   - Kendall
   - Top-K overlap
   - teacher-best recall
6. 输出：
   - surrogate_all_4845_ranked.csv
   - known_pcvs_with_regenerated_kan_predictions.csv
   - topk_overlap_known_pcvs_set.csv
   - topk_overlap_heldout_test_set.csv
   - teacher_best_recall_in_surrogate_topk.csv
   - teacher_top100_known_with_surrogate_rank.csv
   - surrogate_top100_known_with_teacher_rank.csv
   - surrogate_top100_all4845_for_pcvs_review.csv
   - pcvs_review_plan_top100_random300.csv
   - pcvs_review_needed_top100_random300.csv
   - scatter_pcvs_teacher_vs_regenerated_kan_known_set.png
   - scatter_pcvs_teacher_vs_regenerated_kan_test_set.png
   - ranking_consistency_summary.txt

重要口径：
- KAN surrogate 会对全部 4845 个组合做快速评价。
- PC-VS teacher 不会凭空伪造；它读取你已有的 PC-VS review/teacher 结果。
- 如果当前 PC-VS 没覆盖全部 4845，本脚本会输出 Top100 + random300 复核清单。
- 论文中应写：all-candidate surrogate screening + PC-VS-reviewed subset ranking consistency。
"""

from __future__ import annotations

import itertools
import json
import random
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, kendalltau
except Exception:
    spearmanr = None
    kendalltau = None


# ======================================================================================
# 配置
# ======================================================================================

ROOT = Path(__file__).resolve().parent
TEACHER_DIR = ROOT / "ga_pc_vs_20dof_stage5_results"
OUT_DIR = ROOT / "ranking_consistency_20dof_REGENERATED_FINAL"

N_DOF = 20
N_SENSOR = 4
TOTAL_CAND = 4845

SEED = 20260706
TEST_FRAC = 0.25
VAL_FRAC = 0.15

# torch RBF-KAN surrogate 参数
EPOCHS = 2200
PATIENCE = 260
LR = 2e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 96
N_BASIS = 7
BATCH_SIZE = 256

TOPK_LIST = [5, 10, 20, 50, 100]
TOP_N_TABLE = 100

# PC-VS review plan
REVIEW_TOPK = 100
REVIEW_RANDOM_N = 300


# ======================================================================================
# 通用工具
# ======================================================================================

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_any(path: Path) -> Optional[pd.DataFrame]:
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[skip] read failed: {path} | {e}")
        return None


def layout0_to_str(layout0: Sequence[int]) -> str:
    return "-".join([f"DOF{i+1}" for i in sorted(layout0)])


def layout1_to_str(layout1: Sequence[int]) -> str:
    return "-".join([f"DOF{i}" for i in sorted(layout1)])


def all_candidate_layouts() -> pd.DataFrame:
    rows = []
    for comb in itertools.combinations(range(N_DOF), N_SENSOR):
        rows.append({
            "layout": layout0_to_str(comb),
            "layout0": ",".join(map(str, comb)),
            "x_mask": "".join("1" if i in comb else "0" for i in range(N_DOF)),
        })
    df = pd.DataFrame(rows)
    assert len(df) == TOTAL_CAND
    return df


def layout_to_x(layout: str) -> np.ndarray:
    x = np.zeros(N_DOF, dtype=np.float32)
    nums = [int(v) for v in re.findall(r"DOF([0-9]+)", str(layout), flags=re.I)]
    for n in nums:
        if 1 <= n <= N_DOF:
            x[n - 1] = 1.0
    return x


def parse_layout_from_value(value, col_name: str = "") -> List[Tuple[str, str]]:
    """
    严格解析 20DOF + 4 sensor layout，统一输出 DOF1-DOF5-DOF10-DOF16。
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in ["nan", "none"]:
        return []

    lc = str(col_name).lower()

    # 20位二进制字符串
    compact = re.sub(r"[\s,\[\]\(\)_;\-|]+", "", s)
    if len(compact) == N_DOF and set(compact).issubset({"0", "1"}) and compact.count("1") == N_SENSOR:
        idx0 = [i for i, c in enumerate(compact) if c == "1"]
        return [(layout0_to_str(idx0), f"{col_name}:binary-string")]

    nums_all = [int(v) for v in re.findall(r"-?\d+", s)]

    # 20维0/1向量
    if len(nums_all) == N_DOF and set(nums_all).issubset({0, 1}) and sum(nums_all) == N_SENSOR:
        idx0 = [i for i, v in enumerate(nums_all) if v == 1]
        return [(layout0_to_str(idx0), f"{col_name}:binary-vector")]

    # DOF3_DOF8_DOF14_DOF19
    dof_nums = [int(v) for v in re.findall(r"DOF\s*([0-9]+)", s, flags=re.I)]
    dof_nums = sorted(set(dof_nums))
    if len(dof_nums) == N_SENSOR and all(1 <= v <= N_DOF for v in dof_nums):
        return [(layout1_to_str(dof_nums), f"{col_name}:dof-prefix")]

    # 普通4个数字
    nums = sorted(set(nums_all))
    if len(nums) != N_SENSOR:
        return []

    out = []

    # 明确1based
    if "1based" in lc or "1_based" in lc or "floor" in lc or "floors" in lc:
        if all(1 <= v <= N_DOF for v in nums):
            out.append((layout1_to_str(nums), f"{col_name}:numeric-1idx-explicit"))
        return out

    # 明确0based
    if "0based" in lc or "0_based" in lc or "tuple_0" in lc or "zero" in lc:
        if all(0 <= v <= N_DOF - 1 for v in nums):
            out.append((layout0_to_str(nums), f"{col_name}:numeric-0idx-explicit"))
        return out

    # 不明确：双解释
    if all(0 <= v <= N_DOF - 1 for v in nums):
        out.append((layout0_to_str(nums), f"{col_name}:numeric-0idx"))
    if all(1 <= v <= N_DOF for v in nums):
        out.append((layout1_to_str(nums), f"{col_name}:numeric-1idx"))

    # 去重
    seen = set()
    clean = []
    for layout, mode in out:
        if layout not in seen:
            clean.append((layout, mode))
            seen.add(layout)
    return clean


def parse_layouts_from_row(df: pd.DataFrame, idx: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    # A. 整列字段
    key_cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in [
            "mask_name", "layout", "sensor_floors", "sensor_tuple", "candidate",
            "sensors", "placement", "dofs", "sensor"
        ]):
            if not any(bad in lc for bad in ["nrmse", "score", "fitness", "loss", "rank", "time", "noise", "case"]):
                key_cols.append(c)

    for c in key_cols:
        for layout, mode in parse_layout_from_value(df.loc[idx, c], c):
            out.append((layout, mode))

    # B. 分列 sensor1/sensor2/sensor3/sensor4
    split_nums = []
    split_cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["sensor", "dof", "floor", "placement"]):
            if any(bad in lc for bad in ["nrmse", "score", "fitness", "loss", "rank", "time", "noise", "case"]):
                continue
            v = pd.to_numeric(df.loc[idx, c], errors="coerce")
            if pd.notna(v) and -1 <= float(v) <= 30:
                split_nums.append(int(v))
                split_cols.append(c)

    split_nums = sorted(set(split_nums))
    if len(split_nums) == N_SENSOR:
        name = "+".join(map(str, split_cols))
        if all(0 <= v <= N_DOF - 1 for v in split_nums):
            out.append((layout0_to_str(split_nums), f"{name}:split-0idx"))
        if all(1 <= v <= N_DOF for v in split_nums):
            out.append((layout1_to_str(split_nums), f"{name}:split-1idx"))

    # 去重 + 严格检查
    seen = set()
    clean = []
    for layout, mode in out:
        nums = [int(v) for v in re.findall(r"DOF([0-9]+)", layout)]
        if len(nums) == N_SENSOR and all(1 <= v <= N_DOF for v in nums):
            if layout not in seen:
                clean.append((layout, mode))
                seen.add(layout)
    return clean


def choose_numeric_col(df: pd.DataFrame, patterns: List[str], forbidden: Sequence[str] = ()) -> Optional[str]:
    for pat in patterns:
        opts = []
        for c in df.columns:
            lc = str(c).lower()
            if any(f in lc for f in forbidden):
                continue
            if re.search(pat, lc):
                ser = pd.to_numeric(df[c], errors="coerce")
                n = int(ser.notna().sum())
                if n > 0:
                    opts.append((n, c))
        if opts:
            opts.sort(reverse=True)
            return opts[0][1]
    return None


def mean_patterns() -> List[str]:
    return [
        r"^mean_unmon_nrmse$",
        r"mean.*unmon.*nrmse",
        r"mean.*nrmse",
        r"avg.*nrmse",
        r"nrmse",
    ]


def worst_patterns() -> List[str]:
    return [
        r"^worst_unmon_nrmse$",
        r"worst.*unmon.*nrmse",
        r"worst.*nrmse",
        r"max.*nrmse",
    ]


def fitness_patterns() -> List[str]:
    return [
        r"^fitness$",
        r"fitness",
        r"score",
        r"objective",
    ]


def teacher_file_priority(path: Path) -> int:
    s = str(path).lower()
    name = path.name.lower()

    if "combined_clean_mask_ranking.csv" in name:
        return 1
    if "local_clean_overall_mask_ranking.csv" in name:
        return 2
    if "ga_topk_clean_overall_mask_ranking.csv" in name:
        return 3
    if "robust_noise_overall_mask_ranking.csv" in name:
        return 4
    if "baseline_candidate_ranking.csv" in name:
        return 5
    if "ga_candidate_cache.csv" in name:
        return 6
    if "local_neighbor_candidates.csv" in name:
        return 7
    if "mask_ranking" in name:
        return 10
    if "per_case" in name or "metrics" in name:
        return 20
    return 50


def teacher_candidate_files() -> List[Path]:
    if not TEACHER_DIR.exists():
        raise FileNotFoundError(f"找不到 20DOF PC-VS 结果目录：{TEACHER_DIR}")

    files = []
    for p in TEACHER_DIR.rglob("*.csv"):
        name = p.name.lower()
        if any(k in name for k in [
            "mask_ranking",
            "candidate_cache",
            "baseline_candidate",
            "local_neighbor",
        ]):
            files.append(p)

    files = sorted(files, key=lambda p: (teacher_file_priority(p), str(p)))
    return files


def load_teacher_scores() -> pd.DataFrame:
    rows = []
    files = teacher_candidate_files()
    print(f"[teacher] candidate CSV files found: {len(files)}")

    for p in files:
        df = read_csv_any(p)
        if df is None or df.empty:
            continue

        mean_col = choose_numeric_col(df, mean_patterns())
        worst_col = choose_numeric_col(df, worst_patterns())
        fitness_col = choose_numeric_col(df, fitness_patterns(), forbidden=["nrmse", "loss"])

        if mean_col is None and fitness_col is None:
            continue

        for idx in df.index:
            layouts = parse_layouts_from_row(df, idx)
            if not layouts:
                continue

            mean_val = pd.to_numeric(df.loc[idx, mean_col], errors="coerce") if mean_col is not None else np.nan
            worst_val = pd.to_numeric(df.loc[idx, worst_col], errors="coerce") if worst_col is not None else np.nan

            if fitness_col is not None:
                fit_val = pd.to_numeric(df.loc[idx, fitness_col], errors="coerce")
            elif pd.notna(mean_val) and float(mean_val) > 0:
                fit_val = 1.0 / (float(mean_val) + 1e-12)
            else:
                fit_val = np.nan

            if not np.isfinite(float(fit_val)):
                continue

            if pd.isna(mean_val) and float(fit_val) > 0:
                mean_val = 1.0 / float(fit_val)

            if pd.isna(mean_val) or float(mean_val) <= 0:
                continue

            for layout, mode in layouts:
                rows.append({
                    "layout": layout,
                    "teacher_mean_nrmse": float(mean_val),
                    "teacher_worst_nrmse": float(worst_val) if pd.notna(worst_val) else np.nan,
                    "teacher_fitness": float(1.0 / (float(mean_val) + 1e-12)),
                    "source_file": str(p),
                    "source_name": p.name,
                    "parse_mode": mode,
                    "priority": teacher_file_priority(p),
                    "row_index": int(idx),
                    "mean_col": mean_col or "",
                    "worst_col": worst_col or "",
                    "fitness_col": fitness_col or "",
                })

    if not rows:
        raise RuntimeError("没有从 20DOF PC-VS 结果中解析到任何 4-sensor teacher score。")

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT_DIR / "pcvs_teacher_raw_parsed_candidates.csv", index=False, encoding="utf-8-sig")

    # 同一layout多来源：优先级高者；同优先级取teacher_mean_nrmse更低者
    raw = raw.sort_values(["layout", "priority", "teacher_mean_nrmse"], ascending=[True, True, True])
    best = raw.drop_duplicates(subset=["layout"], keep="first").copy()
    best = best.sort_values("teacher_mean_nrmse", ascending=True).reset_index(drop=True)
    best["teacher_rank_known"] = np.arange(1, len(best) + 1)
    best["x_mask"] = best["layout"].apply(lambda s: "".join(str(int(v)) for v in layout_to_x(s).astype(int)))

    return best


# ======================================================================================
# KAN/RBF surrogate：不再保存再加载，避免 torch.load / weights_only 问题
# ======================================================================================

def train_and_predict_surrogate(X_known: np.ndarray, y_log: np.ndarray, X_all: np.ndarray, out_dir: Path) -> Dict:
    """
    返回：
        pred_log_known
        pred_log_all
        split arrays
        backend
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return train_predict_numpy_ridge(X_known, y_log, X_all, out_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    n = len(X_known)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    n_test = max(1, int(round(TEST_FRAC * n)))
    n_val = max(1, int(round(VAL_FRAC * (n - n_test))))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    if len(train_idx) < 10:
        train_idx = idx[n_test:]
        val_idx = idx[:n_test]
        test_idx = idx[:n_test]

    class RBFKAN(nn.Module):
        def __init__(self, n_dof: int, n_basis: int, hidden: int):
            super().__init__()
            centers = torch.linspace(0.0, 1.0, n_basis)
            self.register_buffer("centers", centers)
            self.gamma = nn.Parameter(torch.tensor(12.0))
            self.net = nn.Sequential(
                nn.Linear(n_dof * n_basis + n_dof, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            b = torch.exp(-torch.clamp(self.gamma, 1.0, 50.0) * (x.unsqueeze(-1) - self.centers) ** 2)
            z = torch.cat([x, b.reshape(x.shape[0], -1)], dim=1)
            return self.net(z)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[surrogate] backend=torch_rbf_kan device={device} known={n}")

    model = RBFKAN(N_DOF, N_BASIS, HIDDEN).to(device)
    Xk = torch.tensor(X_known, dtype=torch.float32, device=device)
    yk = torch.tensor(y_log[:, None], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

    best_val = float("inf")
    best_state = None
    bad = 0
    history = []

    # 全 batch，数据量本来很小
    for ep in range(1, EPOCHS + 1):
        model.train()
        pred = model(Xk[train_idx_t])
        loss = loss_fn(pred, yk[train_idx_t])

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xk[val_idx_t]), yk[val_idx_t]).item()

        history.append({"epoch": ep, "train_loss": float(loss.item()), "val_loss": float(val_loss)})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep % 200 == 0:
            print(f"[train] epoch={ep:04d} train={loss.item():.6g} val={val_loss:.6g}")

        if bad >= PATIENCE:
            print(f"[train] early stop at epoch={ep}, best_val={best_val:.6g}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(out_dir / "kan_training_log.csv", index=False, encoding="utf-8-sig")

    # 只保存 state_dict，不在本脚本内再 torch.load，避免 PyTorch 2.6 weights_only 报错
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "n_dof": N_DOF,
            "n_sensor": N_SENSOR,
            "n_basis": N_BASIS,
            "hidden": HIDDEN,
            "target": "log(mean_nrmse)",
        },
    }, out_dir / "regenerated_kan_rbf_surrogate_state_dict.pt")

    def predict_np(X_np: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_np), 2048):
                xb = torch.tensor(X_np[i:i+2048], dtype=torch.float32, device=device)
                preds.append(model(xb).detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds)

    pred_known = predict_np(X_known)
    pred_all = predict_np(X_all)

    return {
        "backend": "torch_rbf_kan",
        "pred_log_known": pred_known,
        "pred_log_all": pred_all,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def train_predict_numpy_ridge(X_known: np.ndarray, y_log: np.ndarray, X_all: np.ndarray, out_dir: Path) -> Dict:
    print("[surrogate] backend=numpy_ridge_rbf")
    rng = np.random.default_rng(SEED)
    n = len(X_known)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_test = max(1, int(round(TEST_FRAC * n)))
    n_val = max(1, int(round(VAL_FRAC * (n - n_test))))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    centers = np.linspace(0.0, 1.0, N_BASIS)
    gamma = 12.0

    def featurize(X):
        Phi = np.exp(-gamma * (X[:, :, None] - centers[None, None, :]) ** 2).reshape(len(X), -1)
        return np.concatenate([np.ones((len(X), 1)), X, Phi], axis=1)

    Phi_k = featurize(X_known)
    Phi_all = featurize(X_all)

    lam = 1e-3
    A = Phi_k[train_idx].T @ Phi_k[train_idx] + lam * np.eye(Phi_k.shape[1])
    b = Phi_k[train_idx].T @ y_log[train_idx]
    w = np.linalg.solve(A, b)

    np.savez(out_dir / "regenerated_numpy_ridge_surrogate.npz", w=w, centers=centers, gamma=gamma)

    return {
        "backend": "numpy_ridge_rbf",
        "pred_log_known": Phi_k @ w,
        "pred_log_all": Phi_all @ w,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


# ======================================================================================
# 评价与输出
# ======================================================================================

def rank_desc(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(-arr)
    ranks = np.empty(len(arr), dtype=int)
    ranks[order] = np.arange(1, len(arr) + 1)
    return ranks


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if spearmanr is not None:
        return float(spearmanr(x[mask], y[mask]).correlation)
    return float(pd.Series(x[mask]).rank().corr(pd.Series(y[mask]).rank()))


def safe_kendall(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    if kendalltau is not None:
        return float(kendalltau(x[mask], y[mask]).correlation)
    return float("nan")


def topk_overlap(df: pd.DataFrame, k_list: Sequence[int], teacher_rank_col: str, surrogate_rank_col: str) -> pd.DataFrame:
    rows = []
    for k in k_list:
        kk = min(k, len(df))
        if kk <= 0:
            continue
        tset = set(df.sort_values(teacher_rank_col).head(kk)["layout"])
        sset = set(df.sort_values(surrogate_rank_col).head(kk)["layout"])
        inter = tset & sset
        rows.append({
            "K": kk,
            "teacher_topK_count": len(tset),
            "surrogate_topK_count": len(sset),
            "overlap_count": len(inter),
            "overlap_rate": len(inter) / kk,
        })
    return pd.DataFrame(rows)


def teacher_best_recall(df: pd.DataFrame, k_list: Sequence[int], n_teacher: int = 10) -> pd.DataFrame:
    teacher_top = df.sort_values("teacher_rank_known").head(min(n_teacher, len(df)))
    sur_order = list(df.sort_values("surrogate_rank_known")["layout"])

    rows = []
    for _, r in teacher_top.iterrows():
        layout = r["layout"]
        sr = sur_order.index(layout) + 1 if layout in sur_order else np.nan
        row = {
            "teacher_layout": layout,
            "teacher_rank_known": int(r["teacher_rank_known"]),
            "surrogate_rank_known": int(sr) if np.isfinite(sr) else np.nan,
            "teacher_mean_nrmse": float(r["teacher_mean_nrmse"]),
            "surrogate_pred_mean_nrmse": float(r["surrogate_pred_mean_nrmse"]),
        }
        for k in k_list:
            row[f"in_surrogate_top{k}"] = bool(np.isfinite(sr) and sr <= k)
        rows.append(row)
    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, out_path: Path, title: str):
    x = df["teacher_fitness"].values
    y = df["surrogate_fitness"].values

    plt.figure(figsize=(7.4, 6.2), dpi=180)
    plt.scatter(x, y, s=20, alpha=0.72)
    if len(x) >= 2:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.xlabel("PC-VS teacher/review fitness")
    plt.ylabel("Regenerated KAN surrogate fitness")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def build_review_plan(all_pred: pd.DataFrame, known_layouts: set, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成 Top100 + random300 的 PC-VS review 计划表。
    """
    rng = np.random.default_rng(SEED)

    top_df = all_pred.sort_values("surrogate_rank_all4845").head(REVIEW_TOPK).copy()
    top_df["review_source"] = f"surrogate_top{REVIEW_TOPK}"

    remaining = all_pred[~all_pred["layout"].isin(set(top_df["layout"]))].copy()
    random_n = min(REVIEW_RANDOM_N, len(remaining))
    rand_idx = rng.choice(remaining.index.to_numpy(), size=random_n, replace=False)
    rand_df = remaining.loc[rand_idx].copy()
    rand_df["review_source"] = f"random{random_n}"

    plan = pd.concat([top_df, rand_df], ignore_index=True)
    plan["has_pcvs_teacher"] = plan["layout"].isin(known_layouts)
    plan = plan.sort_values(["review_source", "surrogate_rank_all4845"]).reset_index(drop=True)

    needed = plan[~plan["has_pcvs_teacher"]].copy()

    plan.to_csv(out_dir / "pcvs_review_plan_top100_random300.csv", index=False, encoding="utf-8-sig")
    needed.to_csv(out_dir / "pcvs_review_needed_top100_random300.csv", index=False, encoding="utf-8-sig")

    return plan, needed


def write_summary(
    out_dir: Path,
    teacher_df: pd.DataFrame,
    known_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_pred: pd.DataFrame,
    ov_known: pd.DataFrame,
    ov_test: pd.DataFrame,
    recall_known: pd.DataFrame,
    train_info: Dict,
    review_plan: pd.DataFrame,
    review_needed: pd.DataFrame,
):
    sp_known = safe_spearman(known_df["teacher_fitness"], known_df["surrogate_fitness"])
    kd_known = safe_kendall(known_df["teacher_fitness"], known_df["surrogate_fitness"])
    sp_test = safe_spearman(test_df["teacher_fitness"], test_df["surrogate_fitness"]) if len(test_df) >= 3 else np.nan
    kd_test = safe_kendall(test_df["teacher_fitness"], test_df["surrogate_fitness"]) if len(test_df) >= 3 else np.nan

    sp_err_known = safe_spearman(-known_df["teacher_mean_nrmse"], -known_df["surrogate_pred_mean_nrmse"])
    kd_err_known = safe_kendall(-known_df["teacher_mean_nrmse"], -known_df["surrogate_pred_mean_nrmse"])
    sp_err_test = safe_spearman(-test_df["teacher_mean_nrmse"], -test_df["surrogate_pred_mean_nrmse"]) if len(test_df) >= 3 else np.nan
    kd_err_test = safe_kendall(-test_df["teacher_mean_nrmse"], -test_df["surrogate_pred_mean_nrmse"]) if len(test_df) >= 3 else np.nan

    teacher_top1 = known_df.sort_values("teacher_rank_known").head(1)
    sur_top1_known = known_df.sort_values("surrogate_rank_known").head(1)
    sur_top1_all = all_pred.sort_values("surrogate_rank_all4845").head(1)

    top100 = all_pred.sort_values("surrogate_rank_all4845").head(100)
    top100_known = int(top100["has_pcvs_teacher"].sum())

    lines = []
    lines.append("=" * 110)
    lines.append("20DOF regenerated KAN surrogate all-candidate ranking consistency")
    lines.append("=" * 110)
    lines.append(f"Project root                         : {ROOT}")
    lines.append(f"Output dir                           : {out_dir}")
    lines.append(f"Teacher source dir                   : {TEACHER_DIR}")
    lines.append(f"Surrogate backend                    : {train_info['backend']}")
    lines.append("")
    lines.append("[Coverage]")
    lines.append(f"All possible 20DOF 4-sensor layouts  : {TOTAL_CAND} / {TOTAL_CAND}")
    lines.append(f"PC-VS teacher/review known layouts   : {len(teacher_df)} / {TOTAL_CAND}")
    lines.append(f"KAN surrogate scored layouts         : {len(all_pred)} / {TOTAL_CAND}")
    lines.append(f"Train layouts                        : {len(train_info['train_idx'])}")
    lines.append(f"Validation layouts                   : {len(train_info['val_idx'])}")
    lines.append(f"Test layouts                         : {len(train_info['test_idx'])}")
    lines.append(f"Surrogate Top100 with PC-VS teacher  : {top100_known} / 100")
    lines.append(f"PC-VS review plan size               : {len(review_plan)}")
    lines.append(f"PC-VS review still needed            : {len(review_needed)}")
    lines.append("")
    lines.append("Important note:")
    lines.append("  本脚本已经对全部 4845 个候选完成 KAN surrogate 快速评价。")
    lines.append("  PC-VS teacher/review 读取自本地已有20DOF复核结果；未覆盖的候选已写入 pcvs_review_needed_top100_random300.csv。")
    lines.append("  因此论文应写作：all-candidate surrogate screening + PC-VS-reviewed subset ranking consistency。")
    lines.append("  不能写成全部 4845 个候选均重新完成 PC-VS teacher 评价。")
    lines.append("")
    lines.append("[Rank correlation on all known PC-VS-reviewed layouts]")
    lines.append(f"Spearman fitness                     : {sp_known:.6f}" if np.isfinite(sp_known) else "Spearman fitness                     : NaN")
    lines.append(f"Kendall fitness                      : {kd_known:.6f}" if np.isfinite(kd_known) else "Kendall fitness                      : NaN")
    lines.append(f"Spearman -mean_NRMSE                 : {sp_err_known:.6f}" if np.isfinite(sp_err_known) else "Spearman -mean_NRMSE                 : NaN")
    lines.append(f"Kendall -mean_NRMSE                  : {kd_err_known:.6f}" if np.isfinite(kd_err_known) else "Kendall -mean_NRMSE                  : NaN")
    lines.append("")
    lines.append("[Rank correlation on held-out test PC-VS layouts]")
    lines.append(f"Spearman fitness test                : {sp_test:.6f}" if np.isfinite(sp_test) else "Spearman fitness test                : NaN")
    lines.append(f"Kendall fitness test                 : {kd_test:.6f}" if np.isfinite(kd_test) else "Kendall fitness test                 : NaN")
    lines.append(f"Spearman -mean_NRMSE test            : {sp_err_test:.6f}" if np.isfinite(sp_err_test) else "Spearman -mean_NRMSE test            : NaN")
    lines.append(f"Kendall -mean_NRMSE test             : {kd_err_test:.6f}" if np.isfinite(kd_err_test) else "Kendall -mean_NRMSE test             : NaN")
    lines.append("")
    lines.append("[Top-1]")
    if len(teacher_top1):
        r = teacher_top1.iloc[0]
        lines.append(f"Teacher Top1 known                   : {r['layout']} | teacher_mean_nrmse={r['teacher_mean_nrmse']:.6g} | surrogate_rank_known={int(r['surrogate_rank_known'])}")
    if len(sur_top1_known):
        r = sur_top1_known.iloc[0]
        lines.append(f"Surrogate Top1 within known          : {r['layout']} | surrogate_pred_mean_nrmse={r['surrogate_pred_mean_nrmse']:.6g} | teacher_rank_known={int(r['teacher_rank_known'])}")
    if len(sur_top1_all):
        r = sur_top1_all.iloc[0]
        lines.append(f"Surrogate Top1 all4845               : {r['layout']} | surrogate_pred_mean_nrmse={r['surrogate_pred_mean_nrmse']:.6g} | has_pcvs_teacher={bool(r['has_pcvs_teacher'])}")
    lines.append("")
    lines.append("[Top-K overlap on all known PC-VS-reviewed layouts]")
    lines.append(ov_known.to_string(index=False) if len(ov_known) else "No overlap.")
    lines.append("")
    lines.append("[Top-K overlap on held-out test layouts]")
    lines.append(ov_test.to_string(index=False) if len(ov_test) else "No overlap.")
    lines.append("")
    lines.append("[Teacher-best recall in surrogate ranking, known set]")
    lines.append(recall_known.to_string(index=False) if len(recall_known) else "No recall.")
    lines.append("")
    lines.append("[Paper-ready wording]")
    lines.append("为进一步检验代理模型用于候选布设快速筛选的可靠性，本文基于已有20DOF PC-VS teacher/review 结果重新训练 KAN/RBF surrogate，并对全部 C(20,4)=4845 个候选布设进行快速评价。随后，在具有 PC-VS 复核标签的候选集合上计算 Spearman、Kendall、Top-K overlap 和 teacher-best recall。该实验用于验证 KAN surrogate 的候选筛选能力；最终布设仍由 PC-VS Top-K review 复核确定，以避免代理模型局部排序误差直接影响最终决策。")

    txt = "\n".join(lines)
    (out_dir / "ranking_consistency_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


# ======================================================================================
# 主程序
# ======================================================================================

def main():
    t0 = time.time()
    ensure_dir(OUT_DIR)

    print("=" * 110)
    print("[20DOF regenerated KAN ranking consistency FINAL]")
    print("=" * 110)
    print(f"[root] {ROOT}")
    print(f"[out ] {OUT_DIR}")
    print(f"[teacher_dir] {TEACHER_DIR}")

    all_df = all_candidate_layouts()
    all_df.to_csv(OUT_DIR / "all_4845_layouts.csv", index=False, encoding="utf-8-sig")

    teacher_df = load_teacher_scores()
    teacher_df.to_csv(OUT_DIR / "pcvs_teacher_known_20dof_layouts.csv", index=False, encoding="utf-8-sig")
    print(f"[teacher] unique strict 20DOF-4sensor layouts = {len(teacher_df)}")

    if len(teacher_df) < 30:
        raise RuntimeError(
            f"可用 20DOF PC-VS teacher 布设只有 {len(teacher_df)} 个，太少，无法训练 ranking surrogate。"
            f"请先用 PC-VS 评价更多20DOF候选。"
        )

    X_known = np.stack([layout_to_x(s) for s in teacher_df["layout"].tolist()]).astype(np.float32)
    y_mean = teacher_df["teacher_mean_nrmse"].astype(float).to_numpy()
    y_log = np.log(np.clip(y_mean, 1e-8, None)).astype(np.float32)

    X_all = np.stack([np.array(list(mask), dtype=np.float32) for mask in all_df["x_mask"].tolist()]).astype(np.float32)

    train_info = train_and_predict_surrogate(X_known, y_log, X_all, OUT_DIR)

    # known predictions
    pred_mean_known = np.exp(train_info["pred_log_known"])
    teacher_df["surrogate_pred_mean_nrmse"] = pred_mean_known
    teacher_df["surrogate_fitness"] = 1.0 / (teacher_df["surrogate_pred_mean_nrmse"].astype(float) + 1e-12)
    teacher_df["teacher_rank_known"] = rank_desc(teacher_df["teacher_fitness"].to_numpy())
    teacher_df["surrogate_rank_known"] = rank_desc(teacher_df["surrogate_fitness"].to_numpy())

    split = np.array(["train"] * len(teacher_df), dtype=object)
    split[train_info["val_idx"]] = "val"
    split[train_info["test_idx"]] = "test"
    teacher_df["split"] = split

    teacher_df = teacher_df.sort_values("teacher_rank_known").reset_index(drop=True)
    teacher_df.to_csv(OUT_DIR / "known_pcvs_with_regenerated_kan_predictions.csv", index=False, encoding="utf-8-sig")

    # all 4845 predictions
    pred_mean_all = np.exp(train_info["pred_log_all"])
    all_pred = all_df.copy()
    all_pred["surrogate_pred_mean_nrmse"] = pred_mean_all
    all_pred["surrogate_fitness"] = 1.0 / (pred_mean_all + 1e-12)
    all_pred["surrogate_rank_all4845"] = rank_desc(all_pred["surrogate_fitness"].to_numpy())

    known_map = teacher_df.set_index("layout")
    known_layouts = set(known_map.index)
    all_pred["has_pcvs_teacher"] = all_pred["layout"].isin(known_layouts)
    all_pred["teacher_mean_nrmse"] = all_pred["layout"].map(known_map["teacher_mean_nrmse"])
    all_pred["teacher_rank_known"] = all_pred["layout"].map(known_map["teacher_rank_known"])
    all_pred = all_pred.sort_values("surrogate_rank_all4845").reset_index(drop=True)
    all_pred.to_csv(OUT_DIR / "surrogate_all_4845_ranked.csv", index=False, encoding="utf-8-sig")

    # Review plan
    review_plan, review_needed = build_review_plan(all_pred, known_layouts, OUT_DIR)

    # metrics
    known_df = teacher_df.copy()
    test_df = teacher_df[teacher_df["split"] == "test"].copy()

    ov_known = topk_overlap(known_df, TOPK_LIST, "teacher_rank_known", "surrogate_rank_known")
    ov_test = topk_overlap(test_df, TOPK_LIST, "teacher_rank_known", "surrogate_rank_known") if len(test_df) >= 3 else pd.DataFrame()
    recall_known = teacher_best_recall(known_df, TOPK_LIST, n_teacher=min(10, len(known_df)))

    ov_known.to_csv(OUT_DIR / "topk_overlap_known_pcvs_set.csv", index=False, encoding="utf-8-sig")
    ov_test.to_csv(OUT_DIR / "topk_overlap_heldout_test_set.csv", index=False, encoding="utf-8-sig")
    recall_known.to_csv(OUT_DIR / "teacher_best_recall_in_surrogate_topk.csv", index=False, encoding="utf-8-sig")

    known_df.sort_values("teacher_rank_known").head(TOP_N_TABLE).to_csv(
        OUT_DIR / f"teacher_top{TOP_N_TABLE}_known_with_surrogate_rank.csv",
        index=False,
        encoding="utf-8-sig"
    )
    known_df.sort_values("surrogate_rank_known").head(TOP_N_TABLE).to_csv(
        OUT_DIR / f"surrogate_top{TOP_N_TABLE}_known_with_teacher_rank.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_scatter(
        known_df,
        OUT_DIR / "scatter_pcvs_teacher_vs_regenerated_kan_known_set.png",
        "20DOF known PC-VS layouts: teacher vs regenerated KAN surrogate"
    )
    if len(test_df) >= 3:
        save_scatter(
            test_df,
            OUT_DIR / "scatter_pcvs_teacher_vs_regenerated_kan_test_set.png",
            "20DOF held-out PC-VS layouts: teacher vs regenerated KAN surrogate"
        )

    write_summary(OUT_DIR, teacher_df, known_df, test_df, all_pred, ov_known, ov_test, recall_known, train_info, review_plan, review_needed)

    print(f"\n[done] elapsed = {time.time() - t0:.1f}s")
    print(f"[done] outputs = {OUT_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
