import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

# ============================================================
# ONLY ONE CURVE:
# Robust g2(τ) over sampled 2×2 patches, using:
#   - Fixed total window Twin_total (e.g., 1s)
#   - Split into subwindows (e.g., 100ms) to reduce non-stationarity
#   - For each subwindow:
#       per patch -> H_p(τ) (pair-count histogram)
#                -> B_p(τ)=λ^2*(T-τ)*Δτ baseline
#       aggregate -> trim_sum(H)/trim_sum(B) BUT we accumulate trimmed H,B
#   - Across subwindows: H_acc += H_trim ; B_acc += B_trim
#   - Final: g2 = H_acc / B_acc
# Optional A: deadtime mask (True/False)
# Optional E: tail normalize (tail=1) with guards
# ============================================================

CONFIG = {
    # -------- input files --------
    "test_files": [
        # 改成你的路径
        "/data/zm/2026.1.12_testdata/1.15_150_580W/0.2_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/0.5_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/0.8_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/1.0_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/1.2_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/1.5_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/1.8_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/2.0_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/2.2_clip.csv",
        "/data/zm/2026.1.12_testdata/1.15_150_580W/2.5_clip.csv",
    ],

    # -------- ROI (pixel) --------
    "row_min": 400,
    "row_max": 499,
    "col_min": 0,
    "col_max": 1280,

    # -------- patch sampling (2×2) --------
    "patch_size": 2,          # 固定 2×2
    "num_patches": 400,
    "seed": 0,

    # -------- correlation bins --------
    "max_lag_us": 3000,
    "bin_size_us": 100,        # 50 或 100 更稳；10 太稀疏

    # -------- time windows --------
    "total_window_us": 1_000_000,    # Twin_total: 1s
    "subwindow_us": 100_000,         # 子窗: 100ms（建议 50~200ms）
    "global_window": True,           # True: 所有patch使用同一全局[t0, t0+total_window)

    # -------- patch validity (per SUBWINDOW) --------
    # 注意：这是“每个子窗”的最小事件数，不能沿用你之前1s窗口的200/300
    # 例如 1s切成10个子窗，若你希望“全程约200事件/patch”，每子窗约20事件 -> 太稀疏
    # 建议：先从 50~120 试起（视你的事件密度）
    "min_events_per_patch_subwin": 40,

    # -------- robust aggregation (trimmed-sum) --------
    "trim_frac": 0.10,        # 去掉上下各10%的patch（每个τ bin上分别trim）

    # -------- optimization A: deadtime mask --------
    "use_deadtime_mask": False,
    "deadtime_us": 50,

    # -------- optimization E: tail normalize --------
    "use_tail_normalize": True,
    "tail_from_us": 2500,
    "tail_min_valid_bins": 5,
    "tail_min_mean": 1e-3,

    # -------- optional: active patch sampling --------
    # used_patches 太低时建议开启：先筛活跃patch再抽样
    "use_active_patch_sampling": False,
    "active_count_threshold_subwin": 30,  # “子窗内”事件数>=该阈值 才视为活跃

    # -------- output --------
    "out_dir": "/data/zm/2026.1.12_testdata/1.15_150_580W/tin_g2_subwin",
    "save_plot": True,
}


@njit
def hist_dt_within_window(sorted_t_in, max_lag_us, bin_size_us):
    """
    Forward pairs within max_lag.
    """
    n = len(sorted_t_in)
    num_bins = max_lag_us // bin_size_us
    hist = np.zeros(num_bins, dtype=np.int64)

    for i in range(n):
        t0 = sorted_t_in[i]
        for j in range(i + 1, n):
            dt = sorted_t_in[j] - t0
            if dt >= max_lag_us:
                break
            b = dt // bin_size_us
            if b < num_bins:
                hist[b] += 1
    return hist


def parse_flow_from_name(name: str) -> float:
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", name)
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return float(m.group(1))
    return np.nan


def build_tau_centers(max_lag_us: int, bin_size_us: int) -> np.ndarray:
    K = max_lag_us // bin_size_us
    return (np.arange(K, dtype=np.float32) + 0.5) * float(bin_size_us)


def baseline_B(N: int, T_us: int, bin_size_us: int, tau: np.ndarray) -> np.ndarray:
    """
    B(τ) = λ^2 * (T-τ) * Δτ , λ=N/T
    """
    T = float(T_us)
    lam = float(N) / T
    B = (lam * lam) * (T - tau) * float(bin_size_us)
    B = np.maximum(B, 0.0).astype(np.float32)
    return B


def trimmed_sums_per_bin(H_mat: np.ndarray, B_mat: np.ndarray, trim_frac: float):
    """
    Return (Hsum[k], Bsum[k]) after trimming per τ bin.
    Trimming is based on H (same indices applied to B).
    """
    P, K = H_mat.shape
    Hsum = np.zeros(K, dtype=np.float32)
    Bsum = np.zeros(K, dtype=np.float32)
    if P == 0:
        return Hsum, Bsum

    for k in range(K):
        Hk = H_mat[:, k]
        Bk = B_mat[:, k]
        valid = Bk > 0
        if np.sum(valid) < 5:
            continue

        Hk_v = Hk[valid]
        Bk_v = Bk[valid]

        idx = np.argsort(Hk_v)
        Hs = Hk_v[idx]
        Bs = Bk_v[idx]

        Pv = len(Hs)
        lo = int(np.floor(Pv * trim_frac))
        hi = int(np.ceil(Pv * (1.0 - trim_frac)))

        # guard: 留足样本
        if hi - lo < max(5, int(0.2 * Pv)):
            lo, hi = 0, Pv

        Hsum[k] = float(np.sum(Hs[lo:hi]))
        Bsum[k] = float(np.sum(Bs[lo:hi]))

    return Hsum, Bsum


def apply_deadtime_mask(g2: np.ndarray, tau: np.ndarray, deadtime_us: int) -> np.ndarray:
    out = g2.copy()
    out[tau < float(deadtime_us)] = np.nan
    return out


def tail_normalize(g2: np.ndarray, tau: np.ndarray) -> np.ndarray:
    if not CONFIG["use_tail_normalize"]:
        return g2
    tail_from = float(CONFIG["tail_from_us"])
    mask = np.isfinite(g2) & (tau >= tail_from)
    if np.sum(mask) < int(CONFIG["tail_min_valid_bins"]):
        return g2
    tail_mean = float(np.nanmean(g2[mask]))
    if tail_mean < float(CONFIG["tail_min_mean"]):
        return g2
    return (g2 / tail_mean).astype(np.float32)


def load_roi_events(file_path: str):
    """
    Ultra-robust event loader.

    Supports:
      - separators: comma, whitespace, tab, semicolon, '$'
      - formats:
          (row, col, t)
          (row$col, t)
          (row$col$t)
      - headers / garbage lines: dropped automatically

    Returns:
      rows(int16), cols(int16), t_in(int64)  or (None,None,None) if no valid rows.
    """
    # ---------- read a small sample to guess format ----------
    with open(file_path, "r", errors="ignore") as f:
        sample_lines = [f.readline() for _ in range(50)]
    sample = "".join([ln for ln in sample_lines if ln is not None])

    # quick heuristics
    has_dollar = "$" in sample
    has_comma = "," in sample
    has_semi  = ";" in sample
    # whitespace always possible

    # candidate parsing attempts: each returns a DataFrame with columns: row, col, t_in (possibly with NaNs)
    dfs = []

    # ---------- Attempt 1: common delims (comma/whitespace/semicolon/tab) into 3 columns ----------
    # Use python engine for flexible separators.
    # We do NOT force usecols because columns might be shifted by extra fields.
    try:
        # choose a regex separator that covers comma/semicolon/whitespace
        sep = r"[,\s;]+"
        df = pd.read_csv(
            file_path,
            header=None,
            sep=sep,
            engine="python",
            comment="#",
            skip_blank_lines=True
        )
        if df.shape[1] >= 3:
            tmp = df.iloc[:, :3].copy()
            tmp.columns = ["row", "col", "t_in"]
            dfs.append(tmp)
    except Exception:
        pass

    # ---------- Attempt 2: '$' separated into 3 columns: row$col$t ----------
    if has_dollar:
        try:
            df = pd.read_csv(
                file_path,
                header=None,
                sep=r"\$",
                engine="python",
                comment="#",
                skip_blank_lines=True
            )
            if df.shape[1] >= 3:
                tmp = df.iloc[:, :3].copy()
                tmp.columns = ["row", "col", "t_in"]
                dfs.append(tmp)
        except Exception:
            pass

    # ---------- Attempt 3: (row$col) + t, where first token contains '$' ----------
    # Read 2 columns with flexible separators, then split first column by '$'.
    try:
        sep = r"[,\s;]+"
        df = pd.read_csv(
            file_path,
            header=None,
            sep=sep,
            engine="python",
            comment="#",
            skip_blank_lines=True
        )
        if df.shape[1] >= 2:
            rc = df.iloc[:, 0].astype(str)
            if rc.str.contains(r"\$").any():
                parts = rc.str.split("$", n=1, expand=True)
                if parts.shape[1] >= 2:
                    tmp = pd.DataFrame({
                        "row": parts.iloc[:, 0],
                        "col": parts.iloc[:, 1],
                        "t_in": df.iloc[:, 1]
                    })
                    dfs.append(tmp)
    except Exception:
        pass

    # ---------- Select the best candidate by valid-row ratio ----------
    best = None
    best_valid = 0

    for cand in dfs:
        # numeric coercion
        r = pd.to_numeric(cand["row"], errors="coerce")
        c = pd.to_numeric(cand["col"], errors="coerce")
        t = pd.to_numeric(cand["t_in"], errors="coerce")
        ok = r.notna() & c.notna() & t.notna()
        valid = int(ok.sum())
        if valid > best_valid:
            best_valid = valid
            best = (r[ok], c[ok], t[ok])

    if best is None or best_valid == 0:
        # 不要 raise，返回空，让上层跳过该文件
        return None, None, None

    r, c, t = best

    # round to integer pixel/time
    rows = np.rint(r.to_numpy(dtype=np.float64)).astype(np.int32)
    cols = np.rint(c.to_numpy(dtype=np.float64)).astype(np.int32)
    tin  = np.rint(t.to_numpy(dtype=np.float64)).astype(np.int64)

    # ROI filter
    m = (
        (rows >= CONFIG["row_min"]) & (rows <= CONFIG["row_max"]) &
        (cols >= CONFIG["col_min"]) & (cols <= CONFIG["col_max"])
    )
    rows, cols, tin = rows[m], cols[m], tin[m]
    if len(tin) == 0:
        return None, None, None

    return rows.astype(np.int16), cols.astype(np.int16), tin.astype(np.int64)




def compute_curve_one_file(file_path: str, patch_ids_init: np.ndarray,
                           grid_base_r: int, grid_base_c: int, grid_w: int, tau: np.ndarray):
    max_lag = int(CONFIG["max_lag_us"])
    bin_size = int(CONFIG["bin_size_us"])
    K = max_lag // bin_size

    total_T = int(CONFIG["total_window_us"])
    sub_T = int(CONFIG["subwindow_us"])
    assert sub_T > max_lag + bin_size, "subwindow_us 必须显著大于 max_lag_us"

    min_ev_sub = int(CONFIG["min_events_per_patch_subwin"])
    trim_frac = float(CONFIG["trim_frac"])

    rows, cols, t_in = load_roi_events(file_path)
    if rows is None:
        return np.full(K, np.nan, dtype=np.float32), {"used_patches_total": 0, "used_subwins": 0}

    # global start
    if bool(CONFIG["global_window"]):
        t0_global = int(np.min(t_in))
    else:
        t0_global = int(np.min(t_in))
    t1_global = t0_global + total_T

    # keep events inside total window
    mask_total = (t_in >= t0_global) & (t_in < t1_global)
    rows = rows[mask_total]
    cols = cols[mask_total]
    t_in = t_in[mask_total]
    if len(t_in) == 0:
        return np.full(K, np.nan, dtype=np.float32), {"used_patches_total": 0, "used_subwins": 0}

    # map to 2×2 patch id
    cell_r = rows >> 1
    cell_c = cols >> 1
    pids = (cell_r - grid_base_r).astype(np.int64) * np.int64(grid_w) + (cell_c - grid_base_c).astype(np.int64)

    # sort by time once (global), so subwindow slicing is cheap
    order_t = np.argsort(t_in)
    t_in = t_in[order_t]
    pids = pids[order_t]
    del order_t

    # accumulators over subwindows
    H_acc = np.zeros(K, dtype=np.float32)
    B_acc = np.zeros(K, dtype=np.float32)

    used_subwins = 0
    used_patches_total = 0  # 统计“被用于构建H/B的patch次数总和”（非unique）

    # initial patch sampling set
    patch_ids = patch_ids_init.astype(np.int64)

    # iterate subwindows
    n_sub = int(np.ceil(total_T / sub_T))
    for w in range(n_sub):
        ws = t0_global + w * sub_T
        we = min(t0_global + (w + 1) * sub_T, t1_global)
        Tw = int(we - ws)
        if Tw <= max_lag + bin_size:
            continue

        # slice events in [ws, we)
        i0 = np.searchsorted(t_in, ws, side="left")
        i1 = np.searchsorted(t_in, we, side="left")
        if i1 - i0 <= 0:
            continue

        t_w = t_in[i0:i1]
        pid_w = pids[i0:i1]

        # optional: active patch sampling per subwindow
        if bool(CONFIG["use_active_patch_sampling"]):
            uniq, cnt = np.unique(pid_w, return_counts=True)
            active = uniq[cnt >= int(CONFIG["active_count_threshold_subwin"])]
            if len(active) >= 10:
                rng = np.random.default_rng(int(CONFIG["seed"]))
                num_patches = min(int(CONFIG["num_patches"]), len(active))
                patch_ids = rng.choice(active, size=num_patches, replace=False).astype(np.int64)
            else:
                patch_ids = patch_ids_init.astype(np.int64)

        # keep only sampled patches
        mask_pid = np.isin(pid_w, patch_ids)
        pid_w = pid_w[mask_pid]
        t_w = t_w[mask_pid]
        if len(t_w) == 0:
            continue

        # group by pid within this subwindow: sort by (pid, t)
        order = np.lexsort((t_w, pid_w))
        pid_w = pid_w[order]
        t_w = t_w[order]
        del order

        unique_pids, start_idx = np.unique(pid_w, return_index=True)

        H_list = []
        B_list = []
        used_this_w = 0

        for gi, pid in enumerate(unique_pids):
            s = start_idx[gi]
            e = start_idx[gi + 1] if gi + 1 < len(start_idx) else len(pid_w)
            times = t_w[s:e]
            N = int(len(times))
            if N < min_ev_sub:
                continue

            hist = hist_dt_within_window(times, max_lag, bin_size)
            if np.sum(hist) <= 0:
                continue

            B = baseline_B(N=N, T_us=Tw, bin_size_us=bin_size, tau=tau)
            H_list.append(hist.astype(np.float32))
            B_list.append(B)
            used_this_w += 1

        if used_this_w < 5:
            continue

        H_mat = np.stack(H_list, axis=0)
        B_mat = np.stack(B_list, axis=0)

        Hsum, Bsum = trimmed_sums_per_bin(H_mat, B_mat, trim_frac=trim_frac)

        H_acc += Hsum
        B_acc += Bsum
        used_subwins += 1
        used_patches_total += used_this_w

    # finalize g2
    g2 = np.full(K, np.nan, dtype=np.float32)
    valid = B_acc > 1e-12
    g2[valid] = (H_acc[valid] / B_acc[valid]).astype(np.float32)

    # A: deadtime mask
    if bool(CONFIG["use_deadtime_mask"]):
        g2 = apply_deadtime_mask(g2, tau, int(CONFIG["deadtime_us"]))

    # E: tail normalize
    g2 = tail_normalize(g2, tau)

    stats = {"used_patches_total": int(used_patches_total), "used_subwins": int(used_subwins)}
    return g2, stats


def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    ps = int(CONFIG["patch_size"])
    assert ps == 2, "本脚本固定 2×2 patch（patch_size=2）"

    # patch grid in 2×2-cell coordinates
    r0 = CONFIG["row_min"] >> 1
    r1 = CONFIG["row_max"] >> 1
    c0 = CONFIG["col_min"] >> 1
    c1 = CONFIG["col_max"] >> 1
    grid_h = (r1 - r0 + 1)
    grid_w = (c1 - c0 + 1)
    total_patches = grid_h * grid_w

    rng = np.random.default_rng(int(CONFIG["seed"]))
    num_patches = min(int(CONFIG["num_patches"]), total_patches)
    patch_ids_init = rng.choice(total_patches, size=num_patches, replace=False).astype(np.int64)

    max_lag = int(CONFIG["max_lag_us"])
    bin_size = int(CONFIG["bin_size_us"])
    tau = build_tau_centers(max_lag, bin_size)

    curves, flows = [], []
    for fp in CONFIG["test_files"]:
        if not os.path.exists(fp):
            print(f"[WARN] missing: {fp}")
            continue
        flow = parse_flow_from_name(os.path.basename(fp))
        print(f"\n=== Processing: {os.path.basename(fp)} | flow={flow} mm/s ===")
        curve, stats = compute_curve_one_file(
            fp, patch_ids_init, grid_base_r=r0, grid_base_c=c0, grid_w=grid_w, tau=tau
        )
        print(f"  used_subwins = {stats['used_subwins']} | used_patches_total = {stats['used_patches_total']}")
        curves.append(curve)
        flows.append(flow)

    if len(curves) == 0:
        print("No valid files.")
        return

    # plot
    plt.figure(figsize=(12, 7))
    for flow, curve in sorted(zip(flows, curves), key=lambda x: x[0]):
        plt.plot(tau, curve, label=f"{flow:g} mm/s")

    ylabel = "g2 (tail=1, trimmed-sum, subwin-acc)" if CONFIG["use_tail_normalize"] else "g2 (trimmed-sum, subwin-acc)"
    plt.xlabel("Time Lag (us)")
    plt.ylabel(ylabel)

    title = "Method B (tin): Robust g2(τ) (subwindow accumulation)"
    title += f" | Twin={CONFIG['total_window_us']/1e6:.3g}s sub={CONFIG['subwindow_us']/1e6:.3g}s"
    title += f" bin={CONFIG['bin_size_us']}us trim={CONFIG['trim_frac']}"
    if CONFIG["use_deadtime_mask"]:
        title += f" | dead<{CONFIG['deadtime_us']}us masked"
    if CONFIG["use_active_patch_sampling"]:
        title += f" | active_sub>={CONFIG['active_count_threshold_subwin']}"

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if CONFIG["save_plot"]:
        out_png = os.path.join(CONFIG["out_dir"], "robust_g2_subwin.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"\nSaved: {out_png}")

    plt.show()

    print("\n[Practical notes]")
    print("1) min_events_per_patch_subwin 是“每个子窗”的阈值，不要沿用1s窗口的200/300。")
    print("2) 如果 used_subwins 很低：降低 min_events_per_patch_subwin 或开启 active patch sampling。")
    print("3) 若仍混叠：优先调 subwindow_us（50~200ms）与 bin_size_us（50/100）。")


if __name__ == "__main__":
    main()
