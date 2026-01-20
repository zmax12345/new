import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

# ============================================================
# 目的：
#   用 CeleX5 的内部时间戳 tIn 做“in-batch correlation”验证
#   在 ROI 内随机采样 2×2 patch，输出：
#     - 平均归一化相关曲线（按 flow 分组）
#     - tau50 vs flow + 线性拟合 R^2
#     - tauc  vs flow + 线性拟合 R^2
#     - logISI_median vs flow + 线性拟合 R^2
# ============================================================

CONFIG = {
    "test_files": [
        "/data/zm/2026.1.12_testdata/gaoyuzhi/0.2mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/0.5mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/0.8mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/1.0mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/1.2mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/1.5mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/1.8mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/2.0mm_clip.csv",
        "/data/zm/2026.1.12_testdata/gaoyuzhi/2.2mm_clip.csv",
    ],

    # ROI (像素坐标)
    "row_min": 400,
    "row_max": 499,
    "col_min": 0,
    "col_max": 1280,

    # patch采样（2×2）
    "patch_size": 2,
    "num_patches": 200,            # 你自己改：随机采样多少个 2×2 patch
    "seed": 0,

    # 相关统计参数
    "max_lag_us": 3000,
    "bin_size_us": 10,

    # 每个 patch 用固定数量事件（避免 event rate 偏置）
    "target_events_per_patch": 100000,   # 你自己改：每个patch累计多少事件后停止
    "max_batches_to_scan": 300000,      # 最多扫多少个 batch（防止死循环/数据太长）

    # 输出目录
    "out_dir": "/data/zm/2026.1.12_testdata/gaoyuzhi/tin_patch_report",
    "plot_mean_curve": True,
}


# ----------------------- Numba: batch内相关直方图 -----------------------
@njit
def hist_dt_within_batch(sorted_t_in, max_lag_us, bin_size_us):
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


# ----------------------- 工具：拟合 & 特征 -----------------------
def smooth_1d(x, k=5):
    if k <= 1:
        return x
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(x, kernel, mode="same")


def get_tau50(curve, x_us):
    """返回曲线第一次低于0.5对应的tau(us)，如果没有则返回nan"""
    # curve 已经归一化到 curve[5] == 1 附近
    thr = 0.5
    for i in range(1, len(curve)):
        if curve[i] <= thr:
            return float(x_us[i])
    return np.nan


def fit_exp_tau(curve, x_us, y_min=0.1, y_max=0.9):
    """
    拟合 y = A * exp(-t/tau) + c（简化为对 log(y) 线性拟合）
    为鲁棒：只用 y在(0.1,0.9)范围内的点做 log 拟合。
    返回 tau(us)；失败则 nan
    """
    y = curve.copy()
    # 避免数值问题
    y = np.clip(y, 1e-8, None)

    mask = (y >= y_min) & (y <= y_max)
    if np.sum(mask) < 10:
        return np.nan

    t = x_us[mask].astype(np.float64)
    ly = np.log(y[mask].astype(np.float64))

    # ly = a*t + b -> tau = -1/a
    a, b = np.polyfit(t, ly, 1)
    if a >= 0:
        return np.nan
    tau = -1.0 / a
    return float(tau)


def parse_flow_from_name(name):
    """
    从文件名提取流速，优先匹配 "0.2mm"；不行则取第一个浮点数
    """
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mm", name)
    if m:
        return float(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return float(m.group(1))
    return np.nan


def linear_fit_r2(x, y):
    """y = a x + b 的线性拟合 + R^2"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return float(a), float(b), float(r2)


# ----------------------- 核心：tIn in-batch 统计（2×2 patch） -----------------------
def compute_inbatch_curve_for_patches(file_path, patch_ids, grid_base_r, grid_base_c, grid_w):
    """
    patch_ids: 采样的patch id列表（基于 cell_r/cell_c 网格）
    每个事件归属一个2×2 patch：cell_r = row//2, cell_c = col//2
    batch 用 tOff 唯一值来分（CSV按 tOff 输出顺序排列）
    在每个 batch 内，对每个 patch 的 tIn 排序后统计 dt 直方图，然后跨 batch 累加。

    返回：
      curves_clean: [P, num_bins] float32  (归一化+平滑后)
      logisi_median_per_patch: [P] float32
      events_used_per_patch: [P] int
    """
    P = len(patch_ids)
    max_lag = int(CONFIG["max_lag_us"])
    bin_size = int(CONFIG["bin_size_us"])
    num_bins = max_lag // bin_size

    # 每个 patch 累加直方图
    hist_sum = np.zeros((P, num_bins), dtype=np.int64)

    # ISI（相邻tIn间隔）用来取 log-median
    isi_collect = [[] for _ in range(P)]

    # 每个 patch 已累计事件数
    events_used = np.zeros(P, dtype=np.int64)

    # patch_id -> local index
    id2idx = {pid: i for i, pid in enumerate(patch_ids)}

    # 读取 csv：row col t_in t_off
    # 强制数值化，避免你之前遇到的 "str vs int" 问题
    df = pd.read_csv(
        file_path,
        header=None,
        usecols=[0, 1, 2, 3],
        names=["row", "col", "t_in", "t_off"],
        low_memory=False,
    )
    for c in ["row", "col", "t_in", "t_off"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df.astype({"row": np.int32, "col": np.int32, "t_in": np.int64, "t_off": np.int64})

    # ROI filter
    df = df[
        (df["row"] >= CONFIG["row_min"]) & (df["row"] <= CONFIG["row_max"]) &
        (df["col"] >= CONFIG["col_min"]) & (df["col"] <= CONFIG["col_max"])
    ]
    if len(df) == 0:
        raise RuntimeError("ROI 内没有事件，检查 row/col 范围是否匹配")

    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    tin  = df["t_in"].to_numpy()
    toff = df["t_off"].to_numpy()

    # 事件归属的 patch cell（2×2）
    cell_r = rows >> 1
    cell_c = cols >> 1

    # patch_id = (cell_r - base_r)*grid_w + (cell_c - base_c)
    pid_all = (cell_r - grid_base_r) * grid_w + (cell_c - grid_base_c)

    # 只保留采样patch的事件
    # 为快，做成mask
    keep = np.zeros(pid_all.shape[0], dtype=np.bool_)
    # 下面这样写比 np.isin 快且可控
    sampled_set = set(patch_ids)
    for i in range(pid_all.shape[0]):
        if int(pid_all[i]) in sampled_set:
            keep[i] = True
    rows = rows[keep]
    cols = cols[keep]
    tin  = tin[keep]
    toff = toff[keep]
    pid_all = pid_all[keep]

    if len(tin) == 0:
        raise RuntimeError("采样的patch里没有事件，建议增大 num_patches 或扩大 ROI")

    # batch扫描：按 toff 分批（toff 在 CSV 里本来就是顺序的）
    # 当前 batch 各 patch 的 tin 列表
    cur_batch_tin = [[] for _ in range(P)]
    cur_batch = int(toff[0])

    batches_scanned = 0

    def process_one_batch():
        # 对每个 patch：排序 tin，更新直方图 & ISI
        for p in range(P):
            if events_used[p] >= CONFIG["target_events_per_patch"]:
                cur_batch_tin[p].clear()
                continue

            arr = cur_batch_tin[p]
            if len(arr) < 3:
                arr.clear()
                continue

            t = np.array(arr, dtype=np.int64)
            t.sort()

            # ISI（相邻）
            dt = np.diff(t)
            if dt.size > 0:
                # 只保留正的
                dt = dt[dt > 0]
                if dt.size > 0:
                    isi_collect[p].append(dt)

            # 相关直方图
            h = hist_dt_within_batch(t, max_lag, bin_size)
            hist_sum[p, :] += h
            events_used[p] += len(t)

            arr.clear()

    # 主循环
    for i in range(len(toff)):
        if batches_scanned >= CONFIG["max_batches_to_scan"]:
            break

        b = int(toff[i])
        if b != cur_batch:
            # 处理上一批
            process_one_batch()
            batches_scanned += 1
            cur_batch = b

            # 如果全部 patch 都达标，提前结束
            if np.all(events_used >= CONFIG["target_events_per_patch"]):
                break

        pid = int(pid_all[i])
        if pid in id2idx:
            idx = id2idx[pid]
            cur_batch_tin[idx].append(int(tin[i]))

    # 处理最后一个 batch
    process_one_batch()

    # 生成曲线（归一化 + 平滑）
    curves = hist_sum.astype(np.float32)

    # 防止全0
    for p in range(P):
        if np.sum(curves[p]) <= 0:
            continue
        # 归一化：用 lag=50us (bin=5) 对齐（与你当前脚本一致）
        norm_idx = 5
        if norm_idx < curves.shape[1] and curves[p, norm_idx] > 0:
            curves[p] = curves[p] / curves[p, norm_idx]
        curves[p] = smooth_1d(curves[p], k=5)

    # logISI median
    logisi_med = np.full((P,), np.nan, dtype=np.float32)
    for p in range(P):
        if len(isi_collect[p]) == 0:
            continue
        dt_all = np.concatenate(isi_collect[p]).astype(np.float64)
        dt_all = dt_all[dt_all > 0]
        if dt_all.size < 50:
            continue
        logisi_med[p] = np.median(np.log(dt_all + 1e-12)).astype(np.float32)

    return curves, logisi_med, events_used


# ----------------------- 主流程 -----------------------
def main():
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    # patch网格的范围（2×2 cell坐标）
    ps = int(CONFIG["patch_size"])
    assert ps == 2, "本脚本固定 2×2 patch（ps=2）"

    r0 = CONFIG["row_min"] >> 1
    r1 = CONFIG["row_max"] >> 1
    c0 = CONFIG["col_min"] >> 1
    c1 = CONFIG["col_max"] >> 1

    grid_h = (r1 - r0 + 1)
    grid_w = (c1 - c0 + 1)
    total_patches = grid_h * grid_w

    # 随机采样 patch id（网格下标）
    rng = np.random.default_rng(CONFIG["seed"])
    num_patches = int(CONFIG["num_patches"])
    num_patches = min(num_patches, total_patches)
    patch_ids = rng.choice(total_patches, size=num_patches, replace=False).astype(np.int64).tolist()

    print(f"ROI patch grid: {grid_h} x {grid_w}  total={total_patches}")
    print(f"Sampled patches: {len(patch_ids)}  (2×2)")

    max_lag = int(CONFIG["max_lag_us"])
    bin_size = int(CONFIG["bin_size_us"])
    num_bins = max_lag // bin_size
    x_us = (np.arange(num_bins) * bin_size).astype(np.float32)

    # 存每个文件的结果
    all_flows = []
    all_tau50 = []    # list of arrays [P]
    all_tauc  = []
    all_logisi = []
    all_mean_curve = []

    for file_path in CONFIG["test_files"]:
        if not os.path.exists(file_path):
            print(f"skip missing: {file_path}")
            continue

        name = os.path.basename(file_path)
        flow = parse_flow_from_name(name)
        print(f"\n========== {name}  flow={flow} ==========")

        curves, logisi_med, events_used = compute_inbatch_curve_for_patches(
            file_path=file_path,
            patch_ids=patch_ids,
            grid_base_r=r0,
            grid_base_c=c0,
            grid_w=grid_w,
        )

        # 对每个 patch 计算 tau50 / tauc
        tau50 = np.full((len(patch_ids),), np.nan, dtype=np.float32)
        tauc  = np.full((len(patch_ids),), np.nan, dtype=np.float32)

        for p in range(len(patch_ids)):
            y = curves[p]
            if not np.isfinite(y).all() or np.sum(y) <= 0:
                continue
            tau50[p] = get_tau50(y, x_us)
            tauc[p]  = fit_exp_tau(y, x_us, y_min=0.1, y_max=0.9)

        # 统计输出
        used_med = float(np.median(events_used))
        used_min = int(np.min(events_used))
        used_max = int(np.max(events_used))
        print(f"events used per patch: median={used_med:.0f}, min={used_min}, max={used_max}")
        print(f"tau50:  finite patches = {np.sum(np.isfinite(tau50))}/{len(tau50)}")
        print(f"tauc :  finite patches = {np.sum(np.isfinite(tauc ))}/{len(tauc )}")
        print(f"logISI median: finite patches = {np.sum(np.isfinite(logisi_med))}/{len(logisi_med)}")

        # 平均曲线（跨patch取 mean）
        mean_curve = np.nanmean(curves, axis=0)
        all_mean_curve.append((flow, mean_curve))

        all_flows.append(flow)
        all_tau50.append(tau50)
        all_tauc.append(tauc)
        all_logisi.append(logisi_med)

    # 排序（按 flow）
    order = np.argsort(np.array(all_flows))
    all_flows = np.array(all_flows)[order]
    all_tau50 = [all_tau50[i] for i in order]
    all_tauc  = [all_tauc[i]  for i in order]
    all_logisi= [all_logisi[i]for i in order]
    all_mean_curve = [all_mean_curve[i] for i in order]

    # ------------------- 图1：每个flow的平均相关曲线 -------------------
    if CONFIG["plot_mean_curve"]:
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, (flow, mean_curve) in enumerate(all_mean_curve):
            plt.plot(x_us[2:], mean_curve[2:], label=f"{flow} mm/s", color=cmap(i % 10), linewidth=2)
        plt.title("Method B (tIn in-batch): Mean normalized correlation over sampled 2x2 patches")
        plt.xlabel("Time Lag (us)")
        plt.ylabel("Normalized Correlation")
        plt.grid(True)
        plt.legend()
        outp = os.path.join(CONFIG["out_dir"], "mean_curve_methodB_tin_inbatch.png")
        plt.savefig(outp, dpi=200)
        print(f"\n✅ saved: {outp}")

    # ------------------- 图2~4：特征 vs flow + 线性拟合 R^2 -------------------
    def plot_feature_vs_flow(feature_list, title, ylab, fname):
        # 把每个flow的 patch feature flatten 后画散点
        xs = []
        ys = []
        for flow, feat in zip(all_flows, feature_list):
            feat = np.asarray(feat, dtype=np.float64)
            mask = np.isfinite(feat)
            xs.extend([float(flow)] * int(np.sum(mask)))
            ys.extend(feat[mask].tolist())

        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)

        a, b, r2 = linear_fit_r2(xs, ys)

        plt.figure(figsize=(7, 5))
        plt.scatter(xs, ys, s=12)
        if np.isfinite(r2):
            xline = np.linspace(np.min(xs), np.max(xs), 100)
            yline = a * xline + b
            plt.plot(xline, yline, linewidth=2)
            plt.title(f"{title}  (linear R^2={r2:.4f})")
        else:
            plt.title(title + "  (not enough points)")
        plt.xlabel("Flow (mm/s)")
        plt.ylabel(ylab)
        plt.grid(True)
        outp = os.path.join(CONFIG["out_dir"], fname)
        plt.savefig(outp, dpi=200)
        print(f"✅ saved: {outp}")
        if np.isfinite(r2):
            print(f"   fit: y = {a:.6g} x + {b:.6g},  R^2={r2:.6f}")

    plot_feature_vs_flow(
        all_tau50,
        title="Method B (tIn in-batch): tau50 vs flow",
        ylab="tau50 (us)  [R(tau)=0.5 crossing]",
        fname="tau50_vs_flow.png",
    )
    plot_feature_vs_flow(
        all_tauc,
        title="Method B (tIn in-batch): tau_c vs flow",
        ylab="tau_c (us)  [exp fit on normalized R(tau)]",
        fname="tauc_vs_flow.png",
    )
    plot_feature_vs_flow(
        all_logisi,
        title="Method B (tIn in-batch): log-ISI median vs flow",
        ylab="median(log(Δt_in))  [within-batch]",
        fname="logisi_vs_flow.png",
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()
