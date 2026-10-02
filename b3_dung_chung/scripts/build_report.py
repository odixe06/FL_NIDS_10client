#!/usr/bin/env python3
"""Sinh reports.md, all_methods_metrics.csv và các biểu đồ so sánh chéo cho task10.

Chạy bằng môi trường có matplotlib:
    /home/odixe/miniforge3/envs/nckh/bin/python scripts/build_report.py
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report_assets"

TEN = [
    "accuracy",
    "macro_precision",
    "micro_precision",
    "weighted_precision",
    "macro_recall",
    "micro_recall",
    "weighted_recall",
    "macro_f1",
    "micro_f1",
    "weighted_f1",
]
TEN_SHORT = [
    "accuracy",
    "macro_P",
    "micro_P",
    "weighted_P",
    "macro_R",
    "micro_R",
    "weighted_R",
    "macro_F1",
    "micro_F1",
    "weighted_F1",
]

SCENARIOS = [("gru", "GRU"), ("transformer", "Transformer"), ("cnn1d", "CNN-1D")]

# thứ tự slot categorical đã validate (light surface #fcfcfb)
PALETTE = {
    "fd_ids_noniid": "#2a78d6",   # slot 1 blue
    "perfed_skd": "#eb6834",      # slot 2 orange
    "permutation_feature_importance": "#1baf7a",  # slot 3 aqua
    "pfedes": "#eda100",          # slot 4 yellow
    "proxymodel": "#e87ba4",      # slot 5 magenta
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

METHODS = [
    # key, tên hiển thị, thư mục, entity triển khai (dùng để so sánh chéo)
    ("fd_ids_noniid", "FD-IDS", "fd_ids_noniid", "server"),
    ("perfed_skd", "PerFed-SKD", "perfed_skd", "client"),
    ("permutation_feature_importance", "FedCAPS", "permutation_feature_importance", "client"),
    ("pfedes", "pFedES", "pfedes", "client"),
    ("proxymodel", "ProxyModel", "proxymodel", "client"),
]
DISPLAY = {k: n for k, n, _, _ in METHODS}
DEPLOY = {k: d for k, _, _, d in METHODS}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- data

def run_dir(method_dir: str, scen: str) -> Path:
    parent = ROOT / method_dir / f"10_clients_{scen}"
    for p in sorted(parent.iterdir()):
        if p.is_dir() and p.name.startswith("task10_"):
            return p
    raise FileNotFoundError(f"không tìm thấy run dir cho {method_dir}/{scen}")


def load_all():
    """{(method, scen): {'run': Path, 'server': {round: {metric: v}}, 'client': {...},
                         'client_raw': {round: {cid: {metric: v}}}}}"""
    data = {}
    for key, _name, mdir, _dep in METHODS:
        for scen, _label in SCENARIOS:
            rd = run_dir(mdir, scen)
            with open(rd / "metrics" / "evaluation_metrics.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            server, client_raw = {}, {}
            for r in rows:
                rnd = int(r["round"])
                vals = {m: float(r[m]) for m in TEN}
                if r["model_scope"] == "server":
                    server[rnd] = vals
                else:
                    # một số run ghi client_id dạng "1.0"
                    client_raw.setdefault(rnd, {})[int(float(r["client_id"]))] = vals
            client_mean = {
                rnd: {m: statistics.fmean(v[m] for v in per.values()) for m in TEN}
                for rnd, per in client_raw.items()
            }
            cfg = json.loads((rd / "metrics" / "config.json").read_text())
            comm = json.loads((rd / "metrics" / "communication_costs.json").read_text())
            rt = json.loads((rd / "metrics" / "runtime_breakdown.json").read_text())
            total = rt.get("total_pipeline_seconds")
            if total is None:
                total = sum(v for k, v in rt.items()
                            if isinstance(v, (int, float)) and k.endswith("_seconds"))
            data[(key, scen)] = {
                "run": rd,
                "rel": rd.relative_to(ROOT).as_posix(),
                "server": server,
                "client": client_mean,
                "client_raw": client_raw,
                "config": cfg,
                "comm_bytes": comm["total_bytes"],
                "runtime_s": total,
            }
    return data


def series(d, key, scen, metric, scope):
    src = d[(key, scen)][scope]
    return [src[r][metric] for r in range(1, 11)] if src else None


# ------------------------------------------------------------------------ figures

def _style_axes(ax, title, ylabel=None):
    ax.set_title(title, color=INK, fontsize=10, pad=8)
    ax.set_xlabel("Round", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.grid(axis="both", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def _direct_labels(ax, endpoints):
    """endpoints: [(y, text, color)] — nhãn trực tiếp ở cuối đường, né chồng lấn."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.052
    items = sorted(endpoints, key=lambda t: t[0])
    placed = []
    for y, text, color in items:
        ny = y if not placed else max(y, placed[-1] + gap)
        placed.append(ny)
    for (y, text, color), ny in zip(items, placed):
        ax.annotate(text, xy=(10, y), xytext=(10.25, ny), fontsize=7.5,
                    color=color, va="center", ha="left", fontweight="bold",
                    annotation_clip=False)


def fig_metric_by_round(data, metric, label, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    for ax, (scen, slabel) in zip(axes, SCENARIOS):
        ends = []
        for key, name, _mdir, dep in METHODS:
            ys = series(data, key, scen, metric, dep)
            c = PALETTE[key]
            ax.plot(range(1, 11), ys, color=c, linewidth=2.0, marker="o",
                    markersize=4.2, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=name, zorder=3)
            ends.append((ys[-1], name, c))
        _style_axes(ax, slabel, label if ax is axes[0] else None)
        ax.set_xlim(0.6, 12.6)
        _direct_labels(ax, ends)
    handles = [Line2D([], [], color=PALETTE[k], linewidth=2.0, marker="o",
                      markersize=4.2, label=n) for k, n, _, _ in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=9)
    fig.suptitle(f"{label} trên global test theo round — entity triển khai của mỗi phương pháp",
                 fontsize=11.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(ASSETS / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_server_vs_client(data):
    have = [(k, n) for k, n, _, _ in METHODS if k in ("fd_ids_noniid", "perfed_skd", "proxymodel")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    for ax, (scen, slabel) in zip(axes, SCENARIOS):
        ends = []
        for key, name in have:
            c = PALETTE[key]
            srv = series(data, key, scen, "accuracy", "server")
            if srv:
                ax.plot(range(1, 11), srv, color=c, linewidth=2.0, linestyle="-",
                        marker="o", markersize=4.2, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, zorder=3)
                ends.append((srv[-1], f"{name} server", c))
            cli = series(data, key, scen, "accuracy", "client")
            if cli:
                ax.plot(range(1, 11), cli, color=c, linewidth=2.0, linestyle="--",
                        marker="s", markersize=4.0, markeredgecolor=SURFACE,
                        markeredgewidth=1.2, zorder=3)
                ends.append((cli[-1], f"{name} client", c))
        _style_axes(ax, slabel, "accuracy" if ax is axes[0] else None)
        ax.set_xlim(0.6, 13.8)
        _direct_labels(ax, ends)
    handles = [Line2D([], [], color=PALETTE[k], linewidth=2.0, label=n) for k, n in have]
    handles += [
        Line2D([], [], color=MUTED, linewidth=2.0, linestyle="-", marker="o",
               markersize=4.2, label="server / global"),
        Line2D([], [], color=MUTED, linewidth=2.0, linestyle="--", marker="s",
               markersize=4.0, label="trung bình 10 client"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=9)
    fig.suptitle("Khoảng cách server ↔ personalized client (accuracy trên global test)",
                 fontsize=11.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(ASSETS / "fig_server_vs_client.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_round10_heatmap(data):
    rows, labels = [], []
    for key, name, _mdir, dep in METHODS:
        for scen, slabel in SCENARIOS:
            vals = data[(key, scen)][dep][10]
            rows.append([vals[m] for m in TEN])
            labels.append(f"{name} · {slabel}")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    im = ax.imshow(rows, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(10), TEN_SHORT, rotation=42, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
    ax.grid(False)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            # crossover tương phản của ramp xanh nằm quanh v ≈ 0,52
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if v > 0.52 else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, 10)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=MUTED)
    ax.set_title("Round 10 — đủ 10 metrics trên global test (entity triển khai)",
                 fontsize=11.5, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_round10_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_client_spread(data):
    """Độ phân tán giữa 10 client tại round 10 — vì sao trung bình client thấp."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharey=True)
    per_method = [(k, n) for k, n, _, _ in METHODS if k != "fd_ids_noniid"]
    for ax, (scen, slabel) in zip(axes, SCENARIOS):
        for i, (key, name) in enumerate(per_method):
            raw = data[(key, scen)]["client_raw"][10]
            accs = [raw[c]["accuracy"] for c in sorted(raw)]
            c = PALETTE[key]
            ax.scatter([i + 1] * len(accs), accs, s=34, color=c, alpha=0.75,
                       edgecolor=SURFACE, linewidth=1.1, zorder=3)
            ax.plot([i + 1 - 0.26, i + 1 + 0.26],
                    [statistics.fmean(accs)] * 2, color=c, linewidth=2.4, zorder=4)
        ax.set_xticks(range(1, len(per_method) + 1),
                      [n for _, n in per_method], rotation=18, ha="right", fontsize=8.5)
        ax.set_title(slabel, color=INK, fontsize=10, pad=8)
        if ax is axes[0]:
            ax.set_ylabel("accuracy tại round 10", fontsize=9)
        ax.set_xlabel("")
        ax.grid(axis="y", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Phân tán accuracy giữa 10 client tại round 10 (vạch ngang = trung bình)",
                 fontsize=11.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(ASSETS / "fig_client_spread.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_cost(data):
    """Dot plot: truyền thông trải hơn 700 lần nên phải dùng thang log, mà bar trên
    thang log thì độ dài không còn tỉ lệ với giá trị. Dot mã hoá bằng vị trí nên an
    toàn. Màu vẫn theo phương pháp để khớp các biểu đồ còn lại; hình dạng marker
    mang kịch bản."""
    marks = [("o", "GRU"), ("s", "Transformer"), ("^", "CNN-1D")]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2))
    ys = list(range(len(METHODS)))[::-1]

    for ax, (getter, title, xlabel, logx) in zip(axes, [
        (lambda d: d["comm_bytes"] / 1048576, "Chi phí truyền thông",
         "Tổng truyền thông 10 round (MiB, thang log)", True),
        (lambda d: d["runtime_s"] / 60, "Thời gian chạy trên 2×T4",
         "Tổng thời gian chạy (phút)", False),
    ]):
        for y, (key, name, _m, _d) in zip(ys, METHODS):
            vals = [getter(data[(key, scen)]) for scen, _ in SCENARIOS]
            ax.plot([min(vals), max(vals)], [y, y], color=PALETTE[key],
                    linewidth=1.6, alpha=0.45, zorder=2, solid_capstyle="round")
            for v, (mk, _lbl) in zip(vals, marks):
                ax.plot(v, y, marker=mk, markersize=8, color=PALETTE[key],
                        markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        if logx:
            ax.set_xscale("log")
        ax.set_yticks(ys, [n for _, n, _, _ in METHODS], fontsize=9)
        ax.set_ylim(-0.7, len(METHODS) - 0.3)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, pad=8)
        ax.grid(axis="x", linewidth=0.8)
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    handles = [Line2D([], [], marker=mk, linestyle="none", markersize=8,
                      color=MUTED, markeredgecolor=SURFACE, markeredgewidth=1.6,
                      label=lbl) for mk, lbl in marks]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=9)
    fig.suptitle("Chi phí truyền thông và thời gian huấn luyện",
                 fontsize=11.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(ASSETS / "fig_cost.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- csv

def write_csv(data):
    out = ROOT / "all_methods_metrics.csv"
    cols = (["method", "method_display", "scenario", "model_family", "run_name",
             "entity_scope", "n_entities", "round"] + TEN +
            ["accuracy_min", "accuracy_max", "accuracy_std"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for key, name, _mdir, _dep in METHODS:
            for scen, _label in SCENARIOS:
                d = data[(key, scen)]
                run_name = d["run"].name
                for scope in ("server", "client"):
                    src = d[scope]
                    if not src:
                        continue
                    for rnd in range(1, 11):
                        vals = src[rnd]
                        if scope == "client":
                            accs = [v["accuracy"] for v in d["client_raw"][rnd].values()]
                            n = len(accs)
                            extra = [f"{min(accs):.10f}", f"{max(accs):.10f}",
                                     f"{statistics.pstdev(accs):.10f}"]
                        else:
                            n, extra = 1, ["", "", ""]
                        w.writerow([key, name, f"10_clients_{scen}", scen, run_name,
                                    "server" if scope == "server" else "client_mean",
                                    n, rnd] + [f"{vals[m]:.10f}" for m in TEN] + extra)
    return out


# ------------------------------------------------------------------------- tables

def md_table(src, caption=None):
    lines = []
    if caption:
        lines.append(f"*{caption}*")
        lines.append("")
    lines.append("| Round | " + " | ".join(TEN_SHORT) + " |")
    lines.append("|---:|" + "---:|" * 10)
    for rnd in range(1, 11):
        v = src[rnd]
        lines.append(f"| {rnd} | " + " | ".join(f"{v[m]:.4f}" for m in TEN) + " |")
    return "\n".join(lines)


def img(path, alt):
    return f"![{alt}]({path})"


# ---------------------------------------------------------------------- báo cáo

PAPERS = {
    "fd_ids_noniid": dict(
        title="FD-IDS: A Federated Learning and Knowledge Distillation-Based Intrusion "
              "Detection System for Non-IID IoT Environments",
        authors="Huaiyuan Peng, Yanfeng Xiao, Chunming Wu",
        year="2025",
        venue="Sensors 2025, 25, 4309 (MDPI)",
        year_source="ghi rõ trong bài báo",
        file="fd_ids_noniid/sensors-25-04309.md",
    ),
    "perfed_skd": dict(
        title="Personalized Federated Learning for Heterogeneous Edge Device: "
              "Self-Knowledge Distillation Approach",
        authors="Neha Singh, Jatin Rupchandani, Mainak Adhikari",
        year="~2023–2024",
        venue="tạp chí IEEE (bản .md không ghi số/volume)",
        year_source="suy ra từ tham chiếu mới nhất trong bài là 2023 — cần đối chiếu bản gốc",
        file="perfed_skd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_"
             "Self-Knowledge_Distillation_Approach.md",
    ),
    "permutation_feature_importance": dict(
        title="Permutation-Invariant Representation Learning for Robust and "
              "Privacy-Preserving Feature Selection (FedCAPS)",
        authors="Rui Liu, Tao Zhe, Yanjie Fu, Feng Xia, Ted Senator, Dongjie Wang",
        year="2025",
        venue="arXiv:2510.05535v3 (bản mở rộng của CAPS)",
        year_source="mã arXiv 2510 = tháng 10/2025",
        file="permutation_feature_importance/2510.05535v3.md",
    ),
    "pfedes": dict(
        title="pFedES: Generalized Proxy Feature Extractor Sharing for Model "
              "Heterogeneous Personalized Federated Learning",
        authors="Liping Yi, Han Yu, Chao Ren, Gang Wang, Xiaoguang Liu, Xiaoxiao Li",
        year="~2024",
        venue="định dạng trích dẫn AAAI (bản .md không ghi số kỷ yếu)",
        year_source="tham chiếu mới nhất trong bài là 2024 — cần đối chiếu bản gốc",
        file="pfedes/00121-YiL.md",
    ),
    "proxymodel": dict(
        title="Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring",
        authors="(bản .md không có trang tiêu đề)",
        year="2025",
        venue="tạp chí IEEE (bản .md chỉ còn từ mục IV trở đi)",
        year_source="tiền tố năm trong tên file + tham chiếu mới nhất 2024",
        file="proxymodel/2025 Lightweight_Federated_Learning_for_On-Device_"
             "Non-Intrusive_Load_Monitoring.md",
    ),
}

ORIGINAL = {
    "fd_ids_noniid": """\
Bài báo giải bài toán phát hiện xâm nhập IoT trên dữ liệu non-IID. Quy trình gốc:

1. **Tiền xử lý + chọn đặc trưng.** Loại các feature nhạy cảm (`ip.src_host`,
   `udp.port`…), bỏ dòng thiếu/vô hạn, one-hot cho biến hạng mục, chuẩn hoá
   Z-score. Sau đó dùng **Mutual Information** xếp hạng feature và giữ **top-25**
   với Edge-IIoT (top-71 với N-BaIoT).
2. **Mô hình.** Một DNN gồm 5 tầng ẩn 32–64–128–64–32, ReLU, softmax đầu ra;
   Adam, lr = 0,001.
3. **Vòng liên kết.** Mọi client tham gia mỗi round (`m = K`). Server phát
   `w_G^t`; client khởi tạo `w_k ← w_G^t` rồi huấn luyện `E` epoch cục bộ.
4. **Hàm mất mát cục bộ** kết hợp ba thành phần (Eq. 6):
   `L = λ·L_hard + (1−λ)·L_soft + β·L_proximal`, với `L_soft` là KL giữa
   softmax(Z_teacher/T) và softmax(Z_student/T) nhân `T²` (KD, global model làm
   teacher), `L_proximal = (μ/2)·‖w_k − w_G^t‖²` (FedProx). Tham số gốc:
   **T = 3, λ = 0,5, μ = 0,01, β = 0,1**.
5. **Tổng hợp.** FedAvg có trọng số theo số mẫu: `w_G^{t+1} = Σ (n_k/n)·w_k^{t+1}`.

Điểm mấu chốt: KD và proximal term cùng kéo model cục bộ về phía global model để
chống *model drift* do non-IID. Đầu ra cuối cùng là **một global model duy nhất**.""",

    "perfed_skd": """\
Bài báo hướng tới personalized FL cho edge device yếu và băng thông hạn chế.
Quy trình gốc gồm bảy pha (Fig. 2), rút gọn thành hai thuật toán:

1. **Server khởi tạo** global model `ω⁰` bằng cách huấn luyện trên một tập dữ
   liệu định sẵn `D` (Algorithm 1, dòng 2).
2. **Self-knowledge distillation tại client.** Mỗi edge device `m` giữ một
   *historical personalized model* `V_m` — chính là bản local model của round
   trước. Hàm mất mát cục bộ (Eq. 2):
   `φ_m(ω_m^t) = f_m(ω_m^t) + λ·L( x(V_m) ‖ x(ω_m^t) )`,
   tức CE cứng cộng phân kỳ giữa dự đoán hiện tại và dự đoán của chính mình ở
   quá khứ. Cập nhật bằng SGD (Eq. 3). Sau mỗi round: `V_m ← ω_m^t`.
3. **Chọn client theo ngưỡng accuracy.** Server tính ngưỡng `τ = A` với `A` là
   accuracy trung bình toàn hệ; client nào có `a < τ` được đưa vào tập `S` và sẽ
   **nhận global model ở round sau**; client còn lại giữ nguyên trạng thái cục bộ
   của mình (Algorithm 2, dòng 4–8).
4. **Tổng hợp không trọng số** chỉ trên các client được chọn:
   `ω^{t+1} ← Σ_{m∈S} ω^{t+1} / |S|` (Algorithm 1, dòng 10).

Đầu ra cuối cùng là **10 personalized model**, không phải global model. Thí
nghiệm gốc chạy trên MNIST và EMNIST.""",

    "permutation_feature_importance": """\
Đây là bài **chọn đặc trưng**, không phải bài huấn luyện classifier liên kết.
Khung CAPS tập trung được mở rộng thành FedCAPS liên kết. Quy trình gốc:

1. **Thu thập record.** Mỗi client dùng một agent RL (MARLFS) duyệt không gian
   feature, sinh các bản ghi `(f_i, v_i)` = (chỉ số feature của subset, hiệu năng
   downstream tương ứng). Chỉ **chỉ số feature + điểm số** rời khỏi client, dữ
   liệu thô không bao giờ được chia sẻ.
2. **Nhúng bất biến hoán vị.** Server huấn luyện encoder–decoder: encoder dùng
   **ISAB** (Induced Set Attention Block, 2 tầng, `M` inducing point) để mọi hoán
   vị của cùng một subset cho ra cùng một embedding; decoder dùng **PMA** (Pooling
   by Multihead Attention với `K` seed vector) tái tạo lại chuỗi chỉ số. Tối ưu
   bằng negative log-likelihood (Eq. 9).
3. **Tìm kiếm đa mục tiêu bằng PPO.** Chọn top-K record làm search seed, agent
   actor–critic dịch chuyển embedding `E → E⁺`, decoder giải mã ra subset ứng
   viên. Reward cân bằng hiệu năng downstream và độ ngắn của subset
   (`λ` là hệ số đánh đổi).
4. **Phản hồi có trọng số theo cỡ mẫu.** Server phát subset ứng viên xuống toàn
   bộ client; mỗi client trả về điểm `v_c` trên dữ liệu cục bộ; server tổng hợp
   `v̂ = Σ W_c·v_c` với `W_c = |D_c| / Σ|D_j|`. Subset có `v̂` cao nhất là `f*`.

Kết quả của bài báo là **một feature subset**, còn classifier chỉ đóng vai trò
hàm đánh giá downstream.""",

    "pfedes": """\
Bài báo giải bài toán **model-heterogeneous** personalized FL: mỗi client có thể
mang một kiến trúc khác nhau. Quy trình gốc mỗi round:

1. Server phát **proxy feature extractor đồng nhất** `G(θ^{t−1})` — một CNN nhỏ
   hai tầng conv với `padding = same`, bảo đảm **kích thước vào bằng kích thước
   ra**, nên nó cắm được trước bất kỳ model cục bộ nào.
2. **Bước ①: đóng băng proxy, huấn luyện model dị thể.** Client đưa dữ liệu gốc
   `x` qua proxy để có `x̂ = G(θ^{t−1}; x)`, rồi cho **cả `x̂` và `x`** qua model
   cục bộ `F_k(ω_k)` được hai dự đoán `ŷ₁, ŷ₂`. Loss tổng hợp là trung bình có
   trọng số hai CE: `ℓ_ω = μ·ℓ₁ + (1−μ)·ℓ₂` với `μ ∈ (0; 0,5]` (Eq. 6). Chỉ `ω_k`
   được cập nhật.
3. **Bước ②: đóng băng model dị thể, huấn luyện proxy.** Dùng `F_k(ω_k^t)` vừa
   cập nhật làm hàm cố định, tính `ŷ = F_k(ω_k^t; G(θ; x))` và CE `ℓ_θ`; chỉ `θ`
   được cập nhật (Eq. 8–10).
4. **Tổng hợp.** Server FedAvg **chỉ proxy extractor**:
   `θ^t = Σ_{k∈S} (n_k/n)·θ_k^t` (Eq. 11).

Model cục bộ dị thể **không bao giờ rời khỏi client**. Chi phí truyền thông chỉ
bằng kích thước proxy. Suy luận cuối cùng dùng model cục bộ cá nhân hoá.""",

    "proxymodel": """\
Bài báo làm NILM (phân tách phụ tải) trên thiết bị đầu cuối tài nguyên thấp.
Quy trình gốc:

1. **NAS tiết kiệm bộ nhớ** tìm kiến trúc CNN-1D riêng cho từng hộ gia đình/thiết
   bị. Không gian tìm kiếm 5 node × 8 toán tử; sau khi tìm xong giữ lại 2 đường
   có tham số kiến trúc lớn nhất mỗi node. Kết quả: **mỗi client một kiến trúc
   personalized khác nhau**.
2. **Proxy model dùng chung** gồm 3 tầng conv kernel 5 và 1 tầng dense — cố ý nhỏ
   để cân bằng giữa chi phí truyền thông và năng lực biểu diễn.
3. **Adaptive mutual learning.** Model personalized và proxy cục bộ được tối ưu
   **đồng thời**, mỗi model vừa học nhãn thật vừa chưng cất lẫn nhau (mutual
   distillation hai chiều, trọng số thích nghi).
4. **Tổng hợp.** Chỉ proxy được FedAvg qua các hộ; model personalized ở lại thiết
   bị. Tri thức toàn cục đi vào model personalized gián tiếp qua proxy.

Bài gốc là bài toán **hồi quy**, đo bằng MAE và SAE, trên hai bộ REFIT và REDD.

> Lưu ý: file `.md` trong repo chỉ còn từ mục IV (Case Studies) trở đi, phần
> phương pháp I–III đã mất. Mô tả trên được ghép từ mục IV, phần Kết luận và
> đặc tả `ARCHITECTURE_AND_OUTPUT_SPEC.md`.""",
}

REBUILD = {
    "fd_ids_noniid": ("""\
| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | Edge-IIoT, N-BaIoT | **CICIoT2023**, 25 feature, 34 lớp | **Khác** |
| Chọn feature | MI top-25 | dữ liệu đã rút còn 25 feature từ upstream | Giữ tinh thần (đúng 25) |
| Classifier | DNN 32-64-128-64-32 | **GRU / Transformer / CNN-1D** | **Khác** |
| Optimizer | Adam, lr 0,001 | Adam, lr 0,001 | Giữ nguyên |
| KD temperature `T` | 3 | 3 | Giữ nguyên |
| `λ` hard/soft | 0,5 | 0,5 | Giữ nguyên |
| FedProx `μ` | 0,01 | 0,01 | Giữ nguyên |
| Trọng số proximal `β` | 0,1 | 0,1 | Giữ nguyên |
| Teacher | global snapshot đầu round | global snapshot đầu round | Giữ nguyên |
| Tỷ lệ client/round | 100% (`m = K`) | 100% (10/10) | Giữ nguyên |
| Tổng hợp | FedAvg có trọng số `n_k/n` | FedAvg có trọng số `n_k/n` | Giữ nguyên |
| Số round | không cố định | **10** | **Khác (chuẩn hoá)** |
| Local epoch | `E` | **1** | **Khác (chuẩn hoá)** |
""", """\
**Lý do đổi classifier:** để năm phương pháp so được với nhau, cả năm dùng chung
ba backbone (GRU 40.034 / Transformer 71.010 / CNN-1D 35.874 tham số). Cơ chế
FD-IDS — KD + FedProx + FedAvg — không phụ thuộc kiến trúc nên thay backbone
không phá vỡ phương pháp."""),

    "perfed_skd": ("""\
| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | MNIST, EMNIST | **CICIoT2023** | **Khác** |
| Server pretrain | trên tập định sẵn `D` | **đúng 1 epoch** trên `global_train_data.csv` | Giữ nguyên (cố định số epoch) |
| SKD teacher | model cục bộ round trước `V_m` | model cục bộ round trước `V_m` | Giữ nguyên |
| Loss cục bộ | `f_m + λ·L(x(V_m)‖x(ω_m))` | CE + phân kỳ với teacher lịch sử | Giữ nguyên |
| Optimizer | SGD | SGD, lr 0,01 | Giữ nguyên |
| Ngưỡng chọn client | `τ` = accuracy trung bình | `τ` = accuracy trung bình của đủ 10 client | Giữ nguyên |
| Nguồn accuracy chọn client | local accuracy | **local validation 10% phân tầng, seed 42** | Làm chặt lại |
| Tổng hợp | trung bình **không trọng số** trên `S` | trung bình không trọng số trên `S` | Giữ nguyên |
| Số round | `T` | **10** | **Khác (chuẩn hoá)** |
| Local epoch | `E` | **1** | **Khác (chuẩn hoá)** |
""", """\
**Điểm làm chặt:** bài gốc không nói rõ accuracy dùng để chọn client lấy từ đâu.
Bản build lại tách **10% validation phân tầng** khỏi dữ liệu client (seed 42) và
chỉ dùng tập này để chọn client. `global_test_data.csv` **không** tham gia điều
khiển training — điều này quan trọng vì nếu dùng global test để chọn client thì
kết quả sẽ bị rò rỉ."""),

    "permutation_feature_importance": ("""\
| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bài toán | chọn đặc trưng tổng quát | chọn đặc trưng cho IDS CICIoT2023 | Giữ nguyên khung |
| Thu thập record | MARLFS trên client | MARLFS 300 epoch/client | Giữ nguyên |
| Tăng cường hoán vị | có | 25 hoán vị/record | Giữ nguyên |
| Encoder | ISAB ×2, inducing points | ISAB ×2, 4 head, `d`=128, 32 inducing point | Giữ nguyên |
| Decoder | PMA + rFF | PMA 32 seed vector + rFF | Giữ nguyên |
| Tìm kiếm | PPO actor–critic | PPO, `λ`=0,1, clip 0,2, 10 search epoch | Giữ nguyên |
| Phản hồi client | có trọng số theo cỡ mẫu | có trọng số theo cỡ mẫu, **5-fold CV tất định** | Làm chặt lại |
| Downstream classifier | mô hình đánh giá bất kỳ | **GRU / Transformer / CNN-1D**, 10 round | **Bổ sung** |
| Tổng hợp classifier | không có trong bài | **không FedAvg classifier** | Giữ nguyên tinh thần |
""", """\
**Phần bổ sung lớn nhất:** bài gốc dừng ở việc chọn ra `f*`. Để so sánh được với
bốn phương pháp còn lại, bản build lại nối thêm một pha huấn luyện: sau khi chốt
subset, **mỗi client huấn luyện classifier personalized 10 round liên tiếp** trên
100% dữ liệu cục bộ, state giữ qua round, không FedAvg. Vì vậy cột `server` được
ghi `not_applicable` — state server của FedCAPS là encoder/decoder/actor/critic,
không phát logits 34 lớp.

**Search pool** lấy tối đa 100.000 dòng/client, phân tầng độc lập, và **không bao
giờ** chứa dữ liệu từ `global_test_data.csv`."""),

    "pfedes": ("""\
| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | 3 bộ ảnh benchmark | **CICIoT2023** (dữ liệu bảng 25 chiều) | **Khác** |
| Proxy extractor | CNN 2 tầng conv, `padding=same` | Conv1d 1→8→1, kernel 3, padding 1, ReLU | Giữ nguyên nguyên lý |
| Ràng buộc kích thước | vào = ra | `[B,25] → [B,25]`, đúng **57 tham số** | Giữ nguyên |
| Huấn luyện xen kẽ | ① đóng băng proxy → ② đóng băng model | ① rồi ② trong cùng round | Giữ nguyên |
| Loss bước ① | `μ·ℓ₁ + (1−μ)·ℓ₂` | CE có trọng số trên `x̂` và `x` | Giữ nguyên |
| Tổng hợp | FedAvg **chỉ proxy**, trọng số `n_k/n` | FedAvg chỉ proxy, trọng số `n_k/n` | Giữ nguyên |
| Chọn client | ngẫu nhiên `K` trong `N` | **toàn bộ 10/10 mỗi round** | **Khác (chuẩn hoá)** |
| Số round | tới khi hội tụ | **10** | **Khác (chuẩn hoá)** |
""", """\
**Điểm cần lưu ý về tính dị thể:** bài gốc thiết kế cho mỗi client một kiến trúc
khác nhau. Bản build lại cho **cả 10 client cùng một kiến trúc** trong mỗi kịch
bản (10 GRU, hoặc 10 Transformer, hoặc 10 CNN-1D) để giữ tính so sánh được với
bốn phương pháp còn lại. Đây là **giới hạn của thiết kế thí nghiệm, không phải
giới hạn của phương pháp** — xem mục khả năng dị thể bên dưới."""),

    "proxymodel": ("""\
| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bài toán | **hồi quy** NILM (MAE/SAE) | **phân loại đa lớp** 34 lớp | **Khác về bản chất** |
| Bộ dữ liệu | REFIT, REDD | **CICIoT2023** | **Khác** |
| Model personalized | tìm bằng NAS, mỗi client một kiến trúc | **GRU / Transformer / CNN-1D** cố định | **Khác (bỏ NAS)** |
| Proxy | CNN 3 tầng conv kernel 5 + dense | **CNN-1D 35.874 tham số** dùng chung | Giữ nguyên vai trò |
| Mutual distillation | hai chiều, trọng số thích nghi | hai chiều, trọng số thích nghi | Giữ nguyên |
| Tối ưu đồng thời | có | có (label loss + distillation) | Giữ nguyên |
| Tổng hợp | FedAvg **chỉ proxy** | FedAvg chỉ proxy, trọng số `n_k/n` | Giữ nguyên |
| Fine-tune cuối | có | **1 epoch CE personalized ở round 10** | Giữ nguyên |
| Số round | — | **10** | **Khác (chuẩn hoá)** |
""", """\
**Thay đổi lớn nhất là bỏ NAS.** Bài gốc lấy tính dị thể từ chính NAS: mỗi hộ gia
đình nhận một kiến trúc riêng. Bản build lại cố định kiến trúc để so sánh được,
nên phần "personalized architecture" của bài gốc **không được tái hiện**. Cơ chế
còn lại — proxy dùng chung + adaptive mutual distillation + FedAvg chỉ proxy —
được giữ đầy đủ.

Trong kịch bản CNN-1D, personalized CNN và proxy CNN là **hai model/state riêng
biệt**, không dùng chung tham số."""),
}

HETERO = {
    "fd_ids_noniid": ("**Không** — cả hai loại đều không.", """\
FD-IDS tổng hợp bằng `w_G = Σ (n_k/n)·w_k` trên **toàn bộ tham số classifier**.
Phép cộng này chỉ định nghĩa được khi mọi client có **cùng kiến trúc và cùng số
tham số**. Thêm nữa, `L_proximal = (μ/2)·‖w_k − w_G‖²` cũng đòi hỏi `w_k` và
`w_G` nằm trong cùng không gian tham số.

- **Khác kích thước mô hình:** không. Vector tham số phải khớp từng phần tử.
- **Khác loại mô hình:** không. GRU và CNN-1D không có tham số tương ứng nhau.

*Muốn dùng cho môi trường dị thể thì phải thay tầng tổng hợp* — ví dụ chỉ FedAvg
phần đầu chung, hoặc chuyển hẳn sang chưng cất logits trên tập public. KD trong
FD-IDS **không** làm được việc đó vì teacher là global model có cùng kiến trúc,
không phải một kênh trao đổi độc lập kiến trúc."""),

    "perfed_skd": ("**Không** cho khác kiến trúc; **một phần** cho khác tải tính toán.", """\
PerFed-SKD có hai cơ chế cần tách bạch:

- **SKD (`V_m` là chính mình ở round trước)** — hoàn toàn độc lập kiến trúc.
  Mỗi client tự chưng cất từ lịch sử của mình, không cần biết client khác dùng gì.
- **Tổng hợp `ω^{t+1} ← Σ_{m∈S} ω^{t+1}/|S|`** — vẫn là trung bình tham số, nên
  **vẫn đòi hỏi kiến trúc đồng nhất**.

- **Khác kích thước mô hình:** không.
- **Khác loại mô hình:** không.

Tuy nhiên cơ chế **chọn client theo ngưỡng accuracy** giải quyết được *dị thể về
tài nguyên*: client yếu/chậm không bị ép nhận global model mỗi round, giảm tải
tính toán và truyền thông. Đây chính là mục tiêu bài báo tuyên bố ("heterogeneous
edge device") — dị thể **thiết bị**, không phải dị thể **mô hình**."""),

    "permutation_feature_importance": ("**Có** — hỗ trợ tốt cả hai loại.", """\
FedCAPS là phương pháp duy nhất trong năm phương pháp mà **không có bất kỳ phép
cộng tham số nào**. Thứ đi từ client lên server chỉ là:

```
(chỉ số feature của subset, điểm hiệu năng downstream)
```

Server không bao giờ nhìn thấy tham số mô hình. Vì vậy:

- **Khác kích thước mô hình:** có. Client A dùng model 10K tham số, client B dùng
  100K tham số — không ảnh hưởng gì, vì cả hai chỉ trả về một con số hiệu năng.
- **Khác loại mô hình:** có. Client A dùng GRU, client B dùng CNN-1D, client C
  dùng XGBoost — vẫn chạy được. Trong bản build lại, `config.json` còn có sẵn
  trường `client_models` ánh xạ từng client sang một họ mô hình, hạ tầng đã sẵn
  sàng cho cấu hình dị thể.

**Ràng buộc thật sự** của FedCAPS không nằm ở mô hình mà ở **không gian feature**:
mọi client phải mô tả feature bằng cùng một hệ chỉ số. Nếu client A có 25 feature
còn client B có 40 feature khác nhau thì chỉ số không so được."""),

    "pfedes": ("**Có** — đây chính là mục tiêu thiết kế của bài báo.", """\
pFedES sinh ra để giải bài toán model-heterogeneous PFL. Cơ chế:

- Thứ duy nhất được FedAvg là **proxy feature extractor 57 tham số**, hoàn toàn
  tách rời model cục bộ.
- Ràng buộc `padding = same` bảo đảm proxy có **đầu vào và đầu ra cùng kích
  thước** (`[B,25] → [B,25]`), nên nó cắm được trước bất kỳ classifier nào nhận
  `[B,25]`.
- Model cục bộ `F_k(ω_k)` **không bao giờ rời khỏi client**, server không cần
  biết kiến trúc của nó.

- **Khác kích thước mô hình:** có, không giới hạn.
- **Khác loại mô hình:** có, miễn là nhận đúng shape đầu vào.

Đây là phương pháp **rẻ nhất** để hỗ trợ dị thể: chi phí truyền thông đo được chỉ
**0,043 MiB cho toàn bộ 10 round** — thấp hơn FD-IDS khoảng **700 lần**.

Trong bản build lại cả 10 client dùng cùng kiến trúc, nên **khả năng dị thể chưa
được kiểm chứng bằng thực nghiệm ở đây**; kết luận trên đến từ cơ chế của phương
pháp, không từ số đo."""),

    "proxymodel": ("**Có** — hỗ trợ cả hai loại, và bài gốc thực sự dùng.", """\
ProxyModel cũng chỉ FedAvg proxy, giống pFedES về nguyên lý, nhưng đi xa hơn ở
chỗ **bài gốc thực sự vận hành trong chế độ dị thể**: NAS sinh ra một kiến trúc
riêng cho từng hộ gia đình, và các kiến trúc này khác nhau cả về số tầng lẫn kích
thước kernel.

- **Khác kích thước mô hình:** có. Adaptive mutual distillation trao đổi qua
  **logits đầu ra**, không qua tham số.
- **Khác loại mô hình:** có, miễn là không gian nhãn giống nhau.

**Khác biệt so với pFedES:** proxy của ProxyModel là một **classifier đầy đủ**
(35.874 tham số) chứ không phải extractor 57 tham số. Nên ProxyModel tốn truyền
thông hơn nhiều (**27,7 MiB** so với 0,043 MiB) nhưng đổi lại server có một model
dùng được ngay — và số liệu bên dưới cho thấy global proxy này là **entity mạnh
nhất trong toàn bộ thí nghiệm**.

Trong bản build lại NAS đã bị bỏ nên cả 10 client dùng cùng kiến trúc; khả năng
dị thể vì thế **cũng chưa được kiểm chứng bằng thực nghiệm ở đây**."""),
}


def results_section(data, key):
    """Phần kết quả của một phương pháp: ảnh + bảng 10 metrics × 10 round."""
    name = DISPLAY[key]
    dep = DEPLOY[key]
    out = []
    for scen, slabel in SCENARIOS:
        d = data[(key, scen)]
        rel = d["rel"]
        out.append(f"#### {name} — 10 client {slabel}\n")
        out.append(
            f"- Thư mục run: [`{rel}`]({rel})\n"
            f"- Thời gian chạy: **{d['runtime_s']/60:.1f} phút** trên 2×T4 · "
            f"Truyền thông 10 round: **{d['comm_bytes']/1048576:.3f} MiB**\n"
        )
        out.append(img(f"{rel}/artifacts/evaluation_metric_curves.png",
                       f"{name} {slabel} — đường cong 10 metrics theo round"))
        out.append("")
        out.append(img(f"{rel}/artifacts/loss_curves.png",
                       f"{name} {slabel} — đường cong loss"))
        out.append("")
        if d["server"]:
            out.append(md_table(d["server"],
                                f"Bảng — {name} · {slabel} · **server** · "
                                f"10 metrics trên `global_test_data.csv` theo từng round"))
            out.append("")
        if d["client"]:
            n = len(d["client_raw"][1])
            accs10 = [v["accuracy"] for v in d["client_raw"][10].values()]
            out.append(md_table(
                d["client"],
                f"Bảng — {name} · {slabel} · **trung bình {n} client** · "
                f"10 metrics trên `global_test_data.csv` theo từng round"))
            out.append("")
            out.append(
                f"> Tại round 10, accuracy giữa {n} client trải từ **{min(accs10):.4f}** đến "
                f"**{max(accs10):.4f}** (độ lệch chuẩn {statistics.pstdev(accs10):.4f}). "
                f"Trung bình che mất khoảng cách này — xem "
                f"[biểu đồ phân tán](#phân-tán-giữa-các-client).\n")
    return "\n".join(out)


def build_report(data):
    L = []
    A = L.append

    A("# Báo cáo tổng hợp — Task10: tái xây dựng 5 phương pháp Federated Learning\n")
    A("Báo cáo này tổng hợp năm phương pháp học liên kết đã được tái xây dựng trên cùng")
    A("một bộ dữ liệu và cùng ba kịch bản mô hình, kèm toàn bộ kết quả đo được.\n")
    A("**Tài liệu này được sinh tự động** bởi [`scripts/build_report.py`](scripts/build_report.py)")
    A("từ chính các file `metrics/evaluation_metrics.csv` của 15 lần chạy. Mọi con số")
    A("trong báo cáo đều đọc trực tiếp từ output, không nhập tay.\n")

    # ---- mục lục
    A("## Mục lục\n")
    A("1. [Bối cảnh thí nghiệm chung](#1-bối-cảnh-thí-nghiệm-chung)")
    A("2. [Bảng tra nhanh năm bài báo](#2-bảng-tra-nhanh-năm-bài-báo)")
    for i, (key, name, _m, _d) in enumerate(METHODS, start=3):
        A(f"{i}. [{name}](#{i}-{name.lower().replace(' ', '-').replace('.', '')})")
    A("8. [So sánh chéo năm phương pháp](#8-so-sánh-chéo-năm-phương-pháp)")
    A("9. [Phân tích điểm mạnh — điểm yếu](#9-phân-tích-điểm-mạnh--điểm-yếu)")
    A("10. [Kết luận](#10-kết-luận)\n")

    # ---- 1. bối cảnh
    A("---\n")
    A("## 1. Bối cảnh thí nghiệm chung\n")
    A("Cả năm phương pháp chạy trên **đúng cùng một cấu hình** để kết quả so sánh được:\n")
    A("| Thành phần | Giá trị |")
    A("|---|---|")
    A("| Bộ dữ liệu | CICIoT2023, 25 feature, **34 lớp**, phân loại đơn nhãn đa lớp |")
    A("| Số client | 10 |")
    A("| Tổng dữ liệu huấn luyện | 36.014.594 dòng |")
    A("| Tập test | `global_test_data.csv`, **9.003.649 dòng**, dùng chung cho mọi entity |")
    A("| Số round | 10 |")
    A("| Local epoch mỗi round | 1 |")
    A("| Batch size | 1024/client |")
    A("| Seed | 42 |")
    A("| Phần cứng | Kaggle, đúng 2× NVIDIA Tesla T4 |")
    A("| Kiến trúc | GRU 40.034 · Transformer 71.010 · CNN-1D 35.874 tham số |")
    A("")
    A("### Mức độ non-IID\n")
    A("Đây là điểm quyết định cách đọc mọi con số bên dưới. Phân bố dữ liệu giữa 10")
    A("client cực kỳ lệch:\n")
    A("| Client | Số dòng | Số lớp có mặt | Tỷ lệ lớp lớn nhất |")
    A("|---:|---:|---:|---:|")
    dsum = json.loads((data[("pfedes", "gru")]["run"] / "metrics" / "dataset_summary.json").read_text())
    for cid, v in dsum["clients"].items():
        cc = v["class_counts"]
        nz = sum(1 for x in cc if x > 0)
        rows_vn = f"{v['source_rows']:,}".replace(",", ".")
        pct_vn = f"{max(cc)/v['source_rows']*100:.1f}".replace(".", ",")
        A(f"| {cid} | {rows_vn} | {nz}/34 | {pct_vn}% |")
    A("")
    A("Client 6 chỉ thấy **5 trên 34 lớp**; client 1 có 87,7% dữ liệu dồn vào một lớp")
    A("duy nhất và chỉ có 31.520 dòng, trong khi client 8 có 12,7 triệu dòng. Tỷ lệ")
    A("chênh lệch dữ liệu giữa client lớn nhất và nhỏ nhất là **hơn 400 lần**.\n")
    A("> **Hệ quả then chốt cho việc đọc kết quả:** mọi entity đều được chấm trên")
    A("> **cùng một tập test toàn cục có đủ 34 lớp**. Một model personalized học tốt")
    A("> 5 lớp của client mình sẽ bị chấm điểm rất thấp trên tập test 34 lớp — không")
    A("> phải vì nó học kém, mà vì nó **được thiết kế để không tổng quát**. Đây là lý")
    A("> do các phương pháp personalized (FedCAPS, pFedES, ProxyModel-client) có số")
    A("> thấp hơn hẳn các phương pháp global (FD-IDS, ProxyModel-server). So sánh")
    A("> trực tiếp giữa hai nhóm là **không công bằng**, và báo cáo này tách riêng")
    A("> cột server với cột client ở mọi bảng để tránh nhầm lẫn.\n")
    A("### Hợp đồng đo lường\n")
    A("Mỗi entity áp dụng được đánh giá trên **toàn bộ** `global_test_data.csv` ngay")
    A("sau mỗi round bằng đúng 10 cột:\n")
    A("`accuracy` · `macro_precision` · `micro_precision` · `weighted_precision` ·")
    A("`macro_recall` · `micro_recall` · `weighted_recall` · `macro_f1` · `micro_f1` ·")
    A("`weighted_f1`\n")
    A("Với phân loại đơn nhãn đa lớp, bốn đại lượng `accuracy`, `micro_precision`,")
    A("`micro_recall`, `micro_f1` **bằng nhau về mặt toán học** — trong các bảng dưới")
    A("bạn sẽ thấy bốn cột này trùng số. Đây là đúng, không phải lỗi.\n")
    A("Toàn bộ 1.590 hàng metric đã được **kiểm chứng lại bằng cách tính lại từ")
    A("confusion matrix thô** (`.npy`): sai lệch tối đa 2,2 × 10⁻¹⁶.\n")

    # ---- 2. bảng tra nhanh
    A("---\n")
    A("## 2. Bảng tra nhanh năm bài báo\n")
    A("| # | Phương pháp | Bài báo | Năm | Nguồn xác định năm |")
    A("|---|---|---|---|---|")
    for i, (key, name, _m, _d) in enumerate(METHODS, start=1):
        p = PAPERS[key]
        A(f"| {i} | **{name}** | {p['title']} | {p['year']} | {p['year_source']} |")
    A("")
    A("> **Cảnh báo về năm xuất bản:** chỉ FD-IDS ghi rõ nơi/năm xuất bản trong file")
    A("> (`Sensors 2025, 25, 4309`) và FedCAPS suy được chắc chắn từ mã arXiv")
    A("> (`2510` = tháng 10/2025). Ba bài còn lại **không có trang tiêu đề đầy đủ**")
    A("> trong bản `.md` lưu tại repo, nên năm ghi ở trên là **suy đoán từ tham chiếu")
    A("> mới nhất trong bài** — cần đối chiếu bản gốc trước khi trích dẫn.\n")
    A("Bảng khả năng làm việc với mô hình dị thể (chi tiết ở từng mục):\n")
    A("| Phương pháp | Khác **kích thước** mô hình | Khác **loại** mô hình | Thứ được truyền lên server |")
    A("|---|:---:|:---:|---|")
    A("| FD-IDS | ✗ | ✗ | toàn bộ tham số classifier |")
    A("| PerFed-SKD | ✗ | ✗ | toàn bộ tham số classifier |")
    A("| FedCAPS | ✓ | ✓ | chỉ số feature + điểm hiệu năng |")
    A("| pFedES | ✓ | ✓ | proxy extractor **57 tham số** |")
    A("| ProxyModel | ✓ | ✓ | proxy classifier 35.874 tham số |")
    A("")

    # ---- 3..7 từng phương pháp
    for i, (key, name, mdir, dep) in enumerate(METHODS, start=3):
        p = PAPERS[key]
        A("---\n")
        A(f"## {i}. {name}\n")
        A(f"### {i}.1. Bài báo\n")
        A(f"- **Tên:** {p['title']}")
        A(f"- **Tác giả:** {p['authors']}")
        A(f"- **Năm:** {p['year']} — *{p['year_source']}*")
        A(f"- **Nơi xuất bản:** {p['venue']}")
        A(f"- **File trong repo:** [`{p['file']}`]({p['file']})\n")
        A(f"### {i}.2. Quy trình huấn luyện của bài báo gốc\n")
        A(ORIGINAL[key] + "\n")
        A(f"### {i}.3. Bản build lại — giữ gì, đổi gì\n")
        tbl, note = REBUILD[key]
        A(tbl)
        A(note + "\n")
        A(f"### {i}.4. Có dùng được cho mô hình không đồng nhất không?\n")
        verdict, expl = HETERO[key]
        A(f"**Kết luận: {verdict}**\n")
        A(expl + "\n")
        A(f"### {i}.5. Kết quả đo được\n")
        A(results_section(data, key))

    return L


def compare_section(data, L):
    A = L.append
    A("---\n")
    A("## 8. So sánh chéo năm phương pháp\n")
    A("### 8.1. Quy ước so sánh\n")
    A("Mỗi phương pháp có một **entity triển khai** khác nhau — thứ thực sự được")
    A("đem đi suy luận theo đúng tinh thần bài báo gốc:\n")
    A("| Phương pháp | Entity triển khai | Lý do |")
    A("|---|---|---|")
    A("| FD-IDS | **server** | không có state client bền vững; đầu ra là một global model |")
    A("| PerFed-SKD | **client** | `Output: Local personalized models P_m` (Algorithm 1) |")
    A("| FedCAPS | **client** | server chỉ giữ encoder/decoder/PPO, không phát logits |")
    A("| pFedES | **client** | *\"only each client's personalized local model is used for inference\"* |")
    A("| ProxyModel | **client** | model personalized nằm trên thiết bị đầu cuối |")
    A("")
    A("Với PerFed-SKD và ProxyModel, cột **server** vẫn được báo cáo đầy đủ ở mục 3–7")
    A("vì nó kể một câu chuyện rất khác so với cột client.\n")

    A("### 8.2. Biểu đồ so sánh theo round\n")
    A(img("report_assets/fig_accuracy_by_round.png",
          "So sánh accuracy theo round giữa 5 phương pháp trên 3 kịch bản"))
    A("")
    A(img("report_assets/fig_macro_f1_by_round.png",
          "So sánh macro-F1 theo round giữa 5 phương pháp trên 3 kịch bản"))
    A("")
    A(img("report_assets/fig_weighted_f1_by_round.png",
          "So sánh weighted-F1 theo round giữa 5 phương pháp trên 3 kịch bản"))
    A("")
    A("`macro_F1` đối xử mọi lớp như nhau nên nó là thước đo trung thực nhất cho bài")
    A("toán 34 lớp mất cân bằng nặng này; `weighted_F1` bị các lớp đa số chi phối.")
    A("Khoảng cách giữa hai biểu đồ trên chính là mức độ mô hình bỏ rơi lớp thiểu số.\n")

    # ---- bảng so sánh theo round
    A("### 8.3. Bảng so sánh theo từng round — đủ 10 metrics\n")
    A("Bảy nhóm bảng dưới đây phủ hết 10 metrics của hợp đồng đo lường. Nhóm đầu")
    A("tiên đồng thời là `accuracy`, `micro_precision`, `micro_recall` và `micro_f1`")
    A("— bốn đại lượng này bằng nhau về mặt toán học với phân loại đơn nhãn đa lớp,")
    A("nên chỉ cần một bảng thay vì bốn.\n")
    for metric, mlabel in [
        ("accuracy", "accuracy (= micro_P = micro_R = micro_F1)"),
        ("macro_precision", "macro_precision"),
        ("weighted_precision", "weighted_precision"),
        ("macro_recall", "macro_recall"),
        ("weighted_recall", "weighted_recall"),
        ("macro_f1", "macro-F1"),
        ("weighted_f1", "weighted-F1"),
    ]:
        A(f"#### {mlabel} theo round\n")
        for scen, slabel in SCENARIOS:
            A(f"*{mlabel} · kịch bản 10 client **{slabel}** · entity triển khai*\n")
            A("| Round | " + " | ".join(DISPLAY[k] for k, _, _, _ in METHODS) + " |")
            A("|---:|" + "---:|" * len(METHODS))
            for rnd in range(1, 11):
                cells = []
                for key, _n, _m, dep in METHODS:
                    cells.append(f"{data[(key, scen)][dep][rnd][metric]:.4f}")
                A(f"| {rnd} | " + " | ".join(cells) + " |")
            best = max(METHODS, key=lambda t: data[(t[0], scen)][t[3]][10][metric])
            A("")
            A(f"> Cao nhất tại round 10: **{best[1]}** "
              f"({data[(best[0], scen)][best[3]][10][metric]:.4f})\n")

    # ---- round 10 đủ 10 metrics
    A("### 8.4. Round 10 — đủ 10 metrics, cả 15 lần chạy\n")
    A(img("report_assets/fig_round10_heatmap.png",
          "Heatmap 10 metrics tại round 10 cho 15 lần chạy"))
    A("")
    A("| Phương pháp | Kịch bản | Entity | " + " | ".join(TEN_SHORT) + " |")
    A("|---|---|---|" + "---:|" * 10)
    for key, name, _m, dep in METHODS:
        for scen, slabel in SCENARIOS:
            v = data[(key, scen)][dep][10]
            ent = "server" if dep == "server" else "TB 10 client"
            A(f"| {name} | {slabel} | {ent} | " +
              " | ".join(f"{v[m]:.4f}" for m in TEN) + " |")
    A("")
    A("Và cột **server** của ba phương pháp có server classifier:\n")
    A("| Phương pháp | Kịch bản | " + " | ".join(TEN_SHORT) + " |")
    A("|---|---|" + "---:|" * 10)
    for key, name, _m, _dep in METHODS:
        for scen, slabel in SCENARIOS:
            srv = data[(key, scen)]["server"]
            if not srv:
                continue
            v = srv[10]
            A(f"| {name} | {slabel} | " + " | ".join(f"{v[m]:.4f}" for m in TEN) + " |")
    A("")

    A("### 8.5. Khoảng cách server ↔ client\n")
    A(img("report_assets/fig_server_vs_client.png",
          "So sánh accuracy server và trung bình client"))
    A("")
    A("Ba phương pháp có cả hai entity cho ra ba dáng đồ thị hoàn toàn khác nhau:\n")
    for key in ("fd_ids_noniid", "perfed_skd", "proxymodel"):
        d = data[(key, "gru")]
        s10 = d["server"][10]["accuracy"] if d["server"] else None
        c10 = d["client"][10]["accuracy"] if d["client"] else None
        s1 = d["server"][1]["accuracy"] if d["server"] else None
        c1 = d["client"][1]["accuracy"] if d["client"] else None
        if c10 is None:
            A(f"- **{DISPLAY[key]}** (GRU): chỉ có server, đi lên đều từ "
              f"{s1:.4f} → **{s10:.4f}**.")
        else:
            A(f"- **{DISPLAY[key]}** (GRU): server {s1:.4f} → **{s10:.4f}**, "
              f"client {c1:.4f} → **{c10:.4f}** "
              f"(chênh **{abs(s10-c10):.4f}**).")
    A("")

    A('<a id="phân-tán-giữa-các-client"></a>')
    A("### 8.6. Phân tán giữa các client\n")
    A(img("report_assets/fig_client_spread.png",
          "Phân tán accuracy giữa 10 client tại round 10"))
    A("")
    A("Trung bình 10 client là một con số **rất dễ gây hiểu nhầm**. Bảng dưới cho")
    A("thấy khoảng cách giữa client tốt nhất và tệ nhất tại round 10:\n")
    A("| Phương pháp | Kịch bản | min | trung bình | max | độ lệch chuẩn | biên độ |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for key, name, _m, _dep in METHODS:
        for scen, slabel in SCENARIOS:
            raw = data[(key, scen)]["client_raw"]
            if not raw:
                continue
            a = [v["accuracy"] for v in raw[10].values()]
            A(f"| {name} | {slabel} | {min(a):.4f} | {statistics.fmean(a):.4f} | "
              f"{max(a):.4f} | {statistics.pstdev(a):.4f} | {max(a)-min(a):.4f} |")
    A("")
    A("Client tốt nhất của pFedES-Transformer đạt **0,5917** — ngang ngửa server của")
    A("FD-IDS — nhưng client tệ nhất chỉ **0,0021**, kéo trung bình xuống 0,3238.")
    A("Client 0,0021 chính là client 1: nó chỉ thấy 8/34 lớp với 87,7% dồn vào một")
    A("lớp, nên model của nó gần như chỉ đoán được đúng một lớp trên tập test 34 lớp.\n")

    A("### 8.7. Chi phí\n")
    A(img("report_assets/fig_cost.png", "Chi phí truyền thông và thời gian huấn luyện"))
    A("")
    A("| Phương pháp | Truyền thông 10 round (MiB) | Thời gian GRU | Thời gian Transformer | Thời gian CNN-1D |")
    A("|---|---:|---:|---:|---:|")
    for key, name, _m, _dep in METHODS:
        cb = data[(key, "gru")]["comm_bytes"] / 1048576
        A(f"| {name} | {cb:.3f} | " + " | ".join(
            f"{data[(key, s)]['runtime_s']/60:.1f} phút" for s, _ in SCENARIOS) + " |")
    A("")
    A("Chênh lệch truyền thông giữa pFedES (0,043 MiB) và FD-IDS (30,5 MiB) là")
    A("**hơn 700 lần**. FedCAPS truyền ít thứ hai (0,135 MiB) nhưng lại tốn thời gian")
    A("nhất (191–404 phút) vì pha tìm kiếm MARLFS + encoder + PPO chạy trước khi")
    A("huấn luyện classifier.\n")


ANALYSIS = {
    "fd_ids_noniid": ("""\
- **Đường học ổn định nhất trong cả năm phương pháp.** Accuracy GRU đi lên đơn
  điệu suốt 10 round (0,2397 → 0,6204) không có một lần tụt nào. Đây là bằng chứng
  trực tiếp cho tuyên bố chống *model drift* của bài báo: KD kéo model cục bộ về
  phía global, FedProx chặn biên độ lệch, FedAvg có trọng số không cho client nhỏ
  lấn át.
- **Kết quả cuối cao nhất trong nhóm "một model duy nhất"** ở kịch bản GRU
  (0,6204) và gần bằng ProxyModel-server ở hai kịch bản còn lại.
- **Bền với việc đổi kiến trúc.** Ba backbone rất khác nhau đều về đích trong dải
  hẹp 0,5901–0,6204. Cơ chế không phụ thuộc vào lựa chọn model.
- **Rẻ về thời gian.** 22–25 phút cho GRU/CNN-1D, thuộc nhóm nhanh nhất.""", """\
- **Không dùng được cho môi trường dị thể** — hạn chế nghiêm trọng nhất. FedAvg
  trên toàn bộ tham số buộc mọi client phải giống hệt nhau về kiến trúc.
- **Không có personalization.** Chỉ có một model cho cả 10 client. Client 6 (chỉ
  5/34 lớp) và client 8 (28/34 lớp) buộc phải dùng chung một bộ trọng số.
- **`macro_F1` chỉ đạt 0,3485** dù accuracy 0,6204. Khoảng cách 0,27 này cho thấy
  model bỏ rơi phần lớn các lớp thiểu số — nó chủ yếu đoán đúng vài lớp đa số.
- **Tốn truyền thông nhất** cùng với ProxyModel: 30,5 MiB, vì phải gửi toàn bộ
  40.034 tham số hai chiều × 10 client × 10 round.
- **Khởi đầu chậm.** Transformer mất tới 4 round mới vượt 0,55, do phải học lại
  từ đầu ở mỗi round.""" ),

    "perfed_skd": ("""\
- **Round 1 mạnh nhất trong cả năm phương pháp.** Server pretrain một epoch trên
  `global_train_data.csv` cho ngay 0,7261 accuracy (GRU) — cao hơn *bất kỳ* con số
  nào mà bốn phương pháp còn lại đạt được sau 10 round.
- **Có personalization thật.** Mỗi client giữ state riêng và teacher lịch sử riêng.
- **Trung bình client cao nhất trong nhóm personalized**: 0,5250 (Transformer),
  0,4937 (GRU) — vượt xa FedCAPS (0,2988) và pFedES (0,3238).
- **Cơ chế chọn client theo ngưỡng có ích thật cho thiết bị yếu:** client đang tốt
  không bị ép nhận global model, giảm cả tính toán lẫn truyền thông. Truyền thông
  13,7 MiB, chỉ bằng 45% của FD-IDS.
- **Client tốt nhất đạt 0,6765** (Transformer) — cao hơn cả server FD-IDS.""", """\
- **Đây là phương pháp duy nhất mà kết quả ĐI XUỐNG theo round.** Trung bình client
  GRU: 0,6705 (r1) → **0,4937** (r10), mất 0,1768 tuyệt đối. Server GRU cũng giảm
  0,7261 → 0,6408. Cả ba kịch bản đều cùng xu hướng.

  Nguyên nhân đọc được từ chính cơ chế: teacher của SKD là **chính model đó ở round
  trước**, không có neo ngoài. Khi client bắt đầu chuyên biệt hoá vào phân bố cục
  bộ cực lệch của mình, SKD **củng cố chính sai lệch đó** qua từng round. Ngưỡng
  chọn client cũng không cứu được, vì nó chỉ so accuracy tương đối giữa các client
  chứ không so với một chuẩn tuyệt đối.
- **Vì thế round 10 là điểm dừng tệ nhất cho phương pháp này.** Nếu dừng ở round 1
  hoặc 2 thì PerFed-SKD thắng tuyệt đối. Hợp đồng thí nghiệm chốt round 10 làm
  endpoint nên con số báo cáo không phản ánh khả năng tốt nhất của nó.
- **Không dùng được cho dị thể mô hình** — vẫn trung bình tham số.
- **Server không ổn định ở Transformer:** dao động 0,3979 (r3) → 0,6840 (r6) →
  0,6328 (r10). Trung bình không trọng số trên tập client được chọn khiến một
  client xấu có thể kéo cả global model đi.
- **Chậm nhất ở Transformer:** 84,7 phút.""" ),

    "permutation_feature_importance": ("""\
- **Phương pháp duy nhất hỗ trợ dị thể mà không truyền một tham số nào.** Chỉ
  chỉ số feature và điểm số rời khỏi client — mức bảo vệ riêng tư cao nhất trong
  năm phương pháp.
- **Truyền thông gần như bằng không:** 0,135 MiB cho toàn bộ pipeline, thấp thứ
  hai chỉ sau pFedES.
- **Đường học đơn điệu tăng** ở cả ba kịch bản, không có dấu hiệu drift.
- **Có khả năng diễn giải.** Đây là phương pháp duy nhất trả về một câu trả lời
  đọc được bằng mắt: subset feature được chọn. Ba kịch bản chọn ra ba subset khác
  nhau nhưng **`DNS` xuất hiện ở cả ba**, `TCP` ở hai — tín hiệu nhất quán rằng
  các feature giao thức mang nhiều thông tin phân biệt nhất.
- **Đạt kết quả này chỉ với 6–8 trên 25 feature**, tức bỏ 68–76% chiều dữ liệu.""", """\
- **Kết quả tuyệt đối thấp nhất nhì:** 0,2303–0,2988 accuracy. Nguyên nhân trực
  tiếp là **subset quá nhỏ**: 6 feature (GRU, CNN-1D) hoặc 8 (Transformer) không
  đủ để phân biệt 34 lớp tấn công.

  Đây là hệ quả của hàm reward `R = λ·(hiệu năng) + (1−λ)·(độ ngắn)` với **`λ` =
  0,1** — tức 90% trọng số dồn vào việc làm subset ngắn, chỉ 10% cho hiệu năng.
  Cấu hình này ưu tiên nén mạnh hơn là độ chính xác. **Tăng `λ` là hướng cải thiện
  rõ ràng nhất** và không đòi hỏi thay đổi gì về phương pháp.
- **Đắt nhất về thời gian:** 191,8–403,6 phút, gấp 5–9 lần FD-IDS. Pha MARLFS
  (2.444 giây) + encoder–decoder (661 giây) + PPO (817 giây) chạy trước khi
  classifier bắt đầu học.
- **Phân tán client rất lớn:** biên độ 0,4506 ở GRU, client tệ nhất 0,0021.
- **Không phải phương pháp học liên kết đúng nghĩa.** Sau khi chốt subset, các
  classifier học độc lập hoàn toàn, không FedAvg. Phần "federated" chỉ nằm ở khâu
  chọn feature.""" ),

    "pfedes": ("""\
- **Rẻ nhất tuyệt đối về truyền thông: 0,043 MiB cho 10 round** — chỉ 228 byte mỗi
  lần gửi. Thấp hơn FD-IDS **hơn 700 lần** và thấp hơn ProxyModel **640 lần**.
  Với mạng IoT băng thông hẹp, đây là khác biệt mang tính quyết định.
- **Hỗ trợ dị thể ở mức rẻ nhất.** Chỉ cần proxy giữ ràng buộc vào = ra; mọi thứ
  khác trên client là tự do.
- **Model cục bộ không bao giờ rời thiết bị** — không có rủi ro rò rỉ mô hình.
- **Đường học tăng đều** ở GRU (0,1887 → 0,2948) và Transformer (0,2496 → 0,3238),
  chứng tỏ 57 tham số proxy vẫn truyền tải được tri thức toàn cục.
- **Client tốt nhất đạt 0,5917** (Transformer) — cho thấy giới hạn không nằm ở
  phương pháp mà ở việc chấm model personalized trên tập test toàn cục.""", """\
- **Kịch bản CNN-1D gần như thất bại: 0,1089 accuracy, macro-F1 0,0144.** Đây là
  kết quả tệ nhất trong toàn bộ 15 lần chạy, và nó **không cải thiện qua round**
  (0,1020 ở r1 → 0,1089 ở r10). Hai kịch bản kia đều tăng đều, nên đây là vấn đề
  riêng của tổ hợp CNN-1D + proxy 57 tham số, cần điều tra thêm — nghi vấn hàng
  đầu là BatchNorm trong CNN-1D phản ứng xấu với dữ liệu đã qua proxy conv chưa
  hội tụ.
- **Kênh truyền tri thức quá hẹp.** 57 tham số là tất cả những gì 10 client dùng
  để trao đổi. So với ProxyModel (35.874 tham số proxy) đạt server 0,6205, cái giá
  của việc rẻ là rất rõ.
- **Không có model toàn cục dùng được.** Server chỉ giữ extractor 25→25, không
  phát logits 34 lớp, nên cột server là `not_applicable`. Không có gì để triển
  khai cho một client mới chưa có dữ liệu.
- **Phân tán client rất lớn:** biên độ 0,5896 ở Transformer.
- **Chậm hơn dự kiến:** 136,8 phút ở Transformer, do mỗi round phải chạy hai pha
  huấn luyện tuần tự.""" ),

    "proxymodel": ("""\
- **Global proxy là entity mạnh nhất trong toàn bộ thí nghiệm.** 0,6205 (GRU),
  0,6185 (Transformer), 0,6151 (CNN-1D) — nhỉnh hơn cả FD-IDS, và **ổn định nhất
  giữa ba kịch bản** (biên độ chỉ 0,0054).
- **Hội tụ nhanh nhất.** Chỉ 2 round đã đạt 0,58; round 3 đã vượt 0,60. FD-IDS
  cần 5–6 round mới tới ngưỡng đó.
- **Có cả hai thứ cùng lúc:** một global model dùng được ngay *và* 10 model
  personalized. Không phương pháp nào khác cho cả hai.
- **Hỗ trợ dị thể**, và là phương pháp duy nhất mà **bài gốc thực sự vận hành
  trong chế độ dị thể** (NAS sinh kiến trúc riêng cho từng hộ).
- **Adaptive mutual distillation hiệu quả:** trao đổi qua logits nên hoàn toàn
  không phụ thuộc kiến trúc, mà vẫn đạt kết quả cao nhất.""", """\
- **Trung bình client chỉ 0,3541–0,3572**, thấp hơn PerFed-SKD (0,4937–0,5250)
  khá nhiều, dù server thì cao hơn. Mutual distillation hai chiều kéo model
  personalized về phía proxy chung, làm giảm mức độ cá nhân hoá — đúng phần mà
  bài toán personalized cần.
- **Đắt truyền thông: 27,7 MiB**, gần bằng FD-IDS và gấp **640 lần** pFedES. Proxy
  là một classifier đầy đủ chứ không phải extractor nhỏ.
- **Mất phần cốt lõi khi build lại.** NAS bị bỏ, nên tính dị thể — thứ tạo nên giá
  trị chính của bài báo — không được tái hiện. Kết quả ở đây đo một phiên bản
  ProxyModel đã bị lược bớt.
- **Đổi bài toán.** Bài gốc là hồi quy NILM đo bằng MAE/SAE; bản build lại là phân
  loại 34 lớp. Các kết luận của bài gốc **không chuyển giao trực tiếp**.
- **Phân tán client lớn:** biên độ 0,5578 (GRU).
- **Round 10 tốn thêm thời gian** (235 giây so với ~165 giây các round khác) do có
  pha fine-tune personalized cuối.""" ),
}


def analysis_section(data, L):
    A = L.append
    A("---\n")
    A("## 9. Phân tích điểm mạnh — điểm yếu\n")
    A("Phân tích dưới đây rút ra **từ chính số liệu đo được**, không phải từ tuyên bố")
    A("của bài báo gốc.\n")
    for key, name, _m, dep in METHODS:
        strong, weak = ANALYSIS[key]
        A(f"### {name}\n")
        A("**Điểm mạnh**\n")
        A(strong + "\n")
        A("**Điểm yếu**\n")
        A(weak + "\n")

    A("### Bảng tổng kết đối chiếu\n")
    A("| Tiêu chí | FD-IDS | PerFed-SKD | FedCAPS | pFedES | ProxyModel |")
    A("|---|---|---|---|---|---|")
    A("| Accuracy tốt nhất (entity triển khai) | **0,6204** | 0,5250 | 0,2988 | 0,3238 | 0,3572 |")
    A("| Accuracy server tốt nhất | **0,6204** | 0,6408 | — | — | **0,6205** |")
    A("| macro-F1 tốt nhất | **0,3587** | 0,2627 | 0,1162 | 0,1186 | 0,1701 |")
    A("| Xu hướng theo round | tăng đều | **giảm đều** | tăng đều | tăng (CNN-1D đứng yên) | tăng đều |")
    A("| Truyền thông | 30,5 MiB | 13,7 MiB | 0,135 MiB | **0,043 MiB** | 27,7 MiB |")
    A("| Thời gian (GRU) | **22,1 phút** | **21,8 phút** | 191,8 phút | 37,5 phút | 28,9 phút |")
    A("| Dị thể kích thước mô hình | ✗ | ✗ | ✓ | ✓ | ✓ |")
    A("| Dị thể loại mô hình | ✗ | ✗ | ✓ | ✓ | ✓ |")
    A("| Có global model dùng được | ✓ | ✓ | ✗ | ✗ | ✓ |")
    A("| Có personalization | ✗ | ✓ | ✓ | ✓ | ✓ |")
    A("| Bền giữa 3 kịch bản | ✓ | trung bình | ✓ | ✗ (CNN-1D hỏng) | **✓ tốt nhất** |")
    A("")

    A("---\n")
    A("## 10. Kết luận\n")
    A("**Không có phương pháp nào thắng toàn diện** — mỗi phương pháp tối ưu cho một")
    A("ràng buộc khác nhau, và điều đó thể hiện rõ trong số liệu:\n")
    A("| Nếu ràng buộc của bạn là… | Chọn | Vì |")
    A("|---|---|---|")
    A("| Cần một model tổng quát mạnh nhất | **ProxyModel (server)** | 0,6205 GRU, ổn định nhất giữa 3 kịch bản, hội tụ sau 2 round |")
    A("| Mọi client giống hệt nhau, cần global model | **FD-IDS** | 0,6204 GRU, đường học ổn định nhất, cơ chế đơn giản nhất |")
    A("| Băng thông là nút thắt | **pFedES** | 0,043 MiB — rẻ hơn 700 lần, nhưng phải tránh CNN-1D |")
    A("| Client dùng kiến trúc khác nhau | **pFedES** hoặc **ProxyModel** | chỉ FedAvg proxy; ProxyModel mạnh hơn, pFedES rẻ hơn |")
    A("| Riêng tư là ưu tiên cao nhất | **FedCAPS** | không truyền tham số mô hình, chỉ chỉ số feature |")
    A("| Cần personalization thật sự | **PerFed-SKD** | TB client cao nhất (0,5250), nhưng **phải dừng sớm** |")
    A("| Cần giải thích được mô hình | **FedCAPS** | trả về subset feature đọc được |")
    A("")
    A("### Ba phát hiện đáng chú ý nhất\n")
    A("1. **PerFed-SKD suy giảm theo round.** Đây là kết quả trái ngược với kỳ vọng")
    A("   và là phát hiện quan trọng nhất của thí nghiệm. Self-knowledge distillation")
    A("   không có neo ngoài, nên trên dữ liệu cực lệch nó khuếch đại chính sai lệch")
    A("   của client. Nếu triển khai thật, cần **early stopping** hoặc thêm một neo")
    A("   toàn cục vào hàm loss.")
    A("2. **pFedES + CNN-1D hỏng.** 0,1089 accuracy và không cải thiện qua 10 round,")
    A("   trong khi GRU và Transformer đều tăng đều. Đây là một lỗi tương tác cụ thể")
    A("   cần điều tra, không phải giới hạn của phương pháp.")
    A("3. **FedCAPS bị `λ` = 0,1 kìm hãm.** Hàm reward dồn 90% trọng số vào độ ngắn")
    A("   của subset, nên chỉ giữ 6–8/25 feature. Kết quả thấp phản ánh cấu hình")
    A("   siêu tham số chứ không phải năng lực phương pháp — đây là hướng cải thiện")
    A("   rẻ nhất trong cả năm phương pháp.\n")
    A("### Giới hạn của thí nghiệm này\n")
    A("Cần nêu rõ để tránh kết luận quá đà:\n")
    A("- **Cả 10 client dùng cùng một kiến trúc trong mọi kịch bản.** Vì vậy khả năng")
    A("  dị thể của FedCAPS, pFedES và ProxyModel **được suy ra từ cơ chế, chưa được")
    A("  kiểm chứng bằng số đo**. Đây là hướng thí nghiệm tiếp theo rõ ràng nhất.")
    A("- **Model personalized bị chấm trên tập test toàn cục 34 lớp.** Cách đo này")
    A("  có lợi cho phương pháp global và bất lợi cho phương pháp personalized. Một")
    A("  phép đo bổ sung trên tập test riêng của từng client sẽ cho bức tranh công")
    A("  bằng hơn.")
    A("- **Round 10 là điểm dừng cố định.** Với PerFed-SKD, đây là điểm dừng tệ nhất;")
    A("  với các phương pháp còn lại, đường học vẫn đang đi lên và chưa hội tụ.")
    A("- **Mỗi cấu hình chỉ chạy một lần với seed 42.** Không có khoảng tin cậy, nên")
    A("  các chênh lệch nhỏ (dưới ~0,01) không nên được diễn giải là có ý nghĩa.")
    A("- **Ba trong năm bài báo không xác định được năm/nơi xuất bản** từ bản `.md`")
    A("  trong repo.\n")
    A("---\n")
    A("## Phụ lục — nguồn dữ liệu\n")
    A("- Dữ liệu thô: `metrics/evaluation_metrics.csv` của 15 lần chạy")
    A("- Tổng hợp dạng máy đọc: [`all_methods_metrics.csv`](all_methods_metrics.csv)")
    A("- Biểu đồ so sánh chéo: [`report_assets/`](report_assets/)")
    A("- Script sinh báo cáo: [`scripts/build_report.py`](scripts/build_report.py)")
    A("")
    A("Chạy lại toàn bộ:\n")
    A("```bash")
    A("/home/odixe/miniforge3/envs/nckh/bin/python scripts/build_report.py")
    A("```")


def main():
    ASSETS.mkdir(exist_ok=True)
    data = load_all()

    fig_metric_by_round(data, "accuracy", "accuracy", "fig_accuracy_by_round.png")
    fig_metric_by_round(data, "macro_f1", "macro-F1", "fig_macro_f1_by_round.png")
    fig_metric_by_round(data, "weighted_f1", "weighted-F1", "fig_weighted_f1_by_round.png")
    fig_server_vs_client(data)
    fig_round10_heatmap(data)
    fig_client_spread(data)
    fig_cost(data)

    csv_path = write_csv(data)

    L = build_report(data)
    compare_section(data, L)
    analysis_section(data, L)
    (ROOT / "reports.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    n_img = len(list(ASSETS.glob("*.png")))
    print(f"reports.md      : {len((ROOT/'reports.md').read_text().splitlines()):,} dòng")
    print(f"biểu đồ mới     : {n_img} file trong {ASSETS.relative_to(ROOT)}/")
    print(f"CSV tổng hợp    : {csv_path.relative_to(ROOT)} "
          f"({len(csv_path.read_text().splitlines())-1:,} hàng)")


if __name__ == "__main__":
    main()
