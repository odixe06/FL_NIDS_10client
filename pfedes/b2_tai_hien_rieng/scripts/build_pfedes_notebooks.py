#!/usr/bin/env python3
"""Generate the three self-contained pFedES Kaggle notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "pfedes_client_parallel_runtime.py"


SCENARIOS = [
    {
        "folder": "01_10_clients_gru",
        "filename": "pfedes_10_clients_gru.ipynb",
        "title": "pFedES — 10 client GRU",
        "run_name": "fd_ids_ciciot2023_10c_gru",
        "scenario_model_family": "10_gru",
        "families": ["gru"] * 10,
    },
    {
        "folder": "02_10_clients_transformer",
        "filename": "pfedes_10_clients_transformer.ipynb",
        "title": "pFedES — 10 client Transformer",
        "run_name": "fd_ids_ciciot2023_10c_transformer",
        "scenario_model_family": "10_transformer",
        "families": ["transformer"] * 10,
    },
    {
        "folder": "03_5_gru_5_transformer",
        "filename": "pfedes_5_gru_5_transformer.ipynb",
        "title": "pFedES — 5 client GRU + 5 client Transformer",
        "run_name": "fd_ids_ciciot2023_5gru_5transformer",
        "scenario_model_family": "5_gru_5_transformer",
        "families": ["gru"] * 5 + ["transformer"] * 5,
    },
]


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def method_markdown(scenario):
    assignment = (
        "Cả 10 client dùng GRU."
        if scenario["scenario_model_family"] == "10_gru"
        else "Cả 10 client dùng Transformer."
        if scenario["scenario_model_family"] == "10_transformer"
        else "Client 1–5 dùng GRU; client 6–10 dùng Transformer."
    )
    return rf"""# {scenario['title']}

Notebook này triển khai **pFedES (generalized proxy feature extractor sharing)** trên CICIoT2023 đã được tiền xử lý thành 25 đặc trưng và 34 lớp. {assignment}

Notebook không sao chép các bảng kết quả MNIST/CIFAR của bài báo. Phần dưới giữ nguyên nội dung toán học cần thiết để xác định phương pháp, rồi ghi rõ phép chuyển đổi từ ảnh sang vector đặc trưng mạng.

## 1. Bài toán MHPFL và mục tiêu

Với client $k$, dữ liệu cục bộ là $D_k$, local model cá nhân hóa là $\mathcal{{F}}_k(\omega_k)$, còn proxy feature extractor đồng nhất được chia sẻ là $\mathcal{{G}}(\theta)$. Mục tiêu FedAvg đồng nhất được nhắc lại bởi:

$$
\min_{{\omega\in\mathbb{{R}}^d}}\sum_{{k=0}}^{{N-1}}\frac{{n_k}}{{n}}\mathcal{{L}}_k(\mathcal{{F}}(\omega);D_k). \tag{{1}}
$$

Trong MHPFL, mỗi client có kiến trúc và số chiều tham số riêng:

$$
\min_{{\omega_0,\ldots,\omega_{{N-1}}}}\sum_{{k=0}}^{{N-1}}\mathcal{{L}}_k(\mathcal{{F}}_k(\omega_k);D_k). \tag{{2}}
$$

pFedES thêm proxy extractor phía trước local model:

$$
\min_{{\theta,\omega_0,\ldots,\omega_{{N-1}}}}\sum_{{k=0}}^{{N-1}}
\mathcal{{L}}_k(\{{\mathcal{{G}}(\theta),\mathcal{{F}}_k(\omega_k)\}};D_k). \tag{{3}}
$$

## 2. Huấn luyện luân phiên pFedES

Ở đầu round $t$, cả 10 client nhận cùng global proxy $\mathcal{{G}}(\theta^{{t-1}})$. Local model không được gửi lên server và giữ trạng thái cá nhân hóa qua các round.

### Bước 1 — đóng băng proxy, cập nhật local model

Với $\hat{{x}}=\mathcal{{G}}(\theta^{{t-1}};x)$ và $\hat{{x}}$ cùng chiều với $x$:

$$
\hat{{y}}_1=\mathcal{{F}}_k(\omega_k^{{t-1}};\hat{{x}}),\qquad
\hat{{y}}_2=\mathcal{{F}}_k(\omega_k^{{t-1}};x). \tag{{4}}
$$

$$
\ell_1=\ell(\hat{{y}}_1,y),\qquad \ell_2=\ell(\hat{{y}}_2,y). \tag{{5}}
$$

$$
\ell_\omega=\mu\ell_1+(1-\mu)\ell_2,\qquad \mu\in(0,0.5]. \tag{{6}}
$$

$$
\omega_k^t\leftarrow\omega_k^{{t-1}}-\eta_\omega\nabla\ell_\omega. \tag{{7}}
$$

Notebook dùng cross-entropy, $\mu=0.1$ và $\eta_\omega=0.01$.

### Bước 2 — đóng băng local model, cập nhật proxy

Local model vừa cập nhật vẫn ở chế độ huấn luyện nhưng toàn bộ tham số được đóng băng bằng `requires_grad=False`; gradient vẫn truyền qua local model về input của nó để cập nhật proxy. Với GRU chạy bằng cuDNN, forward phải ở chế độ huấn luyện thì backward theo đường input này mới hợp lệ. Trạng thái dropout được tái lập từ seed `(seed, round, client_id, phase)` và mỗi worker dùng một stream để giữ tính tái lập:

$$
\hat{{y}}=\mathcal{{F}}_k(\omega_k^t;\mathcal{{G}}(\theta^{{t-1}};x)). \tag{{8}}
$$

$$
\ell_\theta=\ell(\hat{{y}},y). \tag{{9}}
$$

$$
\theta_k^t\leftarrow\theta^{{t-1}}-\eta_\theta\nabla\ell_\theta. \tag{{10}}
$$

Notebook dùng một epoch proxy và $\eta_\theta=0.01$.

### Bước 3 — tổng hợp proxy có trọng số

Chỉ proxy được tính là logical communication. Với $n=\sum_{{k\in\mathcal{{S}}^t}}n_k$:

$$
\theta^t=\sum_{{k\in\mathcal{{S}}^t}}\frac{{n_k}}{{n}}\theta_k^t. \tag{{11}}
$$

Ở đây $N=10$, $C=100\%$ nên $\mathcal{{S}}^t$ luôn gồm đủ 10 client.

## 3. Các giả định và kết quả hội tụ

Với hằng số trơn $L_1$:

$$
\|\nabla\mathcal{{L}}_k^{{t_1}}(\omega_k^t;x,y)-\nabla\mathcal{{L}}_k^{{t_2}}(\omega_k^t;x,y)\|
\le L_1\|\omega_k^{{t_1}}-\omega_k^{{t_2}}\|. \tag{{12}}
$$

$$
\mathcal{{L}}_k^{{t_1}}-\mathcal{{L}}_k^{{t_2}}
\le\langle\nabla\mathcal{{L}}_k^{{t_2}},\omega_k^{{t_1}}-\omega_k^{{t_2}}\rangle
+\frac{{L_1}}{{2}}\|\omega_k^{{t_1}}-\omega_k^{{t_2}}\|_2^2. \tag{{13}}
$$

Gradient ngẫu nhiên của local model và mô hình ghép là không chệch:

$$
\mathbb{{E}}[g_{{\omega,k}}^t]=\nabla\mathcal{{L}}_k^t(\omega_k^t),\qquad
\mathbb{{E}}[g_{{\phi,k}}^t]=\nabla\mathcal{{L}}_k^t(\phi_k^t). \tag{{14}}
$$

Phương sai được chặn bởi $\sigma^2$ và $\delta^2$:

$$
\mathbb{{E}}\|\nabla\mathcal{{L}}_k^t(\omega_k^t;\mathcal{{B}})-\nabla\mathcal{{L}}_k^t(\omega_k^t)\|_2^2\le\sigma^2,
\quad
\mathbb{{E}}\|\nabla\mathcal{{L}}_k^t(\phi_k^t;\mathcal{{B}})-\nabla\mathcal{{L}}_k^t(\phi_k^t)\|_2^2\le\delta^2. \tag{{15}}
$$

Đặt $\tilde{{\mu}}=1-\mu$, loss sau $E$ local iteration thỏa:

$$
\mathbb{{E}}[\mathcal{{L}}_{{(t+1)E}}]\le
\mathcal{{L}}_{{tE+0}}+
\left(\frac{{L_1\eta^2\tilde{{\mu}}^2}}{{2}}-\eta\tilde{{\mu}}\right)
\sum_{{e=0}}^{{E-1}}\|\nabla\mathcal{{L}}_{{tE+e}}\|_2^2
+\frac{{L_1\eta^2(\sigma^2+\delta^2)}}{{2}}. \tag{{16}}
$$

Do đó, với learning rate thỏa điều kiện dưới đây, pFedES có tốc độ hội tụ non-convex $\mathcal{{O}}(1/T)$:

$$
\frac{{1}}{{T}}\sum_{{t=0}}^{{T-1}}\sum_{{e=0}}^{{E-1}}\|\nabla\mathcal{{L}}_{{tE+e}}\|_2^2\le\epsilon,
\qquad
\eta<\frac{{2\epsilon\tilde{{\mu}}}}{{L_1(\sigma^2+\delta^2+\tilde{{\mu}}^2\epsilon)}}. \tag{{17}}
$$

## 4. Chuyển đổi sang CICIoT2023

- Proxy ảnh `Conv 3→8→3` được chuyển tương đương thành `Conv1d 1→8→1`, kernel 3, same padding; ReLU chỉ đặt giữa hai convolution. Input và enhanced data đều là `[B,25]`.
- GRU: chuỗi 25 bước, GRU hai tầng hidden 64, dropout 0,2, linear `64→34`; đúng **40.034** tham số.
- Transformer: scalar projection 64, learned position embedding, hai encoder layer, 4 heads, FFN 128, mean pooling, LayerNorm và linear `64→34`; đúng **71.010** tham số.
- Mỗi client split phân tầng 95/5; singleton class ở train. Global test chỉ được dùng sau khi chọn `best.pt`.
- Report chính là pooled personalized predictions của 10 model trên cùng global test; notebook cũng lưu metric riêng từng client.
"""


def build_config(scenario):
    run_name = scenario["run_name"]
    return {
        "run_name": run_name,
        "method": "pFedES",
        "scenario_model_family": scenario["scenario_model_family"],
        "execution_mode": "client_parallel",
        "input_dir": "/kaggle/input/datasets/odixe0502/data-for-10clients",
        "output_dir": f"/kaggle/working/{run_name}",
        "temp_dir": f"/kaggle/temp/{run_name}",
        "cache_dir": f"/kaggle/temp/{run_name}_cache",
        "payload_dir": f"/kaggle/temp/{run_name}_payloads",
        "num_clients": 10,
        "num_classes": 34,
        "num_features": 25,
        "client_model_families": scenario["families"],
        "communication_rounds": 10,
        "local_epochs": 1,
        "proxy_epochs": 1,
        "per_client_batch_size": 1024,
        "gradient_accumulation_steps": 1,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "scheduler": None,
        "label_loss": "cross_entropy",
        "mu": 0.1,
        "client_participation_fraction": 1.0,
        "seed": 42,
        "initialization_seed": 42,
        "split_seed": 42,
        "mixed_precision": True,
        "gpu_cache_fraction": 0.7,
        "stream_candidates": [1, 2],
        "stochastic_models_force_single_stream": True,
        "csv_chunk_rows": 250000,
        "gpu_copy_chunk_rows": 250000,
        "worker_count": 2,
        "worker_timeout_seconds": 86400,
        "resume_if_available": True,
        "resume_checkpoint_path": None,
        "checkpoint_criterion": "mean_personalized_validation_macro_f1",
        "test_semantics": "ten personalized models each evaluate the full global test",
    }


def notebook_cells(scenario, runtime_source):
    config = build_config(scenario)
    config_json = json.dumps(config, indent=2, ensure_ascii=False)
    environment_cell = '''import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch

NOTEBOOK_STARTED = time.perf_counter()
n_gpu = torch.cuda.device_count()
assert n_gpu == 2, f"Kaggle accelerator must be GPU T4 x2; found {n_gpu} GPUs"
GPU_NAMES = [torch.cuda.get_device_name(index) for index in range(n_gpu)]
assert all("T4" in name for name in GPU_NAMES), f"Expected two NVIDIA T4 GPUs, found {GPU_NAMES}"
print({
    "python": platform.python_version(),
    "pytorch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": GPU_NAMES,
})
'''
    config_cell = f'''CONFIG = json.loads(r\'''{config_json}\''')
OUTPUT_DIR = Path(CONFIG["output_dir"])
TEMP_DIR = Path(CONFIG["temp_dir"])
CLIENT_PARALLEL_SCRIPT_PATH = Path("/kaggle/temp") / f"{{CONFIG['run_name']}}_client_parallel.py"
MANIFEST_PATH = Path("/kaggle/temp") / f"{{CONFIG['run_name']}}_manifest.json"
for directory in [OUTPUT_DIR / "checkpoints", OUTPUT_DIR / "logs", OUTPUT_DIR / "metrics", OUTPUT_DIR / "artifacts", TEMP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

NOTEBOOK_LOGGER = logging.getLogger("pfedes_notebook")
NOTEBOOK_LOGGER.setLevel(logging.INFO)
NOTEBOOK_LOGGER.handlers.clear()
handler = logging.FileHandler(OUTPUT_DIR / "logs" / "run.log", mode="a", encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
NOTEBOOK_LOGGER.addHandler(handler)
NOTEBOOK_LOGGER.info("Notebook initialized for %s", CONFIG["run_name"])
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
'''
    validate_cell = '''FEATURE_COLUMNS = [
    "ack_flag_number", "AVG", "Std", "UDP", "fin_count", "Max", "TCP",
    "syn_count", "Protocol Type", "Rate", "IAT", "syn_flag_number",
    "rst_flag_number", "Tot sum", "HTTPS", "ack_count", "fin_flag_number",
    "HTTP", "rst_count", "Header_Length", "psh_flag_number", "ICMP",
    "Time_To_Live", "ARP", "DNS",
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["Label"]
INPUT_DIR = Path(CONFIG["input_dir"])
REQUIRED_INPUTS = [
    *[INPUT_DIR / f"client_{client_id}_train.csv" for client_id in range(1, 11)],
    INPUT_DIR / "global_train_data.csv",
    INPUT_DIR / "global_test_data.csv",
    INPUT_DIR / "label_mapping.csv",
]
missing = [str(path) for path in REQUIRED_INPUTS if not path.is_file()]
assert not missing, f"Missing Kaggle dataset files: {missing}"
for csv_path in REQUIRED_INPUTS[:-1]:
    assert pd.read_csv(csv_path, nrows=0).columns.tolist() == EXPECTED_COLUMNS, csv_path
mapping = pd.read_csv(REQUIRED_INPUTS[-1])
assert mapping.columns.tolist() == ["Encoded_ID", "Label_Name"]
assert mapping["Encoded_ID"].tolist() == list(range(34))
assert mapping.loc[1, "Label_Name"] == "BENIGN"
NOTEBOOK_LOGGER.info("Validated paths and CSV headers for all required inputs")
print("Input contract validated; the subprocess performs chunked full-value validation.")
'''
    runtime_cell = "CLIENT_PARALLEL_SCRIPT_SOURCE = " + repr(runtime_source) + "\n"
    write_cell = '''CLIENT_PARALLEL_SCRIPT_PATH.write_text(CLIENT_PARALLEL_SCRIPT_SOURCE, encoding="utf-8")
MANIFEST_PATH.write_text(
    json.dumps({"config": CONFIG}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
assert CLIENT_PARALLEL_SCRIPT_PATH.stat().st_size > 0
assert MANIFEST_PATH.stat().st_size > 0
print(CLIENT_PARALLEL_SCRIPT_PATH)
print(MANIFEST_PATH)
'''
    launch_cell = '''for logger_handler in list(NOTEBOOK_LOGGER.handlers):
    logger_handler.flush()
    logger_handler.close()
    NOTEBOOK_LOGGER.removeHandler(logger_handler)

launch_environment = os.environ.copy()
launch_environment.update({
    "TRAINING_MANIFEST": str(MANIFEST_PATH),
    "PYTHONUNBUFFERED": "1",
    "MPLBACKEND": "Agg",
})
subprocess_started = time.perf_counter()
command = [sys.executable, str(CLIENT_PARALLEL_SCRIPT_PATH)]
subprocess.run(command, check=True, env=launch_environment)
subprocess_wall_seconds = time.perf_counter() - subprocess_started

runtime_path = OUTPUT_DIR / "metrics" / "runtime_breakdown.json"
runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
runtime_payload["subprocess_wall_seconds"] = subprocess_wall_seconds
runtime_payload["total_notebook_pipeline_seconds"] = time.perf_counter() - NOTEBOOK_STARTED
runtime_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
summary_path = OUTPUT_DIR / "metrics" / "summary.json"
summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
summary_payload["runtime"] = runtime_payload
summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Training subprocess completed in {subprocess_wall_seconds:,.1f} seconds")
'''
    verify_cell = '''logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(OUTPUT_DIR / "logs" / "run.log", mode="a", encoding="utf-8"), logging.StreamHandler()],
    force=True,
)
REQUIRED_OUTPUTS = [
    "checkpoints/best.pt", "checkpoints/last.pt", "logs/run.log", "logs/rank_1.log",
    "metrics/config.json", "metrics/dataset_summary.json", "metrics/client_class_distribution.csv",
    "metrics/history_pretrain.csv", "metrics/history_pretrain.json",
    "metrics/history_round.csv", "metrics/history_round.json",
    "metrics/history_client.csv", "metrics/history_client.json",
    "metrics/history_local_epoch.csv", "metrics/history_local_epoch.json",
    "metrics/summary.json", "metrics/classification_report.json", "metrics/classification_report.csv",
    "metrics/confusion_matrix.csv", "metrics/confusion_matrix.npy",
    "metrics/personalized_test_metrics.json", "metrics/personalized_test_metrics.csv",
    "metrics/communication_costs.json", "metrics/communication_costs.csv",
    "metrics/runtime_breakdown.json", "artifacts/class_distribution.png",
    "artifacts/accuracy_f1_curves.png", "artifacts/loss_curves.png",
    "artifacts/confusion_matrix.png", "artifacts/per_class_f1.png",
    "artifacts/runtime_per_round.png", "artifacts/communication_cumulative.png",
]
for relative_path in REQUIRED_OUTPUTS:
    path = OUTPUT_DIR / relative_path
    assert path.is_file() and path.stat().st_size > 0, path
assert len(pd.read_csv(OUTPUT_DIR / "metrics/history_pretrain.csv")) == 1
assert len(pd.read_csv(OUTPUT_DIR / "metrics/history_round.csv")) == CONFIG["communication_rounds"]
assert len(pd.read_csv(OUTPUT_DIR / "metrics/history_client.csv")) == CONFIG["communication_rounds"] * 10
assert len(pd.read_csv(OUTPUT_DIR / "metrics/history_local_epoch.csv")) == CONFIG["communication_rounds"] * 10
summary = json.loads((OUTPUT_DIR / "metrics/summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "complete"
logging.info("Verified all required nonempty outputs and exact history row counts")
display(pd.DataFrame([summary["final_mean_personalized_test_metrics"]]))
'''
    display_cell = '''from IPython.display import Image, display

for plot_name in [
    "accuracy_f1_curves.png",
    "loss_curves.png",
    "confusion_matrix.png",
    "per_class_f1.png",
    "runtime_per_round.png",
    "communication_cumulative.png",
]:
    print(plot_name)
    display(Image(filename=str(OUTPUT_DIR / "artifacts" / plot_name)))
'''
    return [
        markdown_cell(method_markdown(scenario)),
        markdown_cell("## 5. Kiểm tra môi trường Kaggle GPU T4 x2\n"),
        code_cell(environment_cell),
        markdown_cell("## 6. Cấu hình thực nghiệm đã khóa\n"),
        code_cell(config_cell),
        markdown_cell("## 7. Kiểm tra hợp đồng dữ liệu trước khi chạy\n"),
        code_cell(validate_cell),
        markdown_cell("## 8. Entry point client-parallel tự chứa\n"),
        code_cell(runtime_cell),
        code_cell(write_cell),
        markdown_cell("## 9. Chạy coordinator và hai GPU worker\n"),
        code_cell(launch_cell),
        markdown_cell("## 10. Nghiệm thu output\n"),
        code_cell(verify_cell),
        markdown_cell("## 11. Hiển thị các đồ thị chẩn đoán\n"),
        code_cell(display_cell),
    ]


def main():
    runtime_source = RUNTIME_PATH.read_text(encoding="utf-8")
    for scenario in SCENARIOS:
        folder = ROOT / scenario["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        notebook = {
            "cells": notebook_cells(scenario, runtime_source),
            "metadata": {
                "accelerator": "GPU",
                "kaggle": {"accelerator": "GPU T4 x2", "internet": False},
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = folder / scenario["filename"]
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
