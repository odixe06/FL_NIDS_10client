# CONTEXT: FD-IDS (DNN + FedProx + round-wise KD) trên CIC-IoT 2023

> Dựng lại paper **FD-IDS** (*Sensors* 2025) cho dataset CIC-IoT 2023 đã tiền xử lý, chạy trên Kaggle 2×T4.
> Notebook chính: `paper1.ipynb`. Tham chiếu paper: `paper_nids_kd.md`. Mô tả dữ liệu: `data_description.md`.

## 1. Quyết định đã chốt (locked)

| Hạng mục | Quyết định | Lý do |
| --- | --- | --- |
| **Model** | DNN nhẹ `[25→32→64→128→64→32→34]`, ReLU, output logits | Đúng model paper dùng (không phải GNN). Hợp dữ liệu flow dạng bảng, nhẹ cho FL/edge |
| **FL strategy** | FedProx (μ=0.01) + **round-wise KD** | Đúng FD-IDS: global=teacher, local=student mỗi round |
| **Loss** | `L = λ·CE + (1−λ)·KD_soft + β·L_proximal` | λ=0.5, β=0.1, T=3 (theo paper) |
| **Aggregation** | weighted FedAvg theo số mẫu | `w_G = Σ (n_k/n)·w_k` |
| **Mất cân bằng** | **Giữ nguyên dữ liệu, CrossEntropy thuần** (không re-balance) | Trọng tâm: lightweight trong môi trường non-IID/mất cân bằng. Ưu tiên accuracy/weighted-F1 |
| **Best model** | theo `weighted_f1` | khớp ưu tiên accuracy/weighted-F1 |
| **Lịch train** | `LOCAL_EPOCHS=1`, `COMMUNICATION_ROUNDS=10`, full data | quy mô ~36M mẫu; client 8 = 12.7M |
| **Batch size** | 4096 (chỉnh được) | DNN nhẹ + data lớn → batch lớn cho throughput (paper dùng 128 cho dataset nhỏ hơn) |
| **Chuẩn hóa** | KHÔNG chuẩn hóa lại | CSV client/test đã QuantileTransformer + đã chọn 25 feature; chỉ map nhãn |

> **Khác biệt so với paper (có chủ ý):** paper dùng 9 client, 40 round, 2 local epoch, batch 128, MI feature selection.
> Ở đây: 10 client (đã chia sẵn), 10 round, 1 local epoch, batch 4096, feature đã chọn bằng XGBoost (offline).

## 2. Dataset (CIC-IoT 2023, đã xử lý)

- 10 client **non-IID** (Dirichlet α=0.2), số mẫu lệch mạnh:

| Client | Mẫu | Client | Mẫu |
| --- | ---: | --- | ---: |
| 1 | 31.520 | 6 | 2.686.617 |
| 2 | 2.211.589 | 7 | 1.824.975 |
| 3 | 3.728.450 | 8 | **12.759.509** |
| 4 | 1.920.254 | 9 | 4.354.192 |
| 5 | 2.531.881 | 10 | 3.965.607 |

- **25 feature** (đã chọn + chuẩn hóa) + 1 cột nhãn. **34 lớp** (33 tấn công + BENIGN), cực mất cân bằng (1.196 → 6.9M).
- Đường dẫn Kaggle: `/kaggle/input/datasets/odixe0502/data-for-10clients/`
  - `client_1_train.csv` … `client_10_train.csv`, `global_test_data.csv`, `label_mapping.csv`
  - Notebook auto-detect đường dẫn nếu khác (rglob `client_1_train.csv`).

## 3. Pipeline notebook (`paper1.ipynb`)

| Cell | Nội dung | Kiểm tra |
| --- | --- | --- |
| 1 | Import, config `CFG`, seed, device, output dirs, in info 2×T4 | nhận đủ 2 GPU |
| 2 | Load `label_mapping.csv`, suy ra cột nhãn, validate 25 feature & 34 lớp, đếm dòng từng client | đúng schema |
| 3 | Load 10 client + test vào RAM 1 lần (float32), guard nhãn ∈[0,33], tạo quick-eval 10k (stratified) | RAM ~4.6GB |
| 4 | Định nghĩa `FDIDS_DNN` + forward smoke test | `logits=(B,34)`, hữu hạn |
| 5 | `kd_loss` (KL Hinton, batchmean), `proximal_term` + smoke test backward | loss/grad hữu hạn |
| 6 | `train_one_client` (teacher frozen + student + KD + FedProx) + debug client 1 | state_dict hợp lệ |
| 7 | `build_gpu_groups` (LPT), `train_client_group`, `weighted_fedavg`, `evaluate`, `compute_metrics` | — |
| 8 | **Vòng train chính 10 round**: 2 GPU song song → FedAvg → quick-eval → log + checkpoint + best + resume | `metrics_round.csv` |
| 9 | Đánh giá cuối trên **full** `global_test_data.csv`: accuracy, macro/weighted P/R/F1, classification report 34 lớp, confusion matrix, FPR/FNR (attack vs benign) | — |
| 10 | Vẽ loss/accuracy/F1 curve + confusion matrix | figures/*.png |

### Loss chi tiết (Cell 5)
```
L_hard = CrossEntropy(student_logits, y)
L_soft = T^2 · KL( softmax(teacher/T) ‖ softmax(student/T) )     # F.kl_div(log_softmax(student/T), softmax(teacher/T), batchmean)
L_prox = (μ/2)·Σ‖w_student − w_global‖²                          # w_global = trọng số global đầu round (frozen)
L      = λ·L_hard + (1−λ)·L_soft + β·L_prox
```

### Tối ưu 2×T4 (Cell 7–8)
- LPT chia 10 client thành 2 nhóm cân tải; mỗi nhóm 1 GPU, train tuần tự trong nhóm.
- 2 nhóm chạy song song bằng `ThreadPoolExecutor(max_workers=2)`; mỗi thread cố định 1 `cuda:id`.
- `global_state` giữ trên CPU giữa các round; mỗi client clone lên GPU khi train.
- Sau mỗi client: `del` + `torch.cuda.empty_cache()`.
- AMP mặc định **tắt** (DNN nhỏ, KL fp16 dễ kém ổn định); bật bằng `CFG.use_amp=True` nếu cần.

## 4. Cấu hình (`CFG` trong Cell 1)
```
num_clients=10  input_dim=25  num_classes=34  hidden_dims=(32,64,128,64,32)
local_epochs=1  comm_rounds=10  batch_size=4096  eval_batch_size=16384
learning_rate=1e-3  weight_decay=0.0
temperature=3.0  lambda_hard=0.5  mu_prox=0.01  beta_prox=0.1
quick_eval_size=10_000  best_metric="weighted_f1"  seed=42
use_amp=False  use_multi_gpu=True
max_samples_per_client=None  max_debug_batches=None
resume=False  resume_round=0  run_id="fdids_dnn_run1"
```

## 5. Chiến lược test & chỉnh trong lúc train

**Smoke test trước khi train full:** đặt `comm_rounds=1, max_debug_batches=10, quick_eval_size=1000` → chạy Cell 1→8, kiểm tra không NaN, ghi được metrics + checkpoint. Sau đó trả về giá trị thật.

**Theo dõi mỗi round (in ra stdout, xem qua ngrok):** `client_loss, quick_loss, accuracy, macro_f1, weighted_f1, round_time, cờ BEST`.

**Khi thông số bất ổn:**
- `quick_loss` tăng/dao động → giảm `learning_rate` hoặc `batch_size`.
- accuracy/weighted_f1 bão hòa sớm → tăng `comm_rounds` hoặc `local_epochs`.
- round quá lâu → tăng `batch_size` (≤8192) hoặc đặt `max_samples_per_client`.
- OOM → giảm `batch_size` / `eval_batch_size`.
- macro_f1 rất thấp (kỳ vọng, do mất cân bằng + non-IID): không phải lỗi — đây là lý do ưu tiên accuracy/weighted-F1.

**Ablation (đổi `run_id` mỗi lần để tách output):**
- Tắt KD: `lambda_hard=1.0`. Tắt FedProx: `beta_prox=0.0`. FedAvg thuần: cả hai.

**Resume khi bị ngắt:** `resume=True`, `resume_round=<round cuối đã lưu>` → Cell 8 load `global_round_{r}.pt`, train tiếp, cắt dòng metrics > r để tránh trùng.

## 6. Artifact & lưu kết quả về local

**Trên Kaggle** (`/kaggle/working/fd_ids_outputs/<run_id>/`):
```
checkpoints/  best_global_model.pt, global_round_1..10.pt
reports/      metrics_round.csv, classification_report.csv, confusion_matrix.npy, final_report.json, results_summary.txt
figures/      loss_curve.png, accuracy_curve.png, f1_curve.png, training_curves.png, confusion_matrix.png
quick_eval_10000.csv
```

**Lưu một bản trên local (máy WSL)** — `paper1/results/`:
- Training chạy trên Kaggle, máy local không tự nhận file ⇒ cần đưa kết quả về theo 1 trong các cách:
  1. Dán log từng round (stdout) cho Claude → Claude lưu vào `results/runs/<run_id>/` và hiển thị.
  2. Download `metrics_round.csv` + `final_report.json` + figures từ Kaggle, bỏ vào `results/incoming/` → Claude archive + hiển thị.
  3. Nếu Kaggle mount Google Drive / có path ngrok đồng bộ được về local → đặt `CFG.local_mirror="<path>"` để notebook tự copy.
- Notebook in `metrics_round.csv` dạng bảng + `results_summary.txt` (dễ copy) để bạn dán lại nhanh.
- Mỗi round đều in ra stdout (flush) → xem trực tiếp qua ngrok.

## 8. Nhật ký vấn đề & điều chỉnh (cập nhật khi có vấn đề)

> Mỗi khi gặp vấn đề trong lúc train (loss bất thường, OOM, metric kém, lệch kỳ vọng...) và cách xử lý, ghi vào đây để phục vụ viết báo cáo và các phiên làm việc sau.

| Ngày | Round/Giai đoạn | Vấn đề | Nguyên nhân | Điều chỉnh | Kết quả |
| --- | --- | --- | --- | --- | --- |
| 2026-06-23 | Khởi tạo | Bản cũ dùng GNN kNN động → macro F1 ~0.25, chậm, lệch paper | Sai lựa chọn model cho dữ liệu bảng | Đổi sang **DNN theo paper** + giữ FedProx/KD | (chờ chạy lại trên Kaggle) |
| 2026-06-23 | Review code (pre-train) | macro/weighted F1 từng round chỉ tính trên các lớp xuất hiện trong quick-eval 10k (lớp hiếm vắng) → macro_f1/round bị "thổi phồng", không so được với Cell 9 | `precision_recall_fscore_support` thiếu `labels=` | Thêm `labels=range(34)` vào `compute_metrics` | Đã fix, validate OK |
| 2026-06-23 | Review code (pre-train) | `resume=True` mà thiếu checkpoint → âm thầm xóa `metrics_round.csv` và train lại từ round 1 (resume_round mặc định 0, `global_round_0.pt` không tồn tại) | Điều kiện AND gộp existence-check vào nhánh resume | `resume=True` → assert checkpoint tồn tại; chỉ xóa metrics ở nhánh train mới | Đã fix, validate OK |
| 2026-06-23 | Review code (pre-train) | Đọc toàn bộ CSV (~36M dòng) 2 lần (Cell 2 đếm dòng + Cell 3 load) → lãng phí I/O | Vòng `count_rows` chỉ để in số dòng | Bỏ `count_rows`; số mẫu in ở Cell 3 lúc load | Đã fix |
| 2026-06-23 | Review code (pre-train) | Client 8 (12.7M) train im lặng vài phút → tưởng treo khi xem qua ngrok | Không log trong vòng batch | Heartbeat `CFG.log_every=200` in tiến độ batch | Đã fix |

## 7. Tiêu chí hoàn thành
- Chạy hết 10 round, mỗi round đủ cập nhật từ 10 client; KD + FedProx tính trong local training mọi round.
- Global & local đều là DNN; aggregation weighted FedAvg.
- Có metric từng round, best model theo weighted_f1, đánh giá cuối full test với đủ accuracy/precision/recall/F1 + confusion matrix + FPR/FNR.
- Có loss/accuracy/F1 curve.
