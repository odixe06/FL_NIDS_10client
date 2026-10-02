# PerFed-SKD trên CICIoT2023: đặc tả kiến trúc, huấn luyện và đầu ra

## 1. Mục đích và phạm vi

Tài liệu này là hợp đồng chung cho ba thí nghiệm PerFed-SKD:

| Notebook | `run_name` | Global/local/student/teacher model |
|---|---|---|
| 10 GRU | `fd_ids_ciciot2023_10c_gru` | GRU hai tầng |
| 10 Transformer | `fd_ids_ciciot2023_10c_transformer` | Transformer encoder hai tầng |
| 10 CNN-1D | `fd_ids_ciciot2023_10c_cnn1d` | CNN-1D |

Kịch bản hỗn hợp 5 GRU + 5 Transformer không còn thuộc phạm vi. Mỗi notebook
có đúng 10 client đồng nhất kiến trúc, vì Algorithm 1 của paper phải tổng hợp
trực tiếp các trọng số local model thành global model.

Phương pháp huấn luyện phải bám sát PerFed-SKD trong paper:

- server pretrain global model một epoch;
- mỗi client giữ personalized model lịch sử `V_m`;
- client được chọn khởi tạo student bằng global model mới và học dưới sự hướng
  dẫn của snapshot `V_m`;
- client không được chọn tiếp tục từ personalized model hiện tại;
- server dùng ngưỡng accuracy trung bình của đủ 10 client để chọn thiết bị cho
  vòng tiếp theo;
- chỉ trọng số của các client thuộc tập được chọn `S_t` được upload và tổng hợp.

Không dùng proxy model riêng, adaptive mutual distillation, FedProx, DDP,
`DataParallel` hoặc gradient all-reduce. Các key output tương thích cũ vẫn được
giữ khi cần để ghép report, nhưng không được gán ngữ nghĩa proxy giả.

## 2. Hợp đồng dữ liệu

### 2.1. Đường dẫn và file

Thư mục Kaggle:

```text
/kaggle/input/datasets/odixe0502/data-for-10clients/
```

Các file bắt buộc:

- `client_1_train.csv` đến `client_10_train.csv`: dữ liệu cục bộ của 10 client;
- `global_train_data.csv`: dữ liệu server dùng để pretrain đúng một epoch;
- `global_test_data.csv`: tập test độc lập, chỉ dùng sau khi đã chọn checkpoint;
- `label_mapping.csv`: ánh xạ nhãn.

`global_train_data.csv` chứa cùng miền dữ liệu huấn luyện đã được phân cho các
client. Việc dùng file này để pretrain là quyết định thí nghiệm đã được xác
nhận, không được mô tả như một đảm bảo privacy mạnh hơn thực tế.

### 2.2. Task, feature và label

- Bài toán: phân loại đa lớp.
- Số lớp: 34.
- Label là số nguyên trong `[0, 33]`.
- Input model: tensor float32 `[batch, 25]`.
- Output model: logits float32/AMP `[batch, 34]`.
- Không đặt softmax trong model.
- Thứ tự đặc trưng cố định:

```text
ack_flag_number, AVG, Std, UDP, fin_count, Max, TCP, syn_count,
Protocol Type, Rate, IAT, syn_flag_number, rst_flag_number, Tot sum,
HTTPS, ack_count, fin_flag_number, HTTP, rst_count, Header_Length,
psh_flag_number, ICMP, Time_To_Live, ARP, DNS
```

Notebook phải kiểm tra chính xác header, thứ tự cột, dtype, NaN/Inf và miền
label trước khi chuyển dữ liệu sang GPU.

### 2.3. Split

Mỗi file client được tách độc lập theo lớp:

- 95% local train;
- 5% local validation;
- lớp chỉ có một mẫu được giữ trong local train;
- seed split là 42;
- `global_test_data.csv` không tham gia chọn checkpoint hoặc device selection.

Các mảng trung gian được ghi dưới dạng NumPy memmap tại:

```text
/kaggle/temp/{run_name}_cache/
```

Cache này là dữ liệu dùng một lần của phiên Kaggle, không phải output lâu dài.

## 3. Cấu hình huấn luyện đã khóa

| Thành phần | Giá trị |
|---|---:|
| Số client | 10 |
| Communication rounds | 10 |
| Local epochs mỗi round | 1 |
| Server pretrain epochs | 1 |
| Batch size mỗi optimizer update | 1024 |
| Gradient accumulation | Không (`1` step) |
| Optimizer | SGD |
| Learning rate `η` | `0.01` |
| Momentum | `0` |
| Weight decay | `0` |
| Scheduler | Không |
| Label loss | Cross-entropy |
| Distillation | KL divergence |
| Distillation weight `λ` | `1.0` |
| Temperature `T` | `1.0` |
| Class weighting/resampling | Không |
| Seed và initialization seed | 42 |
| Checkpoint criterion | Global validation macro-F1 lớn nhất |
| Thiết bị | Đúng 2 GPU NVIDIA T4 |
| Multi-GPU | Client-parallel, không DDP |
| Worker | 2 process cố định, 1 process/GPU |
| GPU cache safety fraction | Tối đa 70% VRAM/GPU |
| CUDA stream candidates | Benchmark `[1, 2]` |
| Mixed precision | CUDA AMP |

Batch 1024 là batch của một model update trên đúng một GPU, không phải global
batch chia qua hai GPU. Mỗi mẫu local train xuất hiện đúng một lần trong local
epoch, không sampler padding.

Server pretrain dùng cùng batch 1024 và SGD contract trên một T4 trong entry
point client-parallel. GPU còn lại có thể chuẩn bị/kiểm tra cache nhưng không
được tạo một update độc lập rồi trung bình trọng số, vì thao tác đó sẽ thay đổi
nghĩa của một central pretrain epoch. Việc không dùng DDP cho riêng pretrain là
chủ ý để giữ một execution topology duy nhất.

## 4. Thuật toán PerFed-SKD được triển khai

### 4.1. Global objective

Paper mô tả mục tiêu FL:

$$
\min_{\omega} F(\omega)
=\sum_{m=1}^{M}\frac{|D_m|}{|D|}F_m(\omega).
$$

Algorithm 1 sau đó dùng trung bình model của tập client được chọn. Để bám sát
Algorithm 1, notebook dùng trung bình cộng không trọng số trên `S_t`, không dùng
FedAvg theo số mẫu.

### 4.2. Server pretrain

Từ initial state xác định bởi seed 42, server huấn luyện global model
`ω^0` đúng một epoch trên `global_train_data.csv` bằng cross-entropy:

$$
\ell_{\mathrm{pretrain}}=\operatorname{CE}(y,z_{\omega^0}).
$$

State sau pretrain là global model được gửi cho cả 10 client ở round 1.

### 4.3. Historical teacher và student loss

Tại đầu mỗi local update, client `m` tạo một frozen snapshot của personalized
model trước update làm historical teacher `V_m`. Student là:

- global state mới nhất nếu `m ∈ S_t`;
- personalized state hiện tại nếu `m ∉ S_t`.

Teacher chạy `eval()` và `no_grad()`. Với student logits `z_m`, teacher logits
`z_{V_m}`, temperature `T=1`:

$$
\ell_{\mathrm{CE}}=\operatorname{CE}(y,z_m),
$$

$$
\ell_{\mathrm{KD}}
=T^2\operatorname{KL}\left(
\operatorname{softmax}(z_{V_m}/T)
\;\middle\|\;
\operatorname{softmax}(z_m/T)
\right),
$$

$$
\phi_m=\ell_{\mathrm{CE}}+\lambda\ell_{\mathrm{KD}},
\qquad \lambda=1.
$$

PyTorch triển khai công thức trên bằng:

```python
F.kl_div(
    F.log_softmax(student_logits / temperature, dim=1),
    F.softmax(teacher_logits / temperature, dim=1),
    reduction="batchmean",
) * (temperature ** 2)
```

Chỉ student nhận gradient. Sau local epoch:

$$
\omega_m \leftarrow \omega_m-\eta\nabla\phi_m,
\qquad V_m\leftarrow\omega_m.
$$

Round 1 chưa có personalized history độc lập; teacher của mỗi client là frozen
snapshot của pretrained global state. KL ban đầu bằng 0 nhưng trở nên khác 0
khi student cập nhật qua các mini-batch.

### 4.4. Device selection và timeline

Đặt `M={1,…,10}` và `S_1=M`. Trong mỗi round `t`:

1. Client trong `S_t` nhận global state `ω^t`; client ngoài `S_t` giữ state cục
   bộ hiện tại.
2. Cả 10 client local-train một epoch. Việc tất cả client tiếp tục train là cần
   để duy trì personalized model và thu được `a_m^t` cho đúng công thức ngưỡng.
3. `a_m^t` là accuracy của personalized model sau local update, đánh giá trên
   validation split của chính client `m`.
4. Server tính:

$$
A^t=\frac{1}{|M|}\sum_{m\in M}a_m^t.
$$

5. Tập nhận global model ở vòng kế tiếp:

$$
S_{t+1}=\{m\in M\mid a_m^t<A^t\}.
$$

Không thay `A^t` bằng global-model accuracy trên dữ liệu local. Không dùng
macro-F1 để chọn thiết bị. So sánh là strict `<`, đúng Algorithm 1.

### 4.5. Aggregation

Các client trong `S_t` upload student state sau local update. Server tổng hợp:

$$
\omega^{t+1}
=\frac{1}{|S_t|}\sum_{m\in S_t}\omega_m^{t+1}.
$$

Client ngoài `S_t` không upload và không tham gia aggregation của round đó.
Nếu `S_t` rỗng, global state được giữ nguyên và logical communication của round
đó bằng 0; các client vẫn local-train để có thể tạo `S_{t+1}`.

Paper có một điểm không nhất quán: prose nhấn mạnh chỉ selected devices trao
đổi model, trong khi ký hiệu dòng aggregation dùng `m∈M` nhưng chia cho `|S|`.
Notebook ưu tiên mục tiêu giảm communication, điều kiện chọn `a<τ`, vòng lặp
selected devices và mẫu số `|S|`; vì vậy tổng chỉ chạy trên `S_t`.

### 4.6. Validation, checkpoint và test

- Device selection dùng local validation accuracy như mục 4.4.
- Sau aggregation, global model được đánh giá đúng một lần trên hợp của 10
  validation split bằng cách cộng raw confusion matrix và loss sums.
- `best.pt` được chọn bằng global validation macro-F1; test không tham gia.
- `last.pt` luôn là state sau round 10.
- Sau round 10, nạp lại `best.pt`; đánh giá global model và từng personalized
  model trên `global_test_data.csv`.
- Không có final fine-tune ngoài 10 round.

## 5. Kiến trúc mô hình

### 5.1. GRU

25 đặc trưng được xem như chuỗi dài 25, mỗi bước chứa một giá trị.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều cuối | `[B,25,1]` |
| GRU | `[B,25,1]` | 2 tầng, hidden 64, dropout 0,2 | `[B,25,64]` |
| Last-step | `[B,25,64]` | lấy bước cuối | `[B,64]` |
| Linear | `[B,64]` | 64 → 34 | `[B,34]` |

Số tham số trainable bắt buộc: **40.034**.

### 5.2. Transformer

Mỗi đặc trưng là một token vô hướng. Giá trị được chiếu lên 64 chiều và cộng
learned positional embedding.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Scalar projection | `[B,25,1]` | Linear 1 → 64 | `[B,25,64]` |
| Positional embedding | `[B,25,64]` | learned, chiều dài 25 | `[B,25,64]` |
| Encoder ×2 | `[B,25,64]` | 4 heads, FFN 128, dropout 0,1 | `[B,25,64]` |
| Mean pooling | `[B,25,64]` | trung bình theo token | `[B,64]` |
| LayerNorm + Linear | `[B,64]` | 64 → 34 | `[B,34]` |

Số tham số trainable bắt buộc: **71.010**.

### 5.3. CNN-1D

25 đặc trưng được xem như tín hiệu một kênh dài 25.

| Khối | Input | Cấu hình | Output chính |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều kênh | `[B,1,25]` |
| Conv block 1 | `[B,1,25]` | Conv 1→32, k3, padding 1, BN, ReLU | `[B,32,25]` |
| Conv block 2 | `[B,32,25]` | Conv 32→64, k3, padding 1, BN, ReLU, MaxPool 2 | `[B,64,12]` |
| Conv block 3 | `[B,64,12]` | Conv 64→128, k3, padding 1, BN, ReLU | `[B,128,12]` |
| Pool + Linear | `[B,128,12]` | adaptive average pool, 128→34 | `[B,34]` |

Số tham số trainable bắt buộc: **35.874**.

### 5.4. Khởi tạo công bằng

- `seed = initialization_seed = 42`.
- Mỗi lần tạo initial state của một họ model phải dùng RNG fork độc lập và gọi
  `torch.manual_seed(42)`.
- Pretrained global state của từng notebook bắt đầu từ initial state này.
- Mười client ở round 1 nhận cùng pretrained global state.
- SHA-256 của ordered tensor state được ghi cho initial state, pretrained global
  state, từng global checkpoint và personalized states.
- Seed train được suy ra từ `(seed, phase, round, client_id)`, không phụ thuộc
  GPU sở hữu.

## 6. Tối ưu hai GPU T4

Notebook dùng CPU coordinator và đúng hai worker `multiprocessing.spawn` cố
định:

- worker 0 gắn `cuda:0`;
- worker 1 gắn `cuda:1`;
- không khởi tạo `torch.distributed`, NCCL, DDP hoặc `DataParallel`;
- aggregation, checkpoint, consolidated metrics và plot chỉ do coordinator
  thực hiện;
- worker chỉ ghi log/payload tạm có tên riêng;
- traceback/exit code của worker phải được truyền về coordinator.

### 6.1. Phân hoạch workload

Sau pretrain, hai worker benchmark optimizer step của model thực tế. Workload
client:

$$
\left\lceil\frac{n_m^{train}}{1024}\right\rceil
\times\text{seconds-per-step}.
$$

Coordinator duyệt mọi bipartition hai phía không rỗng của 10 client, chọn
predicted makespan nhỏ nhất và tie-break bằng client ID. Assignment giữ cố định
trong toàn bộ phiên.

### 6.2. Cache và CUDA stream

- Cache feature, label và split index trên GPU sở hữu nếu tổng persistent bytes
  không vượt 70% VRAM.
- Nếu không đủ, dùng deterministic LRU và pinned-memory async fallback; không
  giảm hoặc lấy mẫu dữ liệu.
- Benchmark 1 và 2 independent CUDA streams bằng model/batch thật.
- Chỉ chọn 2 stream nếu measured aggregate samples/s cao hơn.
- GRU/Transformer có dropout nên runtime vẫn khóa 1 stream/GPU sau benchmark để
  RNG của từng client chỉ phụ thuộc `(seed, round, client_id)`, không phụ thuộc
  lịch xen kẽ kernel; CNN-1D không có stochastic layer nên được phép chọn 2.
- Mỗi concurrent client có model, frozen teacher, optimizer, scaler, stream và
  RNG riêng.
- Tensor permutation/loss accumulator của client phải được tạo trong đúng
  stream context.
- Đồng bộ ở phase/task boundary, không `.item()` trong batch loop.
- Không dùng CUDA Graphs.

### 6.3. AMP và hiệu năng

- Dùng CUDA AMP và `GradScaler`.
- Cho phép TF32 trên T4 khi PyTorch/CUDA hỗ trợ.
- Dùng pinned memory, non-blocking transfer và chunked CSV/memmap pipeline.
- Evaluation dùng inference mode và AMP.
- Không đổi batch, loss, optimizer hoặc sample accounting để đổi lấy tốc độ.

## 7. Quy ước metric

Mọi tỷ lệ trong JSON/CSV nằm trong `[0,1]`. Thời gian dùng giây.

Từ confusion matrix đa lớp, với từng lớp `c`:

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\qquad
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c},
$$

$$
\mathrm{F1}_c=
2\frac{\mathrm{Precision}_c\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c},
$$

$$
\mathrm{FPR}_c=\frac{FP_c}{FP_c+TN_c},\qquad
\mathrm{FNR}_c=\frac{FN_c}{FN_c+TP_c}.
$$

- `macro_*`: trung bình không trọng số trên đủ 34 lớp, zero denominator → 0;
- `weighted_*`: trung bình theo support nhãn thật;
- `accuracy`: trace confusion matrix chia tổng mẫu;
- `multiclass_macro_fpr/fnr`: one-vs-rest trung bình trên 34 lớp;
- `binary_attack_*`: gộp BENIGN thành normal, 33 lớp còn lại là attack positive.

## 8. Cấu trúc output bắt buộc

Mỗi notebook ghi:

```text
/kaggle/working/{run_name}/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── run.log
│   └── rank_1.log
├── metrics/
│   ├── config.json
│   ├── dataset_summary.json
│   ├── client_class_distribution.csv
│   ├── history_pretrain.csv
│   ├── history_pretrain.json
│   ├── history_round.csv
│   ├── history_round.json
│   ├── history_client.csv
│   ├── history_client.json
│   ├── history_local_epoch.csv
│   ├── history_local_epoch.json
│   ├── summary.json
│   ├── classification_report.json
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.npy
│   ├── communication_costs.json
│   ├── communication_costs.csv
│   └── runtime_breakdown.json
└── artifacts/
    ├── class_distribution.png
    ├── accuracy_f1_curves.png
    ├── loss_curves.png
    ├── confusion_matrix.png
    ├── per_class_f1.png
    ├── runtime_per_round.png
    └── communication_cumulative.png
```

## 9. Schema checkpoint và history

### 9.1. Checkpoint

`best.pt` và `last.pt` là dictionary gồm:

- `model_state_dict`: global model state, không có prefix `module.`;
- `personalized_model_state_dicts`: state của đủ 10 client;
- `personalized_model_state_dicts_before_finetune`: compatibility alias bằng
  đúng `personalized_model_state_dicts`, vì đặc tả mới không có final fine-tune;
- `historical_teacher_model_state_dicts`: frozen history dùng cho round kế;
- `initial_model_state_dicts_by_family`;
- `initialization_hashes`;
- `pretrained_global_state_hash`;
- `global_state_hash`;
- `round`;
- `selected_clients_current_round`;
- `selected_clients_next_round`;
- `selection_threshold_accuracy`;
- `local_accuracies`;
- `config`, `feature_columns`, `label_mapping`;
- `validation_metrics`;
- `model_metadata`.

Các key cũ `model_state_dict` và `personalized_model_state_dicts` được giữ
nguyên để công cụ so sánh trọng số không phải đổi giao diện.

### 9.2. `history_pretrain`

Đúng một hàng:

- `epoch=1`, `train_examples`, `optimizer_steps`;
- `cross_entropy_loss`, `epoch_seconds`;
- `gpu_id`, peak allocated/reserved memory.

### 9.3. `history_local_epoch`

Mỗi round có đúng 10 hàng, tổng cộng 100 hàng:

- `phase="federated_self_distillation"`, `round`, `client_id`,
  `local_epoch=1`;
- `selected_for_global_update`, `received_global_state`;
- `train_examples`, `optimizer_steps`, `sampler_padding_rows=0`;
- `hard_loss`: alias của `cross_entropy_loss`;
- `soft_loss`: alias của `distillation_kl_loss`;
- `proximal_loss=0`;
- `total_loss`;
- `cross_entropy_loss`, `distillation_kl_loss`;
- `lambda`, `temperature`;
- `epoch_seconds`, `gpu_id`, `concurrent_streams`, `gpu_cache_mode`.

Các cột `proxy_label_loss` và `proxy_total_loss` của đặc tả cũ không còn tồn
tại vì phương pháp paper gốc không có proxy model.

### 9.4. `history_client`

Mỗi round có đúng 10 hàng, tổng cộng 100 hàng:

- `round`, `client_id`, train/validation samples;
- selected/received/upload flags;
- local loss và local train/validation seconds;
- local validation loss và toàn bộ multiclass/binary metrics;
- threshold `A^t`, `below_global_average_threshold`;
- GPU assignment, cache, stream, peak memory và actual client time.

### 9.5. `history_round`

Đúng 10 hàng:

- mean weighted-by-examples CE/KD/total loss để report loss;
- mean/min/max local accuracy, macro-F1 và fairness standard deviation;
- `selection_threshold_accuracy`;
- `selected_clients_current`, `selected_clients_next` và số lượng tương ứng;
- global validation loss và toàn bộ metrics;
- round/cumulative time;
- exact communication bytes/MiB round và cumulative;
- peak allocated/reserved memory riêng từng GPU, max và tổng;
- predicted/actual worker load và idle time.

### 9.6. `summary.json`

File ghép report chính gồm:

- `run_name`, `model_family`, `proxy_model_family=null`, `status`;
- serialized config và environment;
- model metadata, parameter count và state hashes;
- dataset/split/class distribution;
- pretrain metrics;
- best round/global validation metrics;
- device-selection history và fairness;
- final global test metrics;
- final personalized test metrics từng client và mean/best/worst;
- runtime, communication, cache, utilization và memory summaries;
- relative paths của mọi output.

## 10. Classification report và confusion matrix

- Report/confusion matrix chính là kết quả global model từ `best.pt` trên
  `global_test_data.csv`.
- `classification_report.json/csv`: precision, recall, F1, support cho từng lớp
  và macro/weighted aggregate.
- `confusion_matrix.csv`: hàng nhãn thật, cột nhãn dự đoán.
- `confusion_matrix.npy`: cùng ma trận `int64`.
- Plot confusion matrix chuẩn hóa theo hàng.
- Personalized test metrics được lưu trong `summary.json`, không ghi đè report
  global chính.

## 11. Chi phí truyền thông

`model_state_bytes`:

$$
\sum_{\theta\in\text{state}}\operatorname{numel}(\theta)
\times\operatorname{element\_size}(\theta).
$$

Với tập selected của round `S_t`:

$$
\mathrm{bytes}_t=2|S_t|\times\mathrm{model\_state\_bytes}.
$$

Hệ số 2 gồm global download và local upload. Client ngoài `S_t` có logical
communication bằng 0. Tổng:

$$
\mathrm{total\_bytes}=\sum_{t=1}^{10}\mathrm{bytes}_t.
$$

Round 1 có `|S_1|=10`. Ghi byte chính xác và MiB với
`1 MiB=1.048.576 byte`. Không tính TCP/TLS, serialization, log, retry, cache
payload hoặc optimizer state.

`ddp_gradient_allreduce_estimate` vẫn tồn tại để tương thích report nhưng phải
có `enabled=false`, byte/MiB bằng 0.

## 12. Runtime và logging

`runtime_breakdown.json` phải tách:

- data validation/memmap/split;
- server pretrain;
- workload và stream benchmark;
- GPU cache load;
- 10 round local training/validation/aggregation;
- global validation;
- final global/personalized test;
- plotting;
- client-parallel entry-point runtime;
- subprocess wall time;
- total notebook pipeline;
- per-round/per-client time, predicted/actual load và worker idle time.

Ghi thêm:

- GPU name, VRAM, compute capability;
- peak allocated/reserved memory;
- cache bytes, hit/miss/LRU fallback;
- sampled GPU utilization/memory;
- worker traceback và nonzero exit propagation.

Python logging có argument phải dùng old-style `%` hợp lệ; không dùng `%,d`.

## 13. Điều kiện nghiệm thu notebook

Mỗi notebook phải:

1. parse JSON và syntax-check mọi code cell;
2. assert CUDA và đúng hai GPU có tên chứa `T4`;
3. assert input path/file/header/shape/dtype/label;
4. assert output `[B,34]` và exact parameter count của model;
5. dùng một serialized `CONFIG`;
6. ghi entry point độc lập dưới `/kaggle/temp`;
7. launch bằng subprocess rồi propagate lỗi;
8. dùng đúng client-parallel topology và AMP;
9. có deterministic initialization/training seed;
10. triển khai đúng frozen historical teacher, KL, `λ=1`, `T=1`;
11. tính `A^t` từ mean của đủ 10 local accuracies và chọn strict `a_m<A^t`;
12. chỉ aggregate/upload `S_t`, không sample weighting;
13. kiểm tra đúng row count: pretrain 1, round 10, client 100, local epoch 100;
14. tạo đủ best/last checkpoint, JSON/CSV/NPY/log/plot không rỗng;
15. kiểm tra exact validation/test accounting và communication theo `|S_t|`;
16. không chứa placeholder, TODO, fabricated path hoặc silent subsampling.

Kiểm tra local chỉ là structural validation. Chỉ được tuyên bố runtime validation
sau khi notebook thực sự chạy hoàn tất trên Kaggle `GPU T4 x2`.
