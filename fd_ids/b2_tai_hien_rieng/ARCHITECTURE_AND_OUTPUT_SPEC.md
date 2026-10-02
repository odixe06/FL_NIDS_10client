# FD-IDS trên CICIoT2023: đặc tả kiến trúc và đầu ra

## 1. Mục đích và phạm vi

Tài liệu này là hợp đồng đo lường chung cho ba thí nghiệm:

| Notebook | `run_name` | Mô hình cục bộ/toàn cục |
|---|---|---|
| GRU | `fd_ids_ciciot2023_10c_gru` | GRU hai tầng |
| Transformer | `fd_ids_ciciot2023_10c_transformer` | Transformer encoder hai tầng |
| CNN-1D | `fd_ids_ciciot2023_10c_cnn1d` | CNN một chiều ba khối |

Mỗi notebook chỉ chạy phương pháp chính của paper: **FedProx + round-wise
knowledge distillation (KD)**. Cùng một kiến trúc được dùng cho global teacher
và local student, vì vậy các tham số có thể được tổng hợp trực tiếp.

Các file đầu vào đã được tiền xử lý trước notebook: 25 đặc trưng được chọn bằng
XGBoost, sau đó được chuẩn hóa bằng `QuantileTransformer`. Công thức Mutual
Information (MI) của paper vẫn được ghi lại để bảo toàn mô tả phương pháp,
nhưng notebook không tuyên bố đã chạy lại MI vì dữ liệu 46 đặc trưng ban đầu
không có trong đầu vào hiện tại.

## 2. Hợp đồng dữ liệu

- Thư mục Kaggle:
  `/kaggle/input/datasets/odixe0502/data-for-10clients/`.
- Dữ liệu huấn luyện: `client_1_train.csv` đến
  `client_10_train.csv`.
- Kiểm thử độc lập: `global_test_data.csv`.
- Bảng nhãn: `label_mapping.csv`.
- `global_train_data.csv` chỉ được đếm số dòng để kiểm tra tổng số mẫu của 10
  client; file này không được dùng để huấn luyện lần thứ hai.
- Bài toán: phân loại đa lớp, 34 lớp, nhãn nguyên trong `[0, 33]`.
- Tập đặc trưng theo đúng thứ tự:

```text
ack_flag_number, AVG, Std, UDP, fin_count, Max, TCP, syn_count,
Protocol Type, Rate, IAT, syn_flag_number, rst_flag_number, Tot sum,
HTTPS, ack_count, fin_flag_number, HTTP, rst_count, Header_Length,
psh_flag_number, ICMP, Time_To_Live, ARP, DNS
```

Mỗi client được tách độc lập theo lớp:

- 95% làm local train;
- 5% làm validation;
- lớp chỉ có một mẫu được giữ ở local train;
- `global_test_data.csv` không tham gia chọn checkpoint.

Các mảng trung gian được chuyển từ CSV sang NumPy memmap trong
`/kaggle/temp/{run_name}_cache/`. Đây là cache dùng một lần của phiên Kaggle và
không phải output lâu dài.

## 3. Cấu hình huấn luyện chung

| Thành phần | Giá trị |
|---|---:|
| Số client | 10 |
| Client tham gia mỗi vòng | 10/10 |
| Communication rounds | 10 |
| Local epochs | 1 |
| Global batch size | 1024 |
| Nominal batch size mỗi GPU | 512 |
| Gradient accumulation | Không (`1` step) |
| DataLoader workers | 2/process, 4 tổng cộng |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Scheduler | Không |
| Hard-label loss | Cross-entropy |
| FedProx regularization `μ` | 0,01 |
| Hard-loss weight `λ` | 0,5 |
| Proximal-loss weight `β` | 0,1 |
| Temperature `T` | 3 |
| Class weighting/resampling | Không |
| Checkpoint criterion | Validation macro-F1 lớn nhất |
| Seed | 42 |
| Thiết bị | Đúng 2 GPU T4 |
| Multi-GPU | `torch.nn.parallel.DistributedDataParallel` |
| Process topology | 2 process, 1 process/GPU |
| Distributed backend/launcher | NCCL / `torchrun` |
| Mixed precision | CUDA AMP |

Batch size 1024 được chia đều thành 512 mẫu/GPU đối với các batch đầy đủ.
Mỗi GPU chạy một process DDP độc lập. `DistributedSampler` giữ toàn bộ dữ liệu
và có thể đệm tối đa một mẫu khi local-train split của client có số dòng lẻ,
nhằm bảo đảm hai rank có cùng số optimizer step; số dòng đệm được ghi trong
lịch sử. Validation và test dùng sampler không padding nên mỗi mẫu chỉ được
tính đúng một lần. Adam tiếp tục dùng learning rate 0,001; không áp dụng linear
scaling vì batch toàn cục vẫn là 1024. Các hệ số KD/FedProx giữ nguyên theo cấu
hình đã chốt.

Trong mỗi vòng, global model ở đầu vòng là teacher cố định. Mỗi local model
được khởi tạo từ cùng global state, học một local epoch bằng hard labels, soft
targets và proximal constraint, rồi gửi state về server. Server tính trung bình
có trọng số theo số mẫu local train thực tế.

Notebook ghi entry point Python độc lập vào `/kaggle/temp` rồi khởi chạy bằng
`python -m torch.distributed.run --standalone --nproc-per-node=2`. Cách này
tránh các lỗi multiprocessing thường gặp khi spawn trực tiếp từ kernel
notebook. Rank 0 quản lý FedAvg, checkpoint, metric và artifact; rank 1 chỉ giữ
GPU worker và log chẩn đoán riêng. Với CNN-1D, các BatchNorm layer của local
student được chuyển sang `SyncBatchNorm` để thống kê batch phản ánh đủ hai GPU.

## 4. Công thức và ánh xạ triển khai

### 4.1. Mutual Information

$$
I(V_x;V_y)=E(V_x)+E(V_y)-JE(V_x,V_y)
$$

MI chỉ được mô tả để đối chiếu paper. Đầu vào notebook đã hoàn tất một quy trình
chọn 25 đặc trưng khác bằng XGBoost.

### 4.2. FedAvg có trọng số

$$
w_G^{t+1}=\sum_{k=1}^{K}\frac{n_k}{n}w_k^{t+1},
\qquad n=\sum_{k=1}^{K}n_k
$$

`n_k` là số mẫu **local train sau khi giữ lại validation**, không phải tổng số
dòng ban đầu của client.

### 4.3. FedProx

$$
L_{\mathrm{proximal}}=
\frac{\mu}{2}\left\|w_k-w_G^t\right\|_2^2
$$

Khoảng cách được cộng trên tất cả tham số trainable của student và global
teacher.

### 4.4. Round-wise knowledge distillation

$$
L_{\mathrm{soft}}=
T^2\,KL\left(
\operatorname{softmax}(Z_t/T)
\parallel
\operatorname{softmax}(Z_s/T)
\right)
$$

Trong PyTorch, hướng KL trên được thực hiện bằng
`kl_div(log_softmax(student/T), softmax(teacher/T))`.

$$
L_{\mathrm{total}}=
\lambda L_{\mathrm{hard}}+
(1-\lambda)L_{\mathrm{soft}}+
\beta L_{\mathrm{proximal}}
$$

KD được áp dụng ở mọi batch của mọi communication round. Teacher chạy ở chế
độ `eval()` và không nhận gradient.

## 5. Kiến trúc mô hình

Tất cả mô hình nhận tensor `[batch, 25]` và trả về logits `[batch, 34]`. Không
đặt softmax trong model vì cross-entropy và KD nhận logits trực tiếp.

### 5.1. GRU

25 đặc trưng được xem như một chuỗi có độ dài 25, mỗi bước chứa một giá trị.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều cuối | `[B,25,1]` |
| GRU | `[B,25,1]` | 2 tầng, hidden 64, dropout 0,2 | `[B,25,64]` |
| Last-step | `[B,25,64]` | lấy bước cuối | `[B,64]` |
| Linear | `[B,64]` | 64 → 34 | `[B,34]` |

Số tham số trainable dự kiến: **40.034**.

### 5.2. Transformer

Mỗi đặc trưng là một token vô hướng. Giá trị được chiếu lên 64 chiều và cộng
learned positional embedding để giữ danh tính/vị trí đặc trưng.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Scalar projection | `[B,25,1]` | Linear 1 → 64 | `[B,25,64]` |
| Positional embedding | `[B,25,64]` | learned, chiều dài 25 | `[B,25,64]` |
| Encoder ×2 | `[B,25,64]` | 4 heads, FFN 128, dropout 0,1 | `[B,25,64]` |
| Mean pooling | `[B,25,64]` | trung bình theo token | `[B,64]` |
| LayerNorm + Linear | `[B,64]` | 64 → 34 | `[B,34]` |

Số tham số trainable dự kiến: **71.010**.

### 5.3. CNN-1D

25 đặc trưng được xem như một tín hiệu một kênh có chiều dài 25.

| Khối | Input | Cấu hình | Output chính |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều kênh | `[B,1,25]` |
| Conv block 1 | `[B,1,25]` | Conv 1→32, kernel 3, BN, ReLU | `[B,32,25]` |
| Conv block 2 | `[B,32,25]` | Conv 32→64, kernel 3, BN, ReLU, MaxPool 2 | `[B,64,12]` |
| Conv block 3 | `[B,64,12]` | Conv 64→128, kernel 3, BN, ReLU | `[B,128,12]` |
| Pool + Linear | `[B,128,12]` | adaptive average pool, 128→34 | `[B,34]` |

Số tham số trainable dự kiến: **35.874**.

## 6. Quy ước metric

Mọi tỷ lệ trong JSON/CSV nằm trong `[0,1]`, không phải phần trăm. Thời gian dùng
đơn vị giây và được đo bằng `time.perf_counter()`.

Từ confusion matrix đa lớp, với mỗi lớp `c`:

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\quad
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c}
$$

$$
\mathrm{F1}_c=
2\frac{\mathrm{Precision}_c\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c}
$$

$$
\mathrm{FPR}_c=\frac{FP_c}{FP_c+TN_c},\quad
\mathrm{FNR}_c=\frac{FN_c}{FN_c+TP_c}
$$

- `macro_*`: trung bình không trọng số trên đủ 34 lớp; lớp có mẫu số bằng
  không nhận giá trị 0.
- `weighted_*`: trung bình theo support của lớp thật.
- `accuracy`: tổng phần tử đường chéo chia tổng số mẫu.
- `multiclass_macro_fpr` và `multiclass_macro_fnr`: trung bình one-vs-rest
  trên 34 lớp.
- Nhóm metric `binary_attack_*`: gộp nhãn `BENIGN` thành normal và 33 nhãn còn
  lại thành attack. Attack là positive class.

Không dùng test metric để chọn checkpoint. `best.pt` được chọn bằng validation
macro-F1; test chỉ chạy một lần sau khi nạp lại checkpoint này.

## 7. Cấu trúc output bắt buộc

Mỗi notebook ghi vào:

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

## 8. Schema chi tiết

### 8.1. Checkpoint

`best.pt` và `last.pt` là dictionary gồm:

- `model_state_dict`: state của model gốc, không có tiền tố `module.` của
  wrapper DDP;
- `round`: vòng tạo checkpoint;
- `config`: cấu hình đã serialize;
- `feature_columns` và `label_mapping`;
- `validation_metrics`;
- `model_metadata`: tên kiến trúc, số tham số, raw model-state bytes.

`last.pt` được ghi sau mọi vòng. `best.pt` chỉ được thay khi validation macro-F1
tăng.

### 8.2. `history_local_epoch`

Mỗi hàng tương ứng local epoch duy nhất của một client:

- `round`, `client_id`, `local_epoch`;
- `train_examples`, `optimizer_steps`, `sampler_padding_rows`;
- `hard_loss`, `soft_loss`, `proximal_loss`, `total_loss`;
- `epoch_seconds`.

Loss là trung bình có trọng số theo số mẫu của các batch.

### 8.3. `history_client`

Mỗi hàng tương ứng một client trong một communication round:

- số mẫu train/validation;
- loss local của một epoch;
- `local_train_seconds`, `local_validation_seconds` và
  `global_validation_seconds`;
- nhóm `local_validation_*`: local student sau một local epoch được đánh giá
  trước aggregation;
- nhóm `global_validation_*`: global model sau aggregation được đánh giá trên
  validation split của client tương ứng;
- mỗi nhóm có loss, accuracy, macro/weighted precision, recall, F1,
  multiclass macro FPR/FNR và binary attack accuracy, precision, recall, F1,
  FPR, FNR.

`local_validation_*` dùng để tính best/worst client tương tự phân tích trong
paper. `global_validation_*` cho biết global model tổng quát tới phân bố riêng
của từng client như thế nào.

### 8.4. `history_round`

Mỗi hàng tương ứng một global round:

- bốn thành phần loss local đã gộp;
- best/worst/mean local-client accuracy và best/worst local-client macro-F1;
- global validation loss và toàn bộ metric chung;
- `round_seconds`, `cumulative_training_seconds`;
- `communication_bytes_round`, `communication_mib_round`,
  `communication_bytes_cumulative`, `communication_mib_cumulative`;
- peak CUDA `allocated` và `reserved` memory riêng cho GPU 0/GPU 1, giá trị
  lớn nhất trên một GPU và tổng hai GPU, ở cả byte và MiB.

Global validation metric được tính bằng cách cộng confusion matrix và loss từ
10 validation split, không đánh giá lặp lại dữ liệu.

### 8.5. `summary.json`

Đây là file chính để ghép report, gồm:

- `run_name`, `model_family`, `status`;
- cấu hình, môi trường và metadata mô hình;
- tên GPU, VRAM tổng và compute capability của từng rank;
- thống kê train/validation/test và phân bố lớp;
- round tốt nhất và validation metric tốt nhất;
- toàn bộ final test metrics;
- runtime breakdown;
- communication estimate;
- đường dẫn tương đối của các output.

### 8.6. Classification report và confusion matrix

- `classification_report.json/csv`: precision, recall, F1 và support cho từng
  lớp, cùng macro/weighted aggregate.
- `confusion_matrix.csv`: hàng là nhãn thật, cột là nhãn dự đoán.
- `confusion_matrix.npy`: cùng ma trận ở dạng `int64`.
- `confusion_matrix.png`: ma trận chuẩn hóa theo hàng để các lớp hiếm vẫn quan
  sát được.

## 9. Cách tính chi phí truyền thông

`model_state_bytes` là tổng `numel × element_size` của tensor trong model state,
không phải kích thước file checkpoint có metadata.

Với mọi client tham gia:

$$
\mathrm{bytes\_per\_round}
=2K\times\mathrm{model\_state\_bytes}
$$

Hệ số 2 tương ứng một lượt server gửi global state và một lượt client trả local
state. Tổng:

$$
\mathrm{total\_bytes}
=R\times\mathrm{bytes\_per\_round}
$$

Output ghi cả byte chính xác và MiB, với
`1 MiB = 1.048.576 byte`. Ước tính này không bao gồm TCP/TLS, serialization,
optimizer state, log, retry hoặc compression.

Ngoài chi phí FL logic ở trên, `communication_costs.json` ghi riêng ước tính
payload DDP gradient all-reduce. Với ring all-reduce:

$$
\mathrm{bytes\_per\_rank\_per\_step}
\approx 2\frac{P-1}{P}\times\mathrm{trainable\_parameter\_bytes}
$$

với `P=2`. Tổng DDP estimate nhân tiếp cho hai rank và tổng số optimizer step
thực tế. Con số này không bao gồm bucket padding, broadcast buffer hay overhead
giao thức NCCL, và không được cộng lẫn vào chi phí truyền thông FL.

## 10. Runtime và khả năng so sánh

`runtime_breakdown.json` tách:

- `preprocessing_seconds`: kiểm tra CSV, chuyển memmap và tạo split;
- `training_and_validation_seconds`: toàn bộ 10 round;
- `final_test_seconds`: suy luận trên global test bằng `best.pt`;
- `plotting_seconds`: tạo, lưu và render toàn bộ biểu đồ;
- `ddp_runtime_seconds`: thời gian trong entry point DDP;
- `torchrun_wall_seconds`: thời gian wall-clock của toàn bộ subprocess
  `torchrun`, gồm khởi tạo process group;
- `total_notebook_pipeline_seconds`: tổng từ khi bắt đầu preprocessing đến khi
  hoàn tất biểu đồ và ghi metric cuối;
- tổng thời gian local training, local-student validation và
  global-on-client validation;
- thời gian chi tiết theo từng round và từng client.

Thời gian có thể khác giữa các phiên Kaggle do tải hệ thống và cache I/O. Khi
so sánh các kiến trúc, cần giữ nguyên accelerator, dataset version, batch size,
workers, số round và local epoch. Không so sánh trực tiếp số liệu của notebook
chạy full-data với một notebook đã lấy mẫu.
