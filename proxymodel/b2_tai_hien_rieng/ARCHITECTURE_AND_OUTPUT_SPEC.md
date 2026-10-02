# FD-IDS trên CICIoT2023: đặc tả kiến trúc, khởi tạo và đầu ra

## 1. Mục đích và phạm vi

Tài liệu này là hợp đồng đo lường chung cho ba thí nghiệm:

| Notebook | `run_name` | Mô hình cá nhân hóa | Proxy |
|---|---|---|---|
| 10 GRU | `fd_ids_ciciot2023_10c_gru` | 10 GRU hai tầng | CNN-1D |
| 10 Transformer | `fd_ids_ciciot2023_10c_transformer` | 10 Transformer encoder hai tầng | CNN-1D |
| 5 GRU + 5 Transformer | `fd_ids_ciciot2023_10c_mixed_5gru_5transformer` | client 1–5 GRU, client 6–10 Transformer | CNN-1D |

Mỗi client giữ hai mô hình: một mô hình cá nhân hóa không rời thiết bị và một
proxy CNN-1D đồng nhất giữa các client. Hai mô hình được cập nhật bằng
**adaptive mutual distillation** theo (17)–(19) của paper. Chỉ trọng số proxy
được gửi lên server và tổng hợp bằng FedAvg có trọng số. Không tổng hợp trực
tiếp GRU/Transformer và không dùng FedProx.

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
| Batch size mỗi client update | 1024 |
| Gradient accumulation | Không (`1` step) |
| GPU worker | 2 process cố định, 1 process/GPU |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Scheduler | Không |
| Personalized/proxy label loss | Cross-entropy |
| Distillation discrepancy | MSE giữa hai phân phối softmax |
| Adaptive denominator epsilon `ε` | `1e-8` |
| Class weighting/resampling | Không |
| Checkpoint criterion | Validation macro-F1 lớn nhất |
| Seed | 42 |
| Thiết bị | Đúng 2 GPU T4 |
| Multi-GPU | Client-parallel, không DDP |
| Process topology | CPU coordinator + 2 GPU worker cố định |
| Launcher | Python subprocess + multiprocessing `spawn` |
| Phân hoạch client | Exact two-way partition theo benchmark workload |
| GPU-resident cache | Tối đa 70% VRAM/GPU, deterministic LRU fallback |
| CUDA stream | Benchmark tự động `[1, 2]`, giữ phương án nhanh hơn |
| Mixed precision | CUDA AMP |

Mỗi client update dùng batch 1024 trên đúng một GPU; batch không bị chia giữa
hai GPU. Không dùng sampler padding và mỗi mẫu train xuất hiện đúng một lần
trong local epoch. Adam tiếp tục dùng learning rate 0,001.

Mỗi GPU sở hữu cố định một tập client trong toàn bộ phiên. Trước khi phân
hoạch, hai worker benchmark thời gian một optimizer step cho từng họ mô hình.
Với 10 client, coordinator duyệt toàn bộ phân hoạch hai phía không rỗng và chọn
phương án có predicted makespan nhỏ nhất. Workload của client bằng
`ceil(train_examples / 1024) × measured_seconds_per_step`.

Dữ liệu, split index và personalized model của các client được giữ trên GPU sở
hữu qua các round khi tổng persistent cache không vượt 70% VRAM. Nếu không đủ,
worker dùng deterministic LRU và nạp theo chunk pinned-memory. Worker benchmark
một và hai CUDA stream bằng cùng batch 1024; chỉ dùng hai stream nếu throughput
tổng thực đo cao hơn. Không thay đổi loss, learning rate hay effective batch.

Trong mỗi vòng, mỗi client nạp global proxy mới nhất nhưng tiếp tục từ trọng số
mô hình cá nhân hóa của chính client đó. Personalized model và proxy học đồng
thời trong một local epoch. Client chỉ gửi proxy state về server. Server tính
trung bình có trọng số theo số mẫu local train thực tế rồi phân phối global
proxy mới cho vòng sau. Sau vòng cuối, mỗi personalized model được fine-tune
thêm một local epoch bằng cross-entropy; proxy không thay đổi trong bước này.

Notebook ghi entry point Python độc lập vào `/kaggle/temp` rồi khởi chạy bằng
Python subprocess thông thường. Entry point là CPU coordinator và tạo đúng hai
worker bằng multiprocessing `spawn`; worker 0 gắn `cuda:0`, worker 1 gắn
`cuda:1`. Chỉ coordinator thực hiện FedAvg, checkpoint, metric và artifact.
Proxy dùng BatchNorm thông thường trên batch đầy đủ 1024 của từng client.

### 3.1. Hợp đồng khởi tạo để so sánh công bằng

- `seed = initialization_seed = 42`.
- Mỗi lần tạo trọng số ban đầu của một họ mô hình phải chạy trong RNG fork độc
  lập và gọi lại `torch.manual_seed(42)`. Vì vậy mọi GRU ban đầu giống nhau,
  mọi Transformer ban đầu giống nhau và global proxy CNN-1D ban đầu giống nhau
  trong cả ba notebook.
- Client trong kịch bản hỗn hợp nhận đúng initial state của cùng họ mô hình
  trong hai kịch bản đồng nhất.
- Initial state được lưu trong checkpoint theo từng họ mô hình. SHA-256 của
  tensor state theo thứ tự khóa được ghi vào `config.json`, checkpoint và
  `summary.json`; notebook phải assert các hash mong đợi giữa các lần khởi tạo.
- Seed huấn luyện được suy ra từ `(seed, round, client_id, phase)`, không phụ
  thuộc GPU sở hữu; không được làm thay đổi initial state.

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

### 4.3. Label loss cho bài toán phân loại

$$
\ell_s=\operatorname{CE}(y,z_s),\qquad
\ell_r=\operatorname{CE}(y,z_r)
$$

`s` là personalized model, `r` là proxy, và cả hai model trả logits 34 lớp.

### 4.4. Adaptive mutual distillation

$$
d(z_s,z_r)=
\operatorname{MSE}\left(
\operatorname{softmax}(z_s),\operatorname{softmax}(z_r)
\right)
$$

$$
\ell_d=
\frac{d(z_s,z_r)}
{\operatorname{stopgrad}(\ell_s+\ell_r)+\varepsilon},
\qquad \varepsilon=10^{-8}
$$

$$
L_{\mathrm{train},s}=\ell_s+\ell_d,\qquad
L_{\mathrm{train},r}=\ell_r+\ell_d
$$

Hai model cùng nhận gradient từ `ℓd`. Denominator được detach để chỉ đóng vai
trò trọng số thích ứng. Một backward chung dùng
`ℓs + ℓr + ℓd`, cho gradient của mỗi model đúng bằng label loss riêng cộng một
lần gradient distillation.

## 5. Kiến trúc mô hình

Tất cả mô hình nhận tensor `[batch, 25]` và trả về logits `[batch, 34]`. Không
đặt softmax trong model. GRU/Transformer là personalized model; CNN-1D luôn là
proxy được tổng hợp.

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
│   └── rank_1.log      # tên tương thích; log chẩn đoán GPU worker 1
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

- `model_state_dict`: global proxy CNN-1D state, không có tiền tố `module.`;
- `personalized_model_state_dicts`: state của 10 client tại vòng checkpoint;
- `initial_model_state_dicts_by_family` và `initialization_hashes`;
- `round`: vòng tạo checkpoint;
- `config`: cấu hình đã serialize;
- `feature_columns` và `label_mapping`;
- `validation_metrics`;
- `model_metadata`: metadata proxy và personalized model theo họ.

`last.pt` được ghi sau mọi vòng. `best.pt` chỉ được thay khi validation macro-F1
tăng. Sau khi nạp lại best round và fine-tune personalized model, `best.pt`
giữ state trước fine-tune trong
`personalized_model_state_dicts_before_finetune` và state triển khai cuối trong
`personalized_model_state_dicts`. `last.pt` vẫn biểu diễn đúng trạng thái cuối
communication round 10.

### 8.2. `history_local_epoch`

Mỗi hàng tương ứng local epoch duy nhất của một client:

- `phase`, `round`, `client_id`, `local_epoch`; `phase` là
  `federated_mutual_learning` hoặc `final_personalized_finetune`;
- `train_examples`, `optimizer_steps`, `sampler_padding_rows`;
- `hard_loss`, `soft_loss`, `proximal_loss`, `total_loss` là các cột tương
  thích report cũ, lần lượt ánh xạ tới `personalized_label_loss`,
  `adaptive_distillation_loss`, hằng `0`, và
  `personalized_total_loss`;
- thêm `proxy_label_loss`, `personalized_total_loss`, `proxy_total_loss`;
- `epoch_seconds`.

Loss là trung bình có trọng số theo số mẫu của các batch.
Trong client-parallel, `sampler_padding_rows` luôn bằng `0`; mỗi hàng bổ sung
`gpu_id`, `concurrent_streams` và `gpu_cache_mode`.

### 8.3. `history_client`

Mỗi hàng tương ứng một client trong một communication round:

- số mẫu train/validation;
- loss local của một epoch;
- `local_train_seconds`, `local_validation_seconds` và
  `global_validation_seconds`;
- nhóm `local_validation_*`: personalized model sau một local epoch được đánh giá
  trước aggregation;
- nhóm `global_validation_*`: global proxy sau aggregation được đánh giá trên
  validation split của client tương ứng;
- mỗi nhóm có loss, accuracy, macro/weighted precision, recall, F1,
  multiclass macro FPR/FNR và binary attack accuracy, precision, recall, F1,
  FPR, FNR.

`local_validation_*` đánh giá personalized model và dùng để tính best/worst
client. `global_validation_*` đánh giá global proxy trên validation split của
client tương ứng.

### 8.4. `history_round`

Mỗi hàng tương ứng một global round:

- các loss local đã gộp, gồm đủ các alias tương thích và loss proxy;
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

- `run_name`, `model_family`, `proxy_model_family`, `status`;
- cấu hình, môi trường và metadata mô hình;
- tên GPU, VRAM tổng và compute capability của từng worker;
- thống kê train/validation/test và phân bố lớp;
- round tốt nhất và validation metric tốt nhất;
- toàn bộ final test metrics của global proxy;
- test metrics của từng personalized model và mean/best/worst theo client;
- hash và initial state contract dùng để đối chiếu giữa phương pháp;
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

`model_state_bytes` là tổng `numel × element_size` của global proxy state,
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

GRU/Transformer không được tính vào FL upload/download vì không rời client.
Không có gradient all-reduce trong client-parallel.
`communication_costs.json` vẫn giữ key
`ddp_gradient_allreduce_estimate` để tương thích report cũ nhưng ghi
`enabled=false`, byte/MiB bằng `0`. Payload tạm giữa worker và coordinator được
ghi riêng, không cộng vào logical FL communication.

## 10. Runtime và khả năng so sánh

`runtime_breakdown.json` tách:

- `preprocessing_seconds`: kiểm tra CSV, chuyển memmap và tạo split;
- `training_and_validation_seconds`: toàn bộ 10 round;
- `final_test_seconds`: suy luận trên global test bằng `best.pt`;
- `plotting_seconds`: tạo, lưu và render toàn bộ biểu đồ;
- `client_parallel_runtime_seconds`: thời gian trong entry point;
- `subprocess_wall_seconds`: thời gian wall-clock của subprocess;
- `total_notebook_pipeline_seconds`: tổng từ khi bắt đầu preprocessing đến khi
  hoàn tất biểu đồ và ghi metric cuối;
- tổng thời gian local training, personalized-model validation và
  global-on-client validation;
- benchmark workload/stream, GPU cache load, predicted/actual worker load,
  worker idle time và mẫu utilization/memory từ hai GPU;
- thời gian chi tiết theo từng round và từng client.

Thời gian có thể khác giữa các phiên Kaggle do tải hệ thống và cache I/O. Khi
so sánh các kiến trúc, cần giữ nguyên accelerator, dataset version, batch size,
workers, số round và local epoch. Không so sánh trực tiếp số liệu của notebook
chạy full-data với một notebook đã lấy mẫu.
