# pFedES trên CICIoT2023: đặc tả kiến trúc, huấn luyện và đầu ra

## 1. Mục đích và phạm vi

Tài liệu này là hợp đồng chung cho ba thí nghiệm pFedES:

| Folder / notebook | `run_name` | Local personalized model |
|---|---|---|
| `01_10_clients_gru/pfedes_10_clients_gru.ipynb` | `fd_ids_ciciot2023_10c_gru` | 10 GRU |
| `02_10_clients_transformer/pfedes_10_clients_transformer.ipynb` | `fd_ids_ciciot2023_10c_transformer` | 10 Transformer |
| `03_5_gru_5_transformer/pfedes_5_gru_5_transformer.ipynb` | `fd_ids_ciciot2023_5gru_5transformer` | client 1–5 GRU, client 6–10 Transformer |

Phương pháp huấn luyện phải bám sát pFedES trong `00121-YiL.md`:

- server chỉ chia sẻ một proxy feature extractor nhỏ, đồng nhất;
- mỗi client giữ local model cá nhân hóa qua các communication round;
- client lần lượt đóng băng proxy để cập nhật local model, rồi đóng băng local
  model để cập nhật proxy;
- chỉ local proxy được upload và tổng hợp theo số mẫu train;
- local heterogeneous model không tham gia logical communication;
- cuối cùng chỉ các personalized local model được dùng để phân loại.

Không dùng server pretrain, historical teacher, knowledge distillation, mutual
distillation, FedProx hoặc tổng hợp trực tiếp GRU/Transformer. Các notebook
không chứa lại bảng kết quả MNIST/CIFAR của paper.

## 2. Hợp đồng dữ liệu

### 2.1. Đường dẫn và file

Thư mục Kaggle chính xác:

```text
/kaggle/input/datasets/odixe0502/data-for-10clients/
```

Các file bắt buộc:

- `client_1_train.csv` đến `client_10_train.csv`: dữ liệu cục bộ của 10 client;
- `global_train_data.csv`: được kiểm tra header/số dòng để nghiệm thu dataset,
  nhưng không dùng để huấn luyện vì pFedES không có server pretrain;
- `global_test_data.csv`: tập test độc lập, chỉ dùng sau khi chọn checkpoint;
- `label_mapping.csv`: ánh xạ đủ 34 nhãn.

### 2.2. Task, feature và label

- Bài toán: phân loại đa lớp.
- Số lớp: 34.
- Label là số nguyên trong `[0,33]`.
- Input local model và proxy: float32 `[batch,25]`.
- Output local model: logits float32/AMP `[batch,34]`.
- Không đặt softmax trong model.
- Thứ tự đặc trưng cố định:

```text
ack_flag_number, AVG, Std, UDP, fin_count, Max, TCP, syn_count,
Protocol Type, Rate, IAT, syn_flag_number, rst_flag_number, Tot sum,
HTTPS, ack_count, fin_flag_number, HTTP, rst_count, Header_Length,
psh_flag_number, ICMP, Time_To_Live, ARP, DNS
```

Notebook kiểm tra chính xác file, header, thứ tự cột, shape, dtype, NaN/Inf và
miền label. Dữ liệu đã được `QuantileTransformer` chuẩn hóa ở upstream nên
notebook không fit lại scaler.

### 2.3. Split

Mỗi file client được tách độc lập theo lớp:

- 95% local train;
- 5% local validation;
- lớp chỉ có một mẫu được giữ trong local train;
- mỗi lớp có từ hai mẫu trở lên giữ ít nhất một mẫu train;
- seed split là 42;
- mọi mẫu client xuất hiện chính xác một lần trong train hoặc validation;
- `global_test_data.csv` không tham gia chọn checkpoint.

Các mảng trung gian được ghi dưới dạng NumPy memmap tại:

```text
/kaggle/temp/{run_name}_cache/
```

Cache này chỉ tồn tại trong phiên Kaggle.

## 3. Cấu hình huấn luyện đã khóa

| Thành phần | Giá trị |
|---|---:|
| Số client | 10 |
| Client tham gia mỗi round | đủ 10 (`C=100%`) |
| Communication rounds | 10 |
| Local-model epochs mỗi round | 1 |
| Proxy-extractor epochs mỗi round | 1 |
| Server pretrain | Không |
| Batch size mỗi optimizer update | 1024 |
| Gradient accumulation | Không (`1` step) |
| Optimizer local/proxy | SGD |
| Learning rate `η_ω=η_θ` | `0.01` |
| Momentum | `0` |
| Weight decay | `0` |
| Scheduler | Không |
| Loss | Cross-entropy |
| Enhanced-data loss weight `μ` | `0.1` |
| Original-data loss weight | `0.9` |
| Class weighting/resampling | Không |
| Seed và initialization seed | 42 |
| Checkpoint criterion | Mean personalized validation macro-F1 lớn nhất |
| Thiết bị | Đúng 2 GPU NVIDIA T4 |
| Multi-GPU | Client-parallel |
| Worker | 2 process cố định, 1 process/GPU |
| GPU cache safety fraction | Tối đa 70% VRAM/GPU |
| CUDA stream candidates | Benchmark `[1,2]` |
| Runtime stream | 1/client tại một thời điểm để khóa dropout RNG |
| Mixed precision | CUDA AMP |

Batch 1024 là batch của một client update trên đúng một GPU, không phải global
batch chia qua hai GPU. Không sampler padding hoặc silent subsampling.

## 4. Thuật toán pFedES được triển khai

### 4.1. Mục tiêu

Với local model heterogeneous/personalized `F_k(ω_k)` và proxy feature
extractor homogeneous `G(θ)`, mục tiêu là:

$$
\min_{\theta,\omega_0,\ldots,\omega_{N-1}}
\sum_{k=0}^{N-1}\mathcal{L}_k(
\{\mathcal{G}(\theta),\mathcal{F}_k(\omega_k)\};D_k).
$$

Tại đầu round `t`, cả 10 client nhận cùng global proxy
`G(θ^{t-1})`. Local model không khởi tạo lại và không nhận model từ server.

### 4.2. Đóng băng proxy, cập nhật local model

Với batch `(x,y)` và enhanced data giữ nguyên chiều:

$$
\hat{x}=\mathcal{G}(\theta^{t-1};x),
$$

$$
\hat{y}_1=\mathcal{F}_k(\omega_k^{t-1};\hat{x}),\qquad
\hat{y}_2=\mathcal{F}_k(\omega_k^{t-1};x),
$$

$$
\ell_1=\operatorname{CE}(\hat{y}_1,y),\qquad
\ell_2=\operatorname{CE}(\hat{y}_2,y),
$$

$$
\ell_\omega=\mu\ell_1+(1-\mu)\ell_2,
\qquad \mu=0.1,
$$

$$
\omega_k^t\leftarrow
\omega_k^{t-1}-\eta_\omega\nabla\ell_\omega.
$$

Proxy chạy `eval()`, `requires_grad=False` và enhanced tensor được detach trong
bước này. Local model chạy `train()`.

### 4.3. Đóng băng local model, cập nhật proxy

Local model vừa cập nhật vẫn chạy `train()` nhưng toàn bộ tham số bị khóa bằng
`requires_grad=False`. Gradient vẫn phải truyền từ loss qua local model về
output của proxy; cách tách trạng thái module khỏi trạng thái trainable này là
bắt buộc với GRU/LSTM/RNN chạy bằng cuDNN, vì backward theo input chỉ hợp lệ
khi forward của RNN được thực hiện trong training mode. Local optimizer được
`zero_grad(set_to_none=True)` trước pha này nên không lưu hoặc cập nhật gradient
local. Với Transformer, dropout của local model vẫn hoạt động và RNG được tái
lập theo `(seed, round, client_id, phase)` trên một CUDA stream để bảo đảm tính
tái lập:

$$
\hat{y}=\mathcal{F}_k(\omega_k^t;
\mathcal{G}(\theta^{t-1};x)),
$$

$$
\ell_\theta=\operatorname{CE}(\hat{y},y),
$$

$$
\theta_k^t\leftarrow
\theta^{t-1}-\eta_\theta\nabla\ell_\theta.
$$

Mỗi client tạo local proxy từ cùng round-start global proxy; không tiếp tục từ
local proxy riêng của round trước.

### 4.4. Tổng hợp proxy

Sau khi đủ 10 client hoàn thành cả hai bước, coordinator tổng hợp trên CPU:

$$
\theta^t=
\sum_{k\in\mathcal{S}^t}
\frac{n_k}{\sum_{j\in\mathcal{S}^t}n_j}\theta_k^t,
\qquad \mathcal{S}^t=\{1,\ldots,10\}.
$$

`n_k` là số local-train example sau split 95/5. Không tổng hợp local model.

### 4.5. Trạng thái cá nhân hóa

- round 1: mỗi local model bắt đầu từ initial state xác định bởi seed 42;
- round sau: local model tiếp tục từ personalized state sau round trước;
- global proxy round 1 bắt đầu từ proxy initial state seed 42;
- chỉ aggregation proxy tạo global state cho round kế tiếp;
- file checkpoint ghi local states để resume/so sánh trong mô phỏng; việc ghi
  checkpoint này không được tính là logical federated communication.

## 5. Kiến trúc mô hình

### 5.1. Proxy feature extractor dùng chung

Proxy ảnh `Conv 3→8→3` trong paper được chuyển sang một chiều, giữ đúng nguyên
tắc input và enhanced data cùng shape:

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều channel | `[B,1,25]` |
| Conv1 | `[B,1,25]` | `1→8`, kernel 3, padding 1 | `[B,8,25]` |
| ReLU | `[B,8,25]` | activation giữa hai conv | `[B,8,25]` |
| Conv2 | `[B,8,25]` | `8→1`, kernel 3, padding 1 | `[B,1,25]` |
| Squeeze | `[B,1,25]` | bỏ channel | `[B,25]` |

Không activation sau Conv2 để giữ miền giá trị âm/dương của dữ liệu đã chuẩn
hóa. Số tham số trainable bắt buộc: **57**.

### 5.2. GRU

25 đặc trưng được xem như chuỗi dài 25, mỗi bước chứa một giá trị.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Reshape | `[B,25]` | thêm chiều cuối | `[B,25,1]` |
| GRU | `[B,25,1]` | 2 tầng, hidden 64, dropout 0,2 | `[B,25,64]` |
| Last-step | `[B,25,64]` | lấy bước cuối | `[B,64]` |
| Linear | `[B,64]` | `64→34` | `[B,34]` |

Số tham số trainable bắt buộc: **40.034**.

### 5.3. Transformer

Mỗi đặc trưng là một token vô hướng. Giá trị được chiếu lên 64 chiều và cộng
learned positional embedding.

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Scalar projection | `[B,25,1]` | Linear `1→64` | `[B,25,64]` |
| Positional embedding | `[B,25,64]` | learned, chiều dài 25 | `[B,25,64]` |
| Encoder ×2 | `[B,25,64]` | 4 heads, FFN 128, dropout 0,1 | `[B,25,64]` |
| Mean pooling | `[B,25,64]` | trung bình theo token | `[B,64]` |
| LayerNorm + Linear | `[B,64]` | `64→34` | `[B,34]` |

Số tham số trainable bắt buộc: **71.010**.

### 5.4. Khởi tạo công bằng

- `seed = initialization_seed = 42`;
- mỗi họ local model dùng RNG fork riêng và `torch.manual_seed(42)`;
- mọi client cùng họ bắt đầu từ cùng initial state;
- proxy cũng bắt đầu từ một initial state xác định bởi seed 42;
- SHA-256 của ordered tensor state được ghi cho initial states, initial proxy,
  global proxy từng checkpoint và personalized states cần so sánh;
- seed train suy ra từ `(seed, phase, round, client_id)`, không phụ thuộc GPU.

## 6. Tối ưu hai GPU T4

Notebook dùng CPU coordinator và đúng hai worker `multiprocessing` với start
method `spawn`:

- worker 0 gắn `cuda:0`;
- worker 1 gắn `cuda:1`;
- không dùng gradient all-reduce hoặc ghép batch qua GPU;
- aggregation, checkpoint, consolidated metrics và plot chỉ do coordinator;
- worker chỉ ghi payload tạm riêng và log riêng;
- traceback, task thiếu/trùng và nonzero exit code đều làm notebook fail.

### 6.1. Phân hoạch workload

Hai worker benchmark optimizer step thật cho mọi model family đang hoạt động.
Workload dự đoán của client gồm cả hai pha:

$$
\left\lceil\frac{n_k^{train}}{1024}\right\rceil
\left(t_{local-step}+t_{proxy-step}\right).
$$

Coordinator duyệt mọi bipartition hai phía không rỗng của 10 client, chọn
predicted makespan nhỏ nhất và tie-break theo client ID. Assignment giữ cố định
trong toàn bộ phiên.

### 6.2. Cache GPU

- cache feature, label và split indices trên GPU sở hữu;
- tổng persistent cache không vượt 70% VRAM/GPU;
- dữ liệu được copy theo chunk vào tensor preallocated;
- nếu không đủ, dùng deterministic whole-client LRU và DataLoader pinned-memory
  non-blocking fallback;
- không giảm hoặc lấy mẫu dữ liệu;
- ghi cache bytes, hit, miss, eviction, fallback và peak memory.

### 6.3. CUDA stream và AMP

- benchmark 1 và 2 independent CUDA streams bằng model/batch thật;
- ghi throughput của cả hai candidate;
- GRU/Transformer đều có dropout nên runtime khóa 1 stream/GPU sau benchmark
  để seed client không phụ thuộc lịch xen kẽ kernel;
- mỗi client có model, optimizer, scaler, stream và RNG riêng;
- permutation và loss accumulator được tạo trong đúng stream context;
- đồng bộ ở phase/task boundary, không đọc CPU trong batch loop;
- dùng CUDA AMP, `GradScaler`, TF32 khi T4/PyTorch hỗ trợ;
- không dùng CUDA Graphs.

## 7. Validation, checkpoint và test

### 7.1. Validation và checkpoint

Sau hai bước local training, mỗi personalized model được đánh giá trên local
validation bằng original feature, không đi qua proxy. Mỗi round ghi:

- metric riêng của đủ 10 client;
- mean không trọng số của 10 personalized metrics;
- pooled validation confusion matrix bằng tổng raw confusion matrix của các
  local validation split không chồng lặp.

`best.pt` chọn theo mean personalized validation macro-F1 lớn nhất. `last.pt`
luôn là state sau round gần nhất và sau khi hoàn tất là round 10.

Notebook bật `resume_if_available=true`. Khi `last.pt` đã tồn tại trong output,
runtime kiểm tra run/model assignment, nạp global proxy, đủ 10 personalized
states và các history tương ứng rồi tiếp tục từ round kế. Để resume từ một
Kaggle dataset/output phiên trước, đặt `resume_checkpoint_path` tới `last.pt`;
file `best.pt` và thư mục `metrics/` phải nằm cạnh cấu trúc checkpoint đó.

### 7.2. Final test

Sau round 10:

1. nạp lại `best.pt`;
2. mỗi personalized model đánh giá toàn bộ `global_test_data.csv` đúng một lần;
3. metric so sánh pFedES chính là mean không trọng số của 10 client;
4. report/confusion matrix tương thích chính là pooled personalized
   predictions, tức cộng raw confusion matrix của 10 lần đánh giá;
5. lưu metric và report từng client riêng;
6. không tạo hoặc gán ngữ nghĩa cho một global classifier/ensemble không tồn
   tại trong pFedES.

## 8. Quy ước metric

Mọi tỷ lệ JSON/CSV nằm trong `[0,1]`. Thời gian dùng giây.

Với từng lớp `c`:

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\qquad
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c},
$$

$$
\mathrm{F1}_c=2\frac{\mathrm{Precision}_c\mathrm{Recall}_c}
{\mathrm{Precision}_c+\mathrm{Recall}_c},
$$

$$
\mathrm{FPR}_c=\frac{FP_c}{FP_c+TN_c},\qquad
\mathrm{FNR}_c=\frac{FN_c}{FN_c+TP_c}.
$$

- `macro_*`: trung bình không trọng số trên đủ 34 lớp, zero denominator → 0;
- `weighted_*`: trung bình theo support nhãn thật;
- `accuracy`: trace confusion matrix chia tổng mẫu;
- `multiclass_macro_fpr/fnr`: one-vs-rest trung bình 34 lớp;
- `binary_attack_*`: BENIGN là negative, 33 lớp còn lại là attack positive.

## 9. Cấu trúc output bắt buộc

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
│   ├── personalized_test_metrics.json
│   ├── personalized_test_metrics.csv
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

## 10. Schema checkpoint và history

### 10.1. Checkpoint

`best.pt` và `last.pt` là dictionary gồm:

- `model_state_dict`: compatibility alias của global proxy state;
- `global_proxy_state_dict`;
- `personalized_model_state_dicts`: state đủ 10 local model;
- `personalized_model_state_dicts_before_finetune`: compatibility alias bằng
  personalized states vì không có final fine-tune;
- `historical_teacher_model_state_dicts={}`: compatibility key, không có
  historical teacher trong pFedES;
- `initial_model_state_dicts_by_family`;
- `initialization_hashes`, `initial_proxy_hash`, `global_state_hash`;
- `round`, selected client fields luôn đủ 10;
- `selection_threshold_accuracy=null`: compatibility field, pFedES không chọn
  client theo threshold;
- `local_accuracies`, `config`, `feature_columns`, `label_mapping`;
- `validation_metrics`, `model_metadata`.

Không được diễn giải `model_state_dict` như classifier state.

### 10.2. `history_pretrain`

Giữ một hàng để tương thích output cũ:

- `enabled=false`, `epoch=0`;
- `phase="not_applicable_pfedes_has_no_server_pretrain"`;
- examples, optimizer steps, loss, runtime và memory bằng 0.

Hàng này không được mô tả như một bước huấn luyện.

### 10.3. `history_local_epoch`

Mỗi round đúng 10 hàng, tổng 100 hàng:

- `phase="pfedes_iterative_training"`, `round`, `client_id`, model family;
- local/proxy optimizer steps, train examples, padding rows bằng 0;
- enhanced/original/combined local loss và proxy CE loss;
- `hard_loss` alias combined local loss;
- compatibility `soft_loss=0`, `proximal_loss=0`;
- `μ`, seconds, GPU, stream và cache mode.

### 10.4. `history_client`

Mỗi round đúng 10 hàng, tổng 100 hàng:

- client/model/split size;
- received global proxy và uploaded local proxy flags;
- local/proxy loss và runtime;
- local validation loss và đầy đủ multiclass/binary metrics;
- GPU assignment, cache, stream và peak memory.

### 10.5. `history_round`

Đúng 10 hàng:

- sample-weighted mean local/proxy training loss;
- mean/min/max personalized validation accuracy và fairness standard deviation;
- mean personalized validation macro-F1;
- pooled personalized validation fields được giữ dưới compatibility name
  `global_validation_*`;
- đủ 10 selected clients;
- round/cumulative time;
- exact proxy communication byte/MiB;
- peak memory từng GPU;
- predicted/actual worker load và idle time.

### 10.6. `summary.json`

File ghép report chính gồm:

- run/method/scenario/status và serialized config;
- environment, model metadata và state hashes;
- dataset/split/class distribution;
- best round/validation criterion;
- assignment, benchmark, cache, utilization và memory;
- final pooled personalized test metrics;
- final mean personalized test metrics;
- final metric/report từng client;
- runtime, communication và relative output paths.

## 11. Classification report và confusion matrix

- `classification_report.json/csv` và `confusion_matrix.*` là pooled
  personalized predictions trên global test;
- mỗi global-test row được dự đoán đúng một lần bởi mỗi trong 10 local models,
  nên pooled confusion support bằng `10 × global_test_rows`;
- report có precision, recall, F1 và support từng lớp cùng macro/weighted avg;
- confusion matrix là `int64`, hàng nhãn thật và cột nhãn dự đoán;
- plot confusion matrix chuẩn hóa theo hàng;
- metric/report riêng từng client nằm trong `personalized_test_metrics.*`;
- mean personalized metrics trong `summary.json` là số chính để so sánh pFedES
  với các phương pháp personalized khác.

## 12. Chi phí truyền thông

Chỉ proxy state được truyền. Với:

$$
\mathrm{proxy\_state\_bytes}=
\sum_{\theta\in\mathrm{proxy\ state}}
\operatorname{numel}(\theta)\operatorname{element\_size}(\theta),
$$

chi phí mỗi round là:

$$
\mathrm{bytes}_t=2|\mathcal{S}^t|\times
\mathrm{proxy\_state\_bytes},\qquad |\mathcal{S}^t|=10.
$$

Hệ số 2 gồm global proxy download và local proxy upload. Không tính local
model checkpoint, optimizer, filesystem payload, TCP/TLS, serialization, log,
retry hoặc cache. `1 MiB=1.048.576 byte`.

`ddp_gradient_allreduce_estimate` vẫn tồn tại để tương thích report nhưng phải
có `enabled=false`, byte/MiB bằng 0.

## 13. Runtime và logging

`runtime_breakdown.json` tách:

- data validation/memmap/split;
- server pretrain bằng 0 và disabled;
- workload/stream benchmark;
- per-round/per-worker runtime và idle;
- final personalized test;
- plotting;
- subprocess wall time;
- tổng entry-point và notebook pipeline.

Ghi thêm:

- Python, PyTorch, CUDA, GPU name, VRAM và compute capability;
- peak allocated/reserved memory;
- cache bytes/hit/miss/eviction/fallback;
- sampled GPU utilization/memory;
- worker traceback và nonzero exit propagation.

## 14. Điều kiện nghiệm thu notebook

Mỗi notebook phải:

1. parse JSON và syntax-check mọi code cell/embedded entry point;
2. assert CUDA và đúng hai GPU có tên chứa `T4`;
3. assert input path/file/header/shape/dtype/NaN/Inf/label;
4. assert output `[B,34]` và exact parameter count GRU/Transformer/proxy;
5. dùng một serialized `CONFIG`;
6. ghi entry point độc lập dưới `/kaggle/temp`;
7. launch bằng subprocess và propagate lỗi;
8. dùng đúng client-parallel topology và AMP;
9. có deterministic initialization/training seed;
10. triển khai đúng hai pha đóng băng/cập nhật của pFedES;
11. cùng round-start global proxy cho đủ 10 client;
12. chỉ aggregate proxy, có trọng số local-train examples;
13. giữ personalized local state qua round;
14. kiểm tra row count: pretrain compatibility 1, round 10, client 100,
    local epoch 100;
15. tạo đủ best/last checkpoint, JSON/CSV/NPY/log/plot không rỗng;
16. kiểm tra exact validation/test accounting và proxy communication;
17. không có placeholder, silent subsampling hoặc fabricated path.

Kiểm tra local chỉ là structural validation. Chỉ được tuyên bố runtime
validation sau khi notebook thực sự chạy hoàn tất trên Kaggle `GPU T4 x2`.
