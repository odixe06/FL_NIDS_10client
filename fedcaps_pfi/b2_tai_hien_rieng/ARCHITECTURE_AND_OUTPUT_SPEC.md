# FedCAPS trên CICIoT2023: đặc tả kiến trúc, huấn luyện và đầu ra

## 1. Mục đích và phạm vi

Tài liệu này là hợp đồng chung cho ba thí nghiệm FedCAPS trên
CICIoT2023. Mỗi thí nghiệm có 10 client với downstream classifier không
đồng nhất:

| Folder / notebook | `run_name` | Phân bổ model theo client |
|---|---|---|
| `01_4gru_3transformer_3cnn1d/` | `fedcaps_ciciot2023_4gru_3transformer_3cnn1d` | client 1–4 GRU, 5–7 Transformer, 8–10 CNN-1D |
| `02_6cnn1d_3gru_1transformer/` | `fedcaps_ciciot2023_6cnn1d_3gru_1transformer` | client 1–6 CNN-1D, 7–9 GRU, 10 Transformer |
| `03_5transformer_5gru/` | `fedcaps_ciciot2023_5transformer_5gru` | client 1–5 Transformer, 6–10 GRU |

Mỗi notebook triển khai FedCAPS end-to-end theo `2510.05535v3.md`:

1. từng client thu thập 300 feature-selection record bằng MARLFS;
2. server nhận chỉ feature ID, performance và sample count, không nhận raw data;
3. mỗi record được hoán vị 25 lần để tăng cường;
4. server huấn luyện encoder–decoder bất biến hoán vị dùng ISAB, PMA,
   MAB và rFF;
5. top-25 record là search seed cho PPO;
6. critic dự đoán reward, còn client feedback thật hiệu chỉnh mỗi 100
   search step;
7. sample-aware weighting tổng hợp performance theo kích thước local train;
8. feature subset cuối cùng có kích thước do FedCAPS tự tìm, không ép
   trước số feature;
9. mỗi client huấn luyện classifier cá nhân hóa bằng subset chung đã
   chọn, sau đó đánh giá trên global test.

Không chạy lại 12 baseline CAPS, 4 federated baseline, 14 dataset gốc hoặc
các ablation của paper. GRU/Transformer/CNN-1D là downstream evaluator;
tham số của chúng không được FedAvg hay truyền lên server.

## 2. Hợp đồng dữ liệu

### 2.1. Đường dẫn và file

Thư mục Kaggle chính xác:

```text
/kaggle/input/datasets/odixe0502/data-for-10clients/
```

File bắt buộc:

- `client_1_train.csv` đến `client_10_train.csv`;
- `global_train_data.csv`, chỉ kiểm tra header và tổng số dòng;
- `global_test_data.csv`, chỉ dùng sau khi chọn subset và checkpoint;
- `label_mapping.csv`, đủ 34 nhãn.

### 2.2. Task, feature và label

- Phân loại đa lớp, 34 lớp.
- Label là số nguyên trong `[0,33]`.
- Input gốc float32 `[batch,25]`; input classifier sau chọn là
  `[batch,selected_feature_count]`.
- Output classifier là logits `[batch,34]`; không có softmax trong model.
- Thứ tự 25 feature gốc:

```text
ack_flag_number, AVG, Std, UDP, fin_count, Max, TCP, syn_count,
Protocol Type, Rate, IAT, syn_flag_number, rst_flag_number, Tot sum,
HTTPS, ack_count, fin_flag_number, HTTP, rst_count, Header_Length,
psh_flag_number, ICMP, Time_To_Live, ARP, DNS
```

Notebook kiểm tra file, header, shape, dtype, NaN/Inf, label và accounting.
Dữ liệu đã qua `QuantileTransformer` upstream nên không fit scaler lại.

### 2.3. Split và search pool

Từng client split phân tầng, xác định bởi seed 42:

- 90% local train, 10% local validation;
- lớp chỉ có một mẫu được giữ trong local train;
- lớp có từ hai mẫu trở lên giữ ít nhất một train và một validation;
- mọi dòng xuất hiện đúng một lần trong train hoặc validation;
- test không tham gia MARLFS, PPO, chọn subset hay checkpoint.

Search pool của mỗi client là tập con phân tầng xác định, tối đa
100.000 dòng gồm train và validation theo tỷ lệ 90/10. MARLFS và
PPO client-feedback chỉ dùng pool này. Final classifier dùng toàn bộ local
train/validation, không subsample.

Memmap dùng trong phiên được ghi dưới:

```text
/kaggle/temp/{run_name}_cache/
```

## 3. Topology hai GPU T4

Notebook dùng `client_parallel`, không dùng data-parallel training:

```text
notebook
  └─ CPU coordinator / consolidated writer
       ├─ persistent worker 0 → cuda:0
       └─ persistent worker 1 → cuda:1
```

- assert CUDA và đúng hai GPU có tên chứa `T4`;
- multiprocessing start method `spawn`;
- không DDP, NCCL, `torchrun`, `DataParallel` hay gradient all-reduce;
- aggregation record/performance, PPO, consolidated metric, checkpoint và plot
  chỉ do coordinator;
- traceback, duplicate/missing result hoặc worker exit code khác 0 là lỗi;
- batch 1024 là batch của một client update trên một GPU;
- CUDA AMP, TF32 nếu được hỗ trợ;
- benchmark optimizer step thật theo từng model family;
- duyệt chính xác mọi bipartition không rỗng của 10 client và chọn
  assignment có predicted makespan nhỏ nhất;
- benchmark 1 và 2 CUDA stream và ghi cả throughput/recommendation; vì cả ba
  kịch bản đều có GRU/Transformer với dropout, runtime khóa một stream
  mỗi GPU để RNG xác định không phụ thuộc thứ tự xen kẽ kernel;
- cache feature, label và split index của client trên GPU sở hữu, tối
  đa 70% VRAM; nếu không đủ thì deterministic whole-client LRU và
  pinned-memory non-blocking fallback;
- seed công việc suy ra từ `(42, phase, epoch, client_id)`, không phụ
  thuộc GPU hoặc thứ tự hoàn thành.

## 4. Cấu hình đã khóa

### 4.1. Thông số giữ theo paper

| Thành phần | Giá trị |
|---|---:|
| MARLFS collection epochs | 300/client |
| Permutation augmentation | 25 lần/record |
| Encoder | 2 ISAB |
| Decoder | PMA + MAB + rFF |
| Attention heads | 4 |
| Feature-ID embedding | 128 |
| Inducing points | 32 |
| PMA seed vectors | 32 |
| Encoder–decoder batch | 64 |
| Encoder–decoder learning rate | 0,001 |
| PPO initial search seeds | top 25 record |
| PPO search epochs | 10 |
| PPO batch | 512 |
| Actor learning rate | 0,0003 |
| Critic learning rate | 0,001 |
| Reward trade-off `λ` | 0,1 |
| Discount `γ` | 0,99 |
| Search steps/epoch | 1000 |
| PPO clipping `ε` | 0,2 |

### 4.2. Thông số triển khai do paper không công bố

| Thành phần | Giá trị |
|---|---:|
| Search pool | tối đa 100.000 mẫu/client |
| MARLFS candidate evaluator | 1 epoch, khởi tạo lại công bằng |
| Encoder–decoder optimizer | Adam |
| Encoder–decoder max epochs / patience | 100 / 10 |
| PPO feedback thật | mỗi 100 search step |
| Minimum candidate subset size | 2, để CNN MaxPool hợp lệ |
| Final classifier epochs | 10 |
| Final/candidate optimizer | SGD |
| Classifier learning rate | 0,01 |
| Per-client batch | 1024 |
| Scheduler / accumulation | không / 1 |
| Loss | Cross-entropy |
| Class weighting/resampling | không |
| Seed | 42 |
| Checkpoint criterion | local validation macro-F1 lớn nhất |
| GPU cache safety fraction | 0,70 |
| Stream candidates | `[1,2]` |

## 5. Phương pháp và công thức FedCAPS

### 5.1. Tối ưu toàn cục và sample-aware weighting

Với client `c`, local data `D_c`, subset `f` và downstream metric `M_c`:

$$
\mathcal W_c=\frac{|D_c|}{\sum_{j=1}^{C}|D_j|},
\qquad
\hat v(f)=\sum_{c=1}^{C}\mathcal W_c\mathcal M_c(X_c[f]).
$$

Mục tiêu:

$$
\mathbf f^*=\psi(\mathbf E^*)=
\underset{\mathbf E\in\mathcal E}{\arg\max}
\sum_{c=1}^{C}\mathcal W_c\mathcal M_c
(X_c[\psi(\mathbf E)]).
$$

`|D_c|` trong notebook là số local-train sample sau split 90/10. Metric
performance trong search là Micro-F1; với single-label multiclass, giá trị
này bằng accuracy nhưng vẫn được ghi rõ là Micro-F1.

### 5.2. MAB và ISAB encoder

$$
MAB(Q,K,V)=LayerNorm(H+rFF(H)),
$$

$$
H=LayerNorm(Q+Multihead(Q,K,V;W)).
$$

Với `M=32` inducing point:

$$
H_I=MAB(I,f,f),
\qquad
ISAB_M(f)=MAB(f,H_I,H_I),
$$

$$
\omega(f)=ISAB_M(ISAB_M(f))=E.
$$

Encoder không dùng positional encoding hay dropout. Padding token bị mask;
hoán vị feature ID phải cho reconstructed membership logits bằng nhau trong
sai số số học.

### 5.3. PMA/MAB/rFF decoder và reconstruction NLL

$$
PMA_K(E)=MAB(S,rFF(E),rFF(E)),
$$

$$
\tilde f=rFF(MAB(PMA_K(E))).
$$

Vì output là một **tập** feature có độ dài biến đổi, notebook biểu
diễn `f` bằng vector membership 25 chiều và triển khai negative
log-likelihood trong Eq. (9) dưới dạng Bernoulli set likelihood:

$$
\mathcal L_{rec}=-\log P_\psi(f|E)
=-\sum_{n=1}^{25}\left[
y_n\log p_n+(1-y_n)\log(1-p_n)\right].
$$

Cách này không ép một thứ tự tùy ý lên subset và cho phép decoder
tự sinh số feature cuối cùng. Threshold mặc đị là 0,5; nếu dưới
2 feature thì lấy top-2 logit.

### 5.4. PPO actor–critic

Actor biến đổi embedding `E` thành `E+`; decoder sinh state one-hot
`s=Rep(X[ψ(E+)])`. Critic loss và clipped actor objective:

$$
\mathcal L_{critic}=\frac1T\sum_{t=1}^{T}(V(s_t)-G_t)^2,
$$

$$
\mathcal L_{actor}=\hat{\mathbb E}_t\left[
\min\left(r_t(\theta)\hat A_t,
clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
\right].
$$

PDF in reward với dấu `+ N[f+]` nhưng đồng thời nói mục tiêu là giảm
subset length. Để thực thi đúng mục tiêu, `N` được định nghĩa là
compactness score đã chuẩn hóa:

$$
\mathcal N[f^+]=1-\frac{|f^+|}{25},
$$

$$
R=\lambda\left(\hat v(f^+)-\hat v(f)\right)
+(1-\lambda)\mathcal N[f^+],\qquad \lambda=0.1.
$$

Notebook báo cáo cả subset có reward cao nhất và subset có weighted
Micro-F1 cao nhất. `f*` dùng cho final training là subset có reward cao nhất;
tie-break lần lượt bằng weighted Micro-F1, subset ngắn hơn và danh sách
feature ID theo thứ tự từ điển.

### 5.5. Privacy và communication boundary

Client chỉ upload:

- feature ID của 300 record;
- local Micro-F1 và local sample count;
- scalar feedback cho candidate được calibration.

Server chỉ broadcast feature ID của candidate. Raw row, gradient, classifier
parameter, optimizer state và activation không vượt qua logical client/server
boundary. Việc notebook mô phỏng nhiều client trong một Kaggle process không
làm thay đổi accounting này.

## 6. Kiến trúc downstream classifier

Model nhận `[B,F]`, `2 ≤ F ≤ 25`, và trả logits `[B,34]`. Số tham số
dưới đây tính với positional capacity 25 và không thay đổi theo `F`.

### 6.1. GRU

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Reshape | `[B,F]` | thêm chiều cuối | `[B,F,1]` |
| GRU | `[B,F,1]` | 2 tầng, hidden 64, dropout 0,2 | `[B,F,64]` |
| Last-step | `[B,F,64]` | lấy bước cuối | `[B,64]` |
| Linear | `[B,64]` | `64→34` | `[B,34]` |

Số tham số trainable bắt buộc: **40.034**.

### 6.2. Transformer

| Tầng | Input | Cấu hình | Output |
|---|---|---|---|
| Scalar projection | `[B,F,1]` | Linear `1→64` | `[B,F,64]` |
| Positional embedding | `[B,F,64]` | learned capacity 25 | `[B,F,64]` |
| Encoder ×2 | `[B,F,64]` | 4 heads, FFN 128, dropout 0,1 | `[B,F,64]` |
| Mean pooling | `[B,F,64]` | mean theo token | `[B,64]` |
| LayerNorm + Linear | `[B,64]` | `64→34` | `[B,34]` |

Số tham số trainable bắt buộc: **71.010**.

### 6.3. CNN-1D

Giữ nguyên cấu hình CNN-1D từ spec đã được người dùng cập nhật:

| Khối | Input | Cấu hình | Output chính |
|---|---|---|---|
| Reshape | `[B,F]` | thêm chiều kênh | `[B,1,F]` |
| Conv block 1 | `[B,1,F]` | Conv `1→32`, kernel 3, padding 1, BN, ReLU | `[B,32,F]` |
| Conv block 2 | `[B,32,F]` | Conv `32→64`, kernel 3, padding 1, BN, ReLU, MaxPool 2 | `[B,64,⌊F/2⌋]` |
| Conv block 3 | `[B,64,⌊F/2⌋]` | Conv `64→128`, kernel 3, padding 1, BN, ReLU | `[B,128,⌊F/2⌋]` |
| Pool + Linear | `[B,128,⌊F/2⌋]` | adaptive average pool, `128→34` | `[B,34]` |

Số tham số trainable bắt buộc: **35.874**.

### 6.4. Khởi tạo công bằng

- seed và initialization seed 42;
- mỗi model family có initial state xác định riêng;
- mọi candidate và client cùng family bắt đầu từ cùng initial state;
- final personalized state không tổng hợp;
- SHA-256 của ordered state tensor được ghi trong checkpoint/summary.

## 7. Validation, final test và metric

### 7.1. Chọn checkpoint và final test

- Mỗi final client model chọn epoch có local-validation macro-F1 lớn nhất.
- Sau đó mỗi model đánh giá **toàn bộ** `global_test_data.csv` đúng
  một lần.
- Metric chính là sample-weighted mean của 10 personalized test metrics.
- Xuất thêm mean không trọng số, metric từng client và pooled confusion
  matrix của 10 lần đánh giá.
- `global_test_data.csv` không dùng để chọn epoch hay subset.

### 7.2. Quy ước metric

Mọi tỷ lệ JSON/CSV nằm trong `[0,1]`; thời gian dùng giây. Với lớp
`c`:

$$
Precision_c=\frac{TP_c}{TP_c+FP_c},\quad
Recall_c=\frac{TP_c}{TP_c+FN_c},
$$

$$
F1_c=2\frac{Precision_cRecall_c}{Precision_c+Recall_c},
$$

$$
FPR_c=\frac{FP_c}{FP_c+TN_c},\quad
FNR_c=\frac{FN_c}{FN_c+TP_c}.
$$

- `macro_*`: mean không trọng số trên đủ 34 lớp, zero denominator → 0;
- `weighted_*`: mean theo true support;
- `micro_f1`: pooled single-label Micro-F1;
- `accuracy`: trace confusion matrix chia tổng support;
- `multiclass_macro_fpr/fnr`: one-vs-rest mean;
- `binary_attack_*`: BENIGN (`Encoded_ID=1`) là negative, 33 lớp còn lại
  là attack positive.

## 8. Cấu trúc output bắt buộc

Mỗi notebook ghi:

```text
/kaggle/working/{run_name}/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── logs/
│   ├── run.log
│   ├── worker_0.log
│   └── rank_1.log
├── metrics/
│   ├── config.json
│   ├── dataset_summary.json
│   ├── client_class_distribution.csv
│   ├── feature_selection_records.csv
│   ├── feature_selection_records.json
│   ├── encoder_history.csv
│   ├── encoder_history.json
│   ├── ppo_history.csv
│   ├── ppo_history.json
│   ├── candidate_evaluations.csv
│   ├── selected_features.json
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
    ├── communication_cumulative.png
    ├── subset_size_search.png
    └── selected_feature_importance.png
```

## 9. Checkpoint và history schema

### 9.1. Checkpoint

`best.pt` và `last.pt` gồm:

- `model_state_dict`: compatibility alias; dictionary state của 10 model;
- `personalized_model_state_dicts`: đủ 10 client;
- `personalized_model_state_dicts_before_finetune`: compatibility alias bằng
  personalized states;
- `initial_model_state_dicts_by_family` và `initialization_hashes`;
- `encoder_state_dict`, `decoder_state_dict`, `actor_state_dict`,
  `critic_state_dict`;
- `selected_feature_ids`, `selected_feature_names`, `selected_feature_count`;
- `best_by_reward`, `best_by_weighted_micro_f1`;
- `config`, `feature_columns`, `label_mapping`, `model_metadata`;
- `client_best_epochs`, `client_validation_metrics`, `status`.

### 9.2. Compatibility histories

- `history_round`: đúng 10 hàng, mỗi hàng là một PPO search epoch;
  giữ field accuracy/F1/loss/runtime/communication/memory có ý nghĩa tương
  ứng với FedCAPS.
- `history_local_epoch`: đúng 100 hàng, 10 final epoch × 10 client;
  có train loss, optimizer step, validation metric, runtime, GPU, cache và
  stream.
- `history_client`: đúng 10 hàng; model family, split size, best epoch,
  local validation và final personalized test metric.

### 9.3. `summary.json`

File ghép report chính gồm:

- run/method/scenario/status và serialized config;
- environment, model metadata và initialization hash;
- dataset/split/class distribution;
- client assignment, benchmark, cache, utilization và peak memory;
- MARLFS, encoder–decoder và PPO convergence;
- selected feature ID/name/count;
- final pooled, weighted-mean và unweighted-mean personalized test metrics;
- runtime, communication và relative output paths.

## 10. Classification report và confusion matrix

- `classification_report.*` và `confusion_matrix.*` là pooled predictions
  của 10 personalized model trên global test;
- pooled support bằng `10 × global_test_rows`;
- raw confusion matrix là `int64`, hàng label thật, cột prediction;
- plot confusion matrix chuẩn hóa theo hàng;
- metric từng client nằm trong `personalized_test_metrics.*`;
- sample-weighted mean là số chính để so sánh phương pháp.

## 11. Chi phí truyền thông

Logical FedCAPS communication không tính filesystem nội bộ notebook:

- feature ID: int16, 2 byte/ID;
- performance: float32, 4 byte;
- sample count: int64, 8 byte;
- original record upload: `2×|f| + 4 + 8` byte;
- permutation augmentation ×25 diễn ra trên server, không communication;
- mỗi feedback thật: server broadcast `2×|f|` byte/client và client
  upload 4 byte scalar;
- classifier parameter/checkpoint không tính vào communication.

`communication_costs.*` ghi từng phase, cumulative byte/MiB và compatibility
field `ddp_gradient_allreduce_estimate` với `enabled=false`, byte bằng 0.

## 12. Runtime, logging và nghiệm thu

`runtime_breakdown.json` tách data validation/memmap/split, benchmark/cache,
MARLFS, augmentation, encoder–decoder, PPO/feedback, final training, final test,
plotting, subprocess wall time và tổng pipeline.

Notebook ghi Python/PyTorch/CUDA, tên GPU, VRAM, compute capability, benchmark
throughput, assignment, cache hit/miss/eviction/fallback, CUDA allocated/reserved
peak, utilization sample, worker traceback và nonzero exit propagation.

Coordinator và cả hai worker vừa ghi log file vừa stream log ra cell Kaggle.
Tiến độ được ghi theo stage; MARLFS ghi epoch 1, mỗi 10 epoch và epoch
cuối của từng client; encoder và final classifier ghi mọi epoch; PPO ghi
mọi client-feedback calibration và summary mỗi search epoch.

Mỗi notebook phải:

1. parse JSON và syntax-check mọi code cell/embedded entry point;
2. assert đúng hai NVIDIA T4;
3. validate data/cache classification trước CUDA transfer;
4. assert output `[B,34]` và exact parameter count 40.034/71.010/35.874;
5. dùng một serialized `CONFIG` và entry point dưới `/kaggle/temp`;
6. dùng persistent client-parallel worker, AMP, deterministic assignment/cache;
7. triển khai đủ MARLFS, permutation augmentation, ISAB/PMA, PPO,
   sample-aware feedback và final personalized evaluation;
8. tạo đủ output không rỗng và kiểm tra row count 10/10/100 cho
   round/client/local-epoch history;
9. không placeholder, silent subsampling ngoài search pool đã khai báo,
   fabricated path hoặc test leakage;
10. chỉ tuyên bố structural validation tại local; runtime validation chỉ
    sau khi notebook chạy xong trên Kaggle `GPU T4 x2`.
