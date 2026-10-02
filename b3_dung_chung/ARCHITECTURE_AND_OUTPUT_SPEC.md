# Task10: đặc tả kiến trúc, huấn luyện và đầu ra

## 1. Phạm vi đã chốt

Task10 tái xây dựng năm phương pháp trên cùng bộ dữ liệu CICIoT2023 và cùng ba
kịch bản đồng nhất:

| Phương pháp | Folder | Kịch bản |
|---|---|---|
| FD-IDS | `fd_ids_noniid` | 10 GRU, 10 Transformer, 10 CNN-1D |
| PerFed-SKD | `perfed_skd` | 10 GRU, 10 Transformer, 10 CNN-1D |
| FedCAPS | `permutation_feature_importance` | 10 GRU, 10 Transformer, 10 CNN-1D |
| pFedES | `pfedes` | 10 GRU, 10 Transformer, 10 CNN-1D |
| ProxyModel | `proxymodel` | 10 GRU, 10 Transformer, 10 CNN-1D |

Mỗi notebook chạy đúng 10 communication/training round. Mỗi local update dùng
một epoch, trừ các pha nội tại được paper quy định rõ như tìm feature của
FedCAPS, server pretrain của PerFed-SKD và final personalized fine-tune của
ProxyModel. Không tạo `best.pt`; round 10 là endpoint so sánh cố định.

## 2. Hợp đồng dữ liệu

Đường dẫn Kaggle chính xác:

```text
/kaggle/input/datasets/odixe0502/data-for-10clients/
```

Các file bắt buộc:

- `client_1_train.csv` đến `client_10_train.csv`;
- `global_train_data.csv`;
- `global_test_data.csv`;
- `label_mapping.csv`.

Bài toán là phân loại đơn nhãn đa lớp với 25 feature, 34 lớp và nhãn nguyên
`0..33`. Header và thứ tự feature phải khớp `data_description.md`. Dữ liệu đã
được chuẩn hóa upstream bằng chính sách chung (`QuantileTransformer` fit trên
train rồi áp dụng sang test); notebook không fit lại scaler và không làm rò rỉ
global test.

### 2.1. Chính sách local data

| Phương pháp | Local train | Local validation |
|---|---:|---:|
| PerFed-SKD | 90% phân tầng, seed 42 | 10% phân tầng, chỉ dùng chọn client vòng sau |
| FD-IDS | 100% file client | Không |
| FedCAPS final classifier | 100% file client | Không |
| pFedES | 100% file client | Không |
| ProxyModel | 100% file client | Không |

Riêng FedCAPS tạo một search pool phân tầng độc lập, tối đa 100.000 dòng/client,
và dùng deterministic stratified 5-fold CV trong mọi phép đánh giá candidate.
Search pool không thay thế dữ liệu final training và không bao giờ chứa dữ liệu
từ `global_test_data.csv`.

`global_test_data.csv` chỉ dùng để đo sau từng round. Kết quả test không điều
khiển training, client selection, feature selection hoặc checkpoint ranking.

## 3. Kiến trúc classifier dùng chung

Mọi classifier nhận `[B, 25]` (FedCAPS nhận subset `[B, S]`) và trả logits
`[B, 34]`; softmax không nằm trong model.

### 3.1. GRU — 40.034 tham số

- reshape `[B,25] -> [B,25,1]`;
- GRU 2 tầng, input 1, hidden 64, dropout 0,2;
- lấy hidden ở bước cuối;
- `Linear(64,34)`.

### 3.2. Transformer — 71.010 tham số

- scalar projection `Linear(1,64)`;
- learned positional embedding độ dài 25;
- 2 Transformer encoder layer, 4 heads, FFN 128, dropout 0,1;
- mean pooling, LayerNorm, `Linear(64,34)`.

### 3.3. CNN-1D — 35.874 tham số

- `Conv1d(1,32,3,padding=1) + BatchNorm + ReLU`;
- `Conv1d(32,64,3,padding=1) + BatchNorm + ReLU + MaxPool(2)`;
- `Conv1d(64,128,3,padding=1) + BatchNorm + ReLU`;
- adaptive average pooling và `Linear(128,34)`.

### 3.4. Proxy riêng

- pFedES dùng feature extractor 1D `Conv 1→8→1`, kernel 3, padding 1, ReLU
  giữa hai conv; output vẫn `[B,25]`, đúng 57 tham số. Server pFedES không phải
  classifier.
- ProxyModel dùng một CNN-1D classifier 35.874 tham số làm proxy chung. Trong
  kịch bản CNN-1D, personalized CNN và proxy CNN là hai model/state riêng.
- Server FedCAPS giữ encoder/decoder/actor/critic và feature subset; state này
  không phát logits 34 lớp.

## 4. Cấu hình thực thi chung trên Kaggle

| Thành phần | Giá trị |
|---|---:|
| Client | 10 |
| Round | 10 |
| Local epoch/round | 1 |
| Batch/client update | 1024 trên một GPU |
| Gradient accumulation | 1 |
| Seed/init seed | 42 |
| Mixed precision | CUDA AMP |
| GPU | chính xác 2 NVIDIA T4 |
| Topology | CPU coordinator + 2 persistent GPU worker |
| GPU cache budget | tối đa 70% VRAM/GPU |
| CUDA stream candidates | `[1,2]` |
| DDP/DataParallel | Không |

Hai worker sở hữu hai phân hoạch client cố định. Phân hoạch được chọn bằng
benchmark optimizer-step thực tế để giảm predicted makespan. Dữ liệu được cache
trên GPU trong giới hạn 70%; nếu không đủ VRAM thì dùng pinned-memory/memmap
fallback theo chunk. TF32 và AMP được bật.

GRU/Transformer dùng một stream/GPU để giữ dropout RNG xác định. CNN-1D dùng
hai stream và hai client update đồng thời khi benchmark đo được throughput tăng
ít nhất 10% và dữ liệu nằm trọn trong GPU cache; nếu không thì dùng một stream
để tránh tranh chấp pinned-memory loader. Quy tắc overlap hai stream áp dụng cho
runtime dùng chung của FD-IDS, PerFed-SKD, pFedES và ProxyModel. FedCAPS giữ một
stream/GPU vì MARLFS/PPO có chuỗi state-feedback tuần tự; hai GPU vẫn xử lý hai
phân hoạch client song song trong mọi phương pháp.

## 5. Quy tắc phương pháp

### 5.1. FD-IDS

Mỗi round, mọi client khởi tạo từ global classifier, dùng global snapshot làm
teacher và tối ưu hard CE + KD + FedProx. Server FedAvg có trọng số theo toàn bộ
số dòng local. Vì local state không tồn tại độc lập sau aggregation, chỉ đánh
giá và checkpoint server classifier.

### 5.2. PerFed-SKD

Server pretrain đúng một epoch trên `global_train_data.csv`. Mỗi client giữ
personalized state và historical teacher. Local validation accuracy của đủ 10
client ở round `t` tạo ngưỡng trung bình; client có accuracy nhỏ hơn ngưỡng nhận
global state ở round `t+1`. Chỉ selected clients tham gia server aggregation
không trọng số. Validation ghi đủ 10 metrics nhưng chỉ accuracy được phép điều
khiển selection. Server và đủ 10 client được test sau từng round.

### 5.3. FedCAPS

Pipeline gồm MARLFS record collection, permutation augmentation, set
encoder/decoder, PPO search và 5-fold client feedback. Sau khi chốt feature
subset, mỗi personalized classifier học 10 round liên tiếp, mỗi round một epoch
trên 100% dữ liệu local; client state được giữ qua round và không FedAvg
classifier. Đánh giá đủ 10 client sau từng round; server classification được
ghi `not_applicable`.

### 5.4. pFedES

Mỗi round có hai pha trên 100% dữ liệu local: đóng băng proxy để cập nhật
personalized classifier, sau đó đóng băng classifier để cập nhật proxy. Chỉ
proxy 57 tham số được FedAvg có trọng số. Đánh giá đủ 10 personalized client;
server proxy classification là `not_applicable`.

### 5.5. ProxyModel

Personalized classifier và local CNN-1D proxy tối ưu đồng thời bằng label loss
và adaptive mutual distillation. Chỉ proxy được FedAvg có trọng số. Round 10 có
một final personalized CE fine-tune như phương pháp gốc, rồi mới checkpoint và
đánh giá. Đánh giá đủ 10 client và global CNN-1D proxy sau từng round.

## 6. Hợp đồng đánh giá

Mỗi entity áp dụng được đánh giá trên toàn bộ `global_test_data.csv` ngay sau
mỗi round bằng đúng 10 cột so sánh:

1. `accuracy`
2. `macro_precision`
3. `micro_precision`
4. `weighted_precision`
5. `macro_recall`
6. `micro_recall`
7. `weighted_recall`
8. `macro_f1`
9. `micro_f1`
10. `weighted_f1`

Với single-label multiclass, micro precision, micro recall, micro F1 và
accuracy có cùng giá trị toán học, nhưng vẫn phải ghi thành bốn cột độc lập.
Zero division nhận 0. Mọi metric nằm trong `[0,1]`.

| Phương pháp | Entity/round | Số hàng `evaluation_metrics` | Hàng round 10 |
|---|---:|---:|---:|
| FD-IDS | 1 server | 10 | 1 |
| PerFed-SKD | 1 server + 10 client | 110 | 11 |
| FedCAPS | 10 client | 100 | 10 |
| pFedES | 10 client | 100 | 10 |
| ProxyModel | 1 server + 10 client | 110 | 11 |

Raw confusion matrix được lưu theo round/entity để kiểm tra lại toàn bộ metric.

## 7. Checkpoint

Không có checkpoint “best”. Mỗi round có thư mục riêng:

```text
checkpoints/
├── checkpoint_manifest.json
├── round_001/
│   ├── server.pt
│   ├── client_01.pt      # chỉ phương pháp có persistent client
│   └── ...
└── round_010/
```

- FD-IDS: 10 server checkpoint, không lưu local duplicate.
- PerFed-SKD, FedCAPS, pFedES, ProxyModel: lưu server state và đủ 10 client
  checkpoint tại từng round.
- `checkpoint_manifest.json` ghi round, scope, client ID, path và state hash
  khi áp dụng.

## 8. Output bắt buộc

Mỗi run ghi vào `/kaggle/working/{run_name}/`:

```text
checkpoints/checkpoint_manifest.json
logs/run.log
logs/worker_0.log
logs/worker_1.log
metrics/config.json
metrics/dataset_summary.json
metrics/client_class_distribution.csv
metrics/history_pretrain.csv
metrics/history_pretrain.json
metrics/history_round.csv
metrics/history_round.json
metrics/history_client.csv
metrics/history_client.json
metrics/evaluation_metrics.csv
metrics/evaluation_metrics.json
metrics/final_round_metrics.csv
metrics/final_round_metrics.json
metrics/server_evaluation_status.json
metrics/communication_costs.json
metrics/runtime_breakdown.json
metrics/summary.json
metrics/confusion_matrices/round_*/...
artifacts/class_distribution.png
artifacts/evaluation_metric_curves.png
artifacts/loss_curves.png
artifacts/runtime_per_round.png
artifacts/communication_cumulative.png
artifacts/gpu_utilization.png
```

PerFed-SKD thêm `validation_metrics.csv/json`. FedCAPS thêm record, encoder,
PPO, candidate-evaluation, selected-feature, personalized-test,
classification-report và pooled-confusion outputs.

Notebook phải tự kiểm tra số hàng metric, số checkpoint, file rỗng, support
global test, parameter count, input/output shape, đúng 2 T4 và trạng thái server
classification áp dụng/không áp dụng trước khi kết thúc thành công.
