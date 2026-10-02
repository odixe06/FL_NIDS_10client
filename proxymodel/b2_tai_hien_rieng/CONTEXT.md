# Bối cảnh dự án FD-IDS client-parallel trên Kaggle

## 1. Mục tiêu đã chốt

Tái hiện phương pháp và công thức chính của paper
`2025 Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.md`
cho bài toán IDS trên dữ liệu mô tả trong `data_description.md`, dưới dạng ba
notebook Kaggle độc lập:

| Kịch bản | Notebook |
|---|---|
| 10 client GRU | `scenario_1_10_clients_gru/fd_ids_ciciot2023_10c_gru.ipynb` |
| 10 client Transformer | `scenario_2_10_clients_transformer/fd_ids_ciciot2023_10c_transformer.ipynb` |
| 5 GRU + 5 Transformer | `scenario_3_5_gru_5_transformer/fd_ids_ciciot2023_10c_mixed_5gru_5transformer.ipynb` |

Mỗi client có một personalized model và một proxy CNN-1D. Chỉ proxy được gửi
về coordinator và FedAvg; GRU/Transformer không được tổng hợp.

`ARCHITECTURE_AND_OUTPUT_SPEC.md` là hợp đồng chuẩn cuối cùng về dữ liệu, mô
hình, khởi tạo, metric, checkpoint và output. Nếu tài liệu này mâu thuẫn với
ghi chú cũ thì ưu tiên tài liệu đặc tả.

## 2. Các quyết định trong cuộc trò chuyện

1. Phiên bản đầu tiên dùng DDP trên hai T4.
2. Khi train đã xuất hiện warning `c10d` hostname, biến
   `NCCL_ASYNC_ERROR_HANDLING` deprecated và `barrier()` thiếu `device_id`.
3. Người dùng yêu cầu bỏ hoàn toàn DDP. Phương án cuối được chốt là
   **client-parallel**: mỗi GPU xử lý một nhóm client độc lập và hai GPU làm
   việc đồng thời.
4. Batch được giữ nguyên **1024 cho mỗi client update**, không chia đôi giữa
   GPU và không dùng sampler padding.
5. Để tăng tốc và sử dụng VRAM có ích, người dùng đồng ý:
   - hai process bền vững, mỗi process sở hữu đúng một GPU;
   - chia client tĩnh bằng exact two-way partition theo workload benchmark;
   - cache dữ liệu client trên GPU tối đa 70% VRAM;
   - deterministic LRU nếu toàn bộ cache không vừa;
   - nạp theo chunk qua pinned memory;
   - benchmark 1 và 2 CUDA stream trên từng GPU, chỉ chọn 2 stream khi
     throughput cao hơn ít nhất 3%;
   - CUDA AMP cho train và evaluation.
6. Các output và trọng số khởi tạo phải giữ đúng hợp đồng để so sánh công bằng
   với các phương pháp khác.
7. Phần phân tích paper trong notebook chỉ ghi ngắn gọn những thay đổi so với
   paper gốc.
8. Nếu một quyết định mới có thể thay đổi phương pháp, schema hoặc khả năng so
   sánh, cần hỏi người dùng trước khi thực hiện.

## 3. Dữ liệu và ánh xạ từ paper

- Kaggle input:
  `/kaggle/input/datasets/odixe0502/data-for-10clients/`.
- Train: `client_1_train.csv` đến `client_10_train.csv`.
- Test độc lập: `global_test_data.csv`.
- Nhãn: `label_mapping.csv`, 34 lớp, ID `0..33`.
- `global_train_data.csv` chỉ dùng kiểm tra tổng số dòng, không train lại.
- Có 25 feature theo đúng thứ tự trong `ARCHITECTURE_AND_OUTPUT_SPEC.md`.
- Mỗi client tách stratified 95% local train và 5% validation.
- Test không tham gia chọn checkpoint.

Các thay đổi chính so với paper gốc:

- NILM hồi quy được đổi thành phân loại IDS 34 lớp.
- Dữ liệu REFIT/REDD được đổi thành CICIoT2023 đã chọn 25 feature và chuẩn hóa.
- Personalized model được cố định là GRU/Transformer; không chạy lại MNAS.
- Proxy đồng nhất là CNN-1D.
- Label loss dùng cross-entropy; distillation dùng MSE giữa hai softmax.
- Chỉ proxy dùng weighted FedAvg.
- MI chỉ được mô tả, không tuyên bố chạy lại vì input 46 feature ban đầu không
  có trong bộ dữ liệu hiện tại.

## 4. Công thức đang được triển khai

Với `n_k` là số mẫu local train sau khi giữ validation:

```text
w_G^(t+1) = sum_k [n_k / sum_j(n_j)] * w_(r,k)^(t+1)
```

```text
ell_s = CE(y, z_s)
ell_r = CE(y, z_r)
d     = MSE(softmax(z_s), softmax(z_r))
ell_d = d / (stopgrad(ell_s + ell_r) + 1e-8)
L_s   = ell_s + ell_d
L_r   = ell_r + ell_d
```

Một backward chung trên `ell_s + ell_r + ell_d` để mỗi model nhận label
gradient riêng và đúng một lần distillation gradient.

## 5. Kiến trúc và hợp đồng khởi tạo

- GRU: input `[B,25,1]`, 2 layer, hidden 64, dropout 0.2, last step, Linear
  `64 -> 34`; dự kiến 40.034 tham số.
- Transformer: scalar projection `1 -> 64`, learned position length 25,
  2 encoder layer, 4 head, FFN 128, dropout 0.1, mean pool, LayerNorm, Linear
  `64 -> 34`; dự kiến 71.010 tham số.
- CNN-1D proxy: channel `[32,64,128]`, kernel 3, BatchNorm, ReLU, MaxPool sau
  block 2, adaptive average pool, Linear `128 -> 34`; dự kiến 35.874 tham số.

Khởi tạo:

- `seed = initialization_seed = 42`.
- Mỗi họ model được tạo trong CPU RNG fork độc lập và seed lại bằng 42.
- Mọi client cùng họ bắt đầu từ state giống nhau.
- Cùng một họ model có initial state giống nhau giữa cả ba notebook.
- Runtime tạo state lần hai và assert SHA-256 trùng trước khi train.
- Initial states và SHA-256 được ghi vào `config.json`, checkpoint và
  `summary.json`.
- Checkpoint không có prefix `module.` vì không dùng DDP.

Không thay đổi cách khởi tạo, tên layer hoặc số tham số nếu chưa cập nhật đặc
tả và chưa được người dùng chốt, vì việc đó làm mất khả năng so sánh.

## 6. Topology hai GPU cuối cùng

Notebook ghi runtime độc lập vào `/kaggle/temp` rồi chạy bằng Python subprocess
thông thường. Runtime gồm:

```text
CPU coordinator
├── GPU worker 0 -> cuda:0 -> nhóm client cố định
└── GPU worker 1 -> cuda:1 -> nhóm client cố định
```

- Multiprocessing start method: `spawn`.
- Parent không khởi tạo CUDA trước khi spawn.
- Mỗi worker gọi `torch.cuda.set_device(gpu_id)`.
- Không import `torch.distributed`, không process group, không DDP, không
  NCCL, không `torchrun`, không `DistributedSampler`.
- Hai worker benchmark thời gian optimizer step cho các họ model cần dùng.
- Coordinator duyệt toàn bộ phân hoạch hai phía không rỗng của 10 client và
  tối thiểu hóa predicted makespan.
- Assignment giữ cố định qua 10 communication round để model và cache không
  phải di chuyển giữa GPU.
- Trong mỗi worker, client nặng hơn chạy trước.
- Cache bảo vệ tất cả client đang hoạt động; LRU không thể loại tensor đang
  được stream sử dụng.
- Nếu hai cache client hoạt động đồng thời vượt ngân sách, worker tự hạ còn
  một stream dù benchmark throughput chọn hai.
- Metrics GPU ghi cả allocated/reserved, peak, utilization và VRAM sử dụng.

Việc cố tình cấp phát VRAM rỗng không giúp tăng tốc. Thiết kế này chỉ dùng thêm
VRAM cho dữ liệu/cache và concurrency có thể giảm I/O hoặc tăng throughput.

## 7. Huấn luyện và checkpoint

- 10 communication round, 10/10 client tham gia.
- 1 local epoch/round, Adam, learning rate 0.001, không scheduler.
- Mỗi round, client tiếp tục personalized state của mình và nạp global proxy
  mới nhất.
- Chọn `best.pt` theo validation macro-F1 của global proxy.
- `last.pt` là trạng thái sau round 10, trước final fine-tune.
- Sau khi chọn best round, personalized model được nạp từ best checkpoint và
  fine-tune thêm một epoch chỉ bằng cross-entropy.
- `best.pt` giữ cả
  `personalized_model_state_dicts_before_finetune` và state triển khai cuối.
- Global test chỉ chạy sau khi chọn best checkpoint.

## 8. Output bắt buộc

Mỗi run ghi dưới `/kaggle/working/{run_name}/`:

```text
checkpoints/best.pt
checkpoints/last.pt
logs/run.log
logs/rank_1.log
metrics/config.json
metrics/dataset_summary.json
metrics/client_class_distribution.csv
metrics/history_round.csv
metrics/history_round.json
metrics/history_client.csv
metrics/history_client.json
metrics/history_local_epoch.csv
metrics/history_local_epoch.json
metrics/summary.json
metrics/classification_report.json
metrics/classification_report.csv
metrics/confusion_matrix.csv
metrics/confusion_matrix.npy
metrics/communication_costs.json
metrics/communication_costs.csv
metrics/runtime_breakdown.json
artifacts/class_distribution.png
artifacts/accuracy_f1_curves.png
artifacts/loss_curves.png
artifacts/confusion_matrix.png
artifacts/per_class_f1.png
artifacts/runtime_per_round.png
artifacts/communication_cumulative.png
```

`rank_1.log` là tên tương thích report cũ nhưng chứa log GPU worker 1.
`ddp_gradient_allreduce_estimate` vẫn tồn tại trong communication report để
tương thích schema, nhưng `enabled=false` và mọi byte bằng 0.

## 9. File nguồn quan trọng

- `ARCHITECTURE_AND_OUTPUT_SPEC.md`: hợp đồng cuối.
- `scripts/client_parallel_runtime.py`: runtime được nhúng giống hệt vào cả ba
  notebook.
- `scripts/build_federated_notebooks.py`: generator cho ba notebook; không sửa
  notebook bằng tay nếu có thể tái sinh từ generator.
- `.agents/skills/kaggle-training-notebook/SKILL.md`: skill đã được cập nhật để
  chọn client-parallel cho FL có client độc lập.
- `.agents/skills/kaggle-training-notebook/references/client_parallel_templates.md`:
  hướng dẫn topology/cache/stream/assignment.
- `.agents/skills/kaggle-training-notebook/scripts/validate_client_parallel_notebook.py`:
  validator chuyên dụng.

Quy trình sau khi sửa runtime hoặc generator:

```bash
python3 scripts/build_federated_notebooks.py
python3 .agents/skills/kaggle-training-notebook/scripts/validate_client_parallel_notebook.py \
  scenario_1_10_clients_gru/fd_ids_ciciot2023_10c_gru.ipynb \
  scenario_2_10_clients_transformer/fd_ids_ciciot2023_10c_transformer.ipynb \
  scenario_3_5_gru_5_transformer/fd_ids_ciciot2023_10c_mixed_5gru_5transformer.ipynb
```

## 10. Trạng thái kiểm định tại thời điểm ghi file

- Cả ba notebook là JSON nbformat 4 và mọi code cell parse được bằng AST.
- Cả ba qua `validate_client_parallel_notebook.py`.
- Skill qua `quick_validate.py`.
- Runtime nhúng của ba notebook có cùng SHA-256; cấu hình chung giống nhau,
  ngoại trừ `run_name`, tên kịch bản và danh sách kiến trúc client.
- Validator cấm DDP/distributed/NCCL/torchrun, `.item()` trong training loop,
  TODO/placeholder; đồng thời kiểm tra batch 1024, AMP, spawn, cache 70%,
  stream/event, error propagation, checkpoint và đủ 28 output.
- Cell cuối notebook tự kiểm tra file thiếu/rỗng, số hàng history, padding bằng
  0, GPU ID, stream count, cột memory, schema checkpoint và initialization hash.

Giới hạn kiểm định:

- Máy phát triển hiện tại không có PyTorch/CUDA/dataset Kaggle nên chưa thể
  chạy full training tại chỗ.
- Bước tiếp theo nên là chạy một notebook trên Kaggle T4 x2 bằng `Run All`,
  kiểm tra `status=completed`, peak utilization/VRAM trong
  `runtime_breakdown.json`, rồi mới chạy hai kịch bản còn lại.
- Nếu Kaggle thay phiên bản PyTorch, hash số cụ thể có thể thay theo thư viện,
  nhưng trong cùng môi trường ba notebook vẫn phải trùng hash cho cùng họ model.

## 11. Kỳ vọng về warning

Các warning đã nêu trong cuộc trò chuyện đến từ c10d/NCCL/DDP. Runtime cuối
không dùng các thành phần đó nên không còn đường code tạo các warning hostname,
deprecated `NCCL_ASYNC_ERROR_HANDLING` hay `barrier()` thiếu `device_id`.
Warning khác từ Kaggle/PyTorch vẫn có thể xuất hiện nếu môi trường thay đổi;
khi đó cần đọc `run.log` và `rank_1.log` trước khi chỉnh cấu hình.

## 12. Sửa lỗi CUDA device-side assert ở round đầu

Lần chạy notebook GRU trên Kaggle từng lỗi tại
`enqueue_training_step()`/`backward()` với `CUDA error: device-side assert
triggered`. CUDA báo bất đồng bộ nên `backward()` không phải vị trí phát sinh
ban đầu.

Nguyên nhân là race giữa default stream và client stream:

- client stream gọi `wait_stream(default_stream)`;
- sau lệnh wait, `make_training_context()` lại tạo `torch.randperm()` và
  `loss_sums` trên default stream;
- client stream có thể dùng permutation trong `index_select()` trước khi
  default stream ghi xong.

`wait_stream()` chỉ chờ công việc đã enqueue trước thời điểm gọi, không tạo phụ
thuộc cho công việc default-stream được enqueue sau đó.

Bản sửa tạo permutation và loss accumulator ngay trong
`with torch.cuda.stream(stream)`. Runtime cũng kiểm tra shape/dtype/range của
feature, label và train/validation index trên CPU trước khi copy cache sang
GPU; lỗi shutdown thứ cấp không còn che traceback worker ban đầu.

Validator client-parallel hiện dùng AST để bắt buộc `order` và `loss_sums`
được tạo trong owning stream. Cả ba notebook đã được tái sinh từ generator và
qua structural validation. Vẫn cần chạy lại trên phiên Kaggle T4 x2 mới để xác
nhận full training; sau một device-side assert phải restart session vì CUDA
context cũ không còn đáng tin cậy.
