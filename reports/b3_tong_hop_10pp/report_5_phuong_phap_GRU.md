# Báo cáo tổng hợp — 5 phương pháp Federated Learning Lightweight trên GRU

Báo cáo này tổng hợp năm phương pháp học liên kết được xây dựng trên cùng một bộ
dữ liệu CICIoT2023 với nền tảng GRU, kèm toàn bộ kết quả đo được.

**Tài liệu này được tạo thủ công** từ chính các file `federated_learning_results.csv`
của 5 lần chạy trong 5 thư mục `GRU_*`. Mọi con số trong báo cáo đều đọc trực tiếp
từ output, không nhập tay.

## Mục lục

1. [Bối cảnh thí nghiệm chung](#1-bối-cảnh-thí-nghiệm-chung)
2. [Bảng tra nhanh năm bài báo](#2-bảng-tra-nhanh-năm-bài-báo)
3. [FedAvg (baseline)](#3-fedavg-baseline)
4. [CustomAggregation + Unstructured Magnitude](#4-customaggregation--unstructured-magnitude)
5. [Hetero-FedDistillation](#5-hetero-feddistillation)
6. [Zero-shot Pruning + PTD Quantization](#6-zero-shot-pruning--ptd-quantization)
7. [Zero-shot Pruning](#7-zero-shot-pruning)
8. [So sánh chéo năm phương pháp](#8-so-sánh-chéo-năm-phương-pháp)
9. [Phân tích điểm mạnh — điểm yếu](#9-phân-tích-điểm-mạnh--điểm-yếu)
10. [Kết luận](#10-kết-luận)

---

## 1. Bối cảnh thí nghiệm chung

Cả năm phương pháp chạy trên **đúng cùng một cấu hình** để kết quả so sánh được:

| Thành phần | Giá trị |
|---|---|
| Bộ dữ liệu | CICIoT2023, 25 feature, **34 lớp**, phân loại đơn nhãn đa lớp |
| Số client | 10 |
| Tập test | `global_test_data.csv`, dùng chung cho mọi phương pháp |
| Số round | 10 |
| Local epoch mỗi round | 1 |
| Train batch size | 8192/client |
| Eval batch size | 1024 |
| Optimizer | Adam, lr = 0,001 |
| Kiến trúc | **GRU 2 lớp, 128 units, dropout 0,2** + Linear(128, 34) |
| Phần cứng | Kaggle, GPU |

Bốn phương pháp "lightweight" được xây dựng trên đúng cấu hình này, chỉ khác nhau
ở kỹ thuật nén (pruning / quantization / distillation) và cách tổng hợp ở server:

| Phương pháp | Kỹ thuật nén | Tổng hợp server |
|---|---|---|
| FedAvg (baseline) | không có | FedAvg có trọng số `n_k/n` |
| CustomAgg + Unstructured Mag. | unstructured magnitude pruning, tỷ lệ động theo client | tùy biến theo mask (chia trung bình) |
| Hetero-FedDistillation | chưng cất logits trên public data + INT8 | tổng hợp logits rồi chưng cất |
| Zero-shot Pruning + PTD Quant | structured pruning (20%) + INT8 PTD | FedAvg có trọng số `n_k/n` |
| Zero-shot Pruning | structured pruning (20%) | FedAvg có trọng số `n_k/n` |

### Hợp đồng đo lường

Mỗi phương pháp được đánh giá trên **toàn bộ** `global_test_data.csv` ngay sau mỗi
round bằng đúng 10 cột:

`accuracy` · `macro_precision` · `micro_precision` · `weighted_precision` ·
`macro_recall` · `micro_recall` · `weighted_recall` · `macro_f1` · `micro_f1` ·
`weighted_f1`

---

## 2. Bảng tra nhanh năm bài báo

| # | Phương pháp | Bài báo | Năm | Nguồn xác định năm |
|---|---|---|---|---|
| 1 | **FedAvg** (baseline) | DeepFed: Federated Deep Learning for Intrusion Detection in Industrial Cyber–Physical Systems | 2021 | ghi rõ trong file (`IEEE TII, vol. 17, no. 8`) |
| 2 | **CustomAgg + Unstructured Mag.** | OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT | 2025 | mã arXiv `2510` = tháng 10/2025 |
| 3 | **Hetero-FedDistillation** | Adaptive personalized federated learning with lightweight depthwise convolutional bottleneck network for novel IDS in internet of vehicles | 2025 | ghi rõ trong file (`Scientific Reports 15, 35604`) |
| 4 | **Zero-shot Pruning + PTD Quant.** | Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS) | 2024 | ghi rõ trong file (`IEEE Access, vol. 12`) |
| 5 | **Zero-shot Pruning** | Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS) | 2024 | ghi rõ trong file (`IEEE Access, vol. 12`) |

> **Cảnh báo:** phương pháp 4 và 5 dùng chung **cùng một bài báo nguồn**
> (Lightweight-Fed-NIDS) — chúng là hai biến thể của cùng một kỹ thuật, khác nhau
> ở việc có thêm post-training quantization hay không. Khi so sánh, nên xem 4 và 5
> như một cặp "pruning-only" và "pruning + quantization" thay vì hai phương pháp
> độc lập.

Bảng khả năng làm việc với mô hình dị thể (chi tiết ở từng mục):

| Phương pháp | Khác **kích thước** mô hình | Khác **loại** mô hình | Thứ được truyền lên server |
|---|:---:|:---:|---|
| FedAvg | ✗ | ✗ | toàn bộ tham số GRU |
| CustomAgg + Unstructured Mag. | ✗ | ✗ | tham số thưa + mask |
| Hetero-FedDistillation | ✓ | ✓ | logits trên public data |
| Zero-shot Pruning + PTD Quant. | ✗ | ✗ | tham số model đã prune + quantize |
| Zero-shot Pruning | ✗ | ✗ | tham số model đã prune |

---

## 3. FedAvg (baseline)

### 3.1. Bài báo

- **Tên:** DeepFed: Federated Deep Learning for Intrusion Detection in Industrial Cyber–Physical Systems
- **Tác giả:** Beibei Li, Yuhao Wu, Jiarui Song, Rongxing Lu, Tao Li, Liang Zhao
- **Năm:** 2021 — *ghi rõ trong bài báo*
- **Nơi xuất bản:** IEEE Transactions on Industrial Informatics, vol. 17, no. 8, pp. 5615–5624, Aug 2021
- **File trong repo:** [`GRU_FedAvg_NoPrunning/09195012.pdf`](GRU_FedAvg_NoPrunning/09195012.pdf)

### 3.2. Quy trình huấn luyện của bài báo gốc

DeepFed giải bài toán phát hiện xâm nhập cho industrial cyber–physical system (CPS)
bằng học liên kết. Quy trình gốc gồm năm pha (Algorithm 1):

1. **Khởi tạo hệ thống.** Trust Authority sinh cặp khóa Paillier, lập kênh an toàn
   giữa cloud server và từng agent. Server khởi tạo tham số `w₀`, learning rate `η`,
   batch size `B`, và tính tỷ lệ đóng góp `α_k = N_k / ΣN_j` cho từng agent.
2. **Huấn luyện cục bộ.** Mỗi agent huấn luyện mô hình trên dữ liệu riêng `D_k`
   (Algorithm 2).
3. **Mã hóa tham số.** Mỗi agent mã hóa toàn bộ tham số `w_k` bằng Paillier
   (`ParaEncrypt`) trước khi gửi lên cloud — đây là điểm mấu chốt bảo mật của DeepFed.
4. **Tổng hợp trên ciphertext.** Server cộng trọng số có trọng số theo `α_k`
   (`ParaAggregate`) ngay trên dữ liệu đã mã hóa, gửi kết quả về.
5. **Giải mã và cập nhật.** Agent giải mã `ParaDecrypt` rồi thay model cục bộ bằng
   model toàn cục mới. Lặp lại `R` round.

Mô hình của bài gốc là **CNN-GRU** (trích feature bằng CNN, học tuần tự bằng GRU).
Đầu ra cuối cùng là **một global model duy nhất**.

### 3.3. Bản build lại — giữ gì, đổi gì

| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | dữ liệu CPS thật | **CICIoT2023**, 25 feature, 34 lớp | **Khác** |
| Model | CNN-GRU | **GRU thuần** (2 lớp, 128 units) | **Khác** |
| Mã hóa Paillier | có | **không** | **Bỏ** |
| Tổng hợp | FedAvg trọng số `α_k = n_k/n` | FedAvg trọng số `n_k/n` | Giữ nguyên |
| Optimizer | Adam | Adam, lr 0,001 | Giữ nguyên |
| Số round | `R` | **10** | **Khác (chuẩn hoá)** |
| Local epoch | — | **1** | Giữ nguyên |
| Pruning/Quantization | không | không | Giữ nguyên (đây là baseline) |

**Vai trò:** đây là **baseline** — không có bất kỳ kỹ thuật nén nào, chỉ có FedAvg
thuần túy, để bốn phương pháp lightweight còn lại chứng minh lợi ích về kích
thước/truyền thông so với nó.

### 3.4. Có dùng được cho mô hình không đồng nhất không?

> **Ghi chú nguồn gốc:** hạn chế này **đã tồn tại trong bài báo gốc**, không phải do
> bản build lại thêm vào. DeepFed chỉ đề cập "heterogeneous industrial CPSs" về mặt
> *môi trường/dữ liệu*, còn mô hình là một CNN-GRU **dùng chung** cho mọi agent với
> tổng hợp FedAvg thuần — bài gốc vốn không thiết kế cho mô hình dị thể.

**Kết luận: **Không** — cả hai loại đều không.**

FedAvg tổng hợp bằng `w_G = Σ (n_k/n)·w_k` trên **toàn bộ tham số**. Phép cộng này
chỉ định nghĩa được khi mọi client có **cùng kiến trúc và cùng số tham số**.

- **Khác kích thước mô hình:** không. Vector tham số phải khớp từng phần tử.
- **Khác loại mô hình:** không. GRU và CNN không có tham số tương ứng nhau.

Đây cũng chính là động lực để các phương pháp còn lại (Hetero-FedDistillation,
pruning-based) tồn tại.

### 3.5. Kết quả đo được

- Thư mục run: [`GRU_FedAvg_NoPrunning/output`](GRU_FedAvg_NoPrunning/output)
- Checkpoint round 1: `baseline_grunet_round_1.pth`, **655.281 byte (~640 KB)**
- File kết quả: [`baseline_federated_learning_results.csv`](GRU_FedAvg_NoPrunning/output/baseline_federated_learning_results.csv)

*Bảng — FedAvg · GRU · 10 metrics trên `global_test_data.csv` theo từng round*

| Round | accuracy | macro_P | micro_P | weighted_P | macro_R | micro_R | weighted_R | macro_F1 | micro_F1 | weighted_F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5417 | 0.2425 | 0.5417 | 0.4122 | 0.2470 | 0.5417 | 0.5417 | 0.2160 | 0.5417 | 0.4454 |
| 2 | 0.5795 | 0.2987 | 0.5795 | 0.4999 | 0.3242 | 0.5795 | 0.5795 | 0.2696 | 0.5795 | 0.4872 |
| 3 | 0.6022 | 0.3628 | 0.6022 | 0.5420 | 0.3682 | 0.6022 | 0.6022 | 0.3128 | 0.6022 | 0.5133 |
| 4 | 0.6111 | 0.3841 | 0.6111 | 0.6451 | 0.3940 | 0.6111 | 0.6111 | 0.3307 | 0.6111 | 0.5210 |
| 5 | 0.6192 | 0.4162 | 0.6192 | 0.6616 | 0.4248 | 0.6192 | 0.6192 | 0.3627 | 0.6192 | 0.5314 |
| 6 | 0.6223 | 0.4215 | 0.6223 | 0.6642 | 0.4404 | 0.6223 | 0.6223 | 0.3787 | 0.6223 | 0.5355 |
| 7 | 0.6224 | 0.4240 | 0.6224 | 0.6653 | 0.4443 | 0.6224 | 0.6224 | 0.3850 | 0.6224 | 0.5370 |
| 8 | 0.6265 | 0.4314 | 0.6265 | 0.6693 | 0.4536 | 0.6265 | 0.6265 | 0.3972 | 0.6265 | 0.5426 |
| 9 | 0.6311 | 0.4475 | 0.6311 | 0.6764 | 0.4681 | 0.6311 | 0.6311 | 0.4091 | 0.6311 | 0.5484 |
| 10 | 0.6326 | 0.4526 | 0.6326 | 0.6803 | 0.4704 | 0.6326 | 0.6326 | 0.4155 | 0.6326 | 0.5518 |

---

## 4. CustomAggregation + Unstructured Magnitude

### 4.1. Bài báo

- **Tên:** OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT
- **Tác giả:** Saida Elouardi, Mohammed Jouhari, Anas Motii
- **Năm:** 2025 — *mã arXiv 2510.05180v2 = tháng 10/2025*
- **Nơi xuất bản:** arXiv:2510.05180v2
- **File trong repo:** [`GRU_Lightweight_CustomAggregation_UnstructuredMagnitude/2510.05180v2.pdf`](GRU_Lightweight_CustomAggregation_UnstructuredMagnitude/2510.05180v2.pdf)

### 4.2. Quy trình huấn luyện của bài báo gốc

OptiFLIDS nhúng **model pruning** trực tiếp vào vòng lặp FL để giảm năng lượng và
tài nguyên trên thiết bị IoT. Quy trình gốc:

1. **Mô hình hóa non-IID.** Dữ liệu được chia cho client bằng phân phối Gamma
   chuẩn hoá, tham số hình dạng `α` kiểm soát mức độ lệch.
2. **Tỷ lệ pruning do DRL quyết định.** Pruning được công thức hoá thành bài toán
   tối ưu đa mục tiêu (giảm năng lượng ↔ giữ hiệu năng); một agent **Deep
   Reinforcement Learning (DRL)** chọn tỷ lệ cắt `ρ_k` cho từng client. Kết quả
   trong bài: `ρ` tối ưu 65,8–68,4% tùy dataset.
3. **Unstructured magnitude pruning.** Trọng số nhỏ theo trị tuyệt đối bị đặt về 0
   theo mask `M_k ∈ {1, 0}`, áp dụng `W_k = W_k ⊙ M_k` (Eq. 7). Mask được tính **chỉ
   ở round 1** và gửi lên server một lần, tái sử dụng các round sau.
4. **Customized aggregation.** Vì các client cắt ở vị trí khác nhau (non-IID), server
   **chỉ tổng hợp những trọng số chưa bị cắt ở mọi client**:
   `W_global = Σ_k p_k·w_k·m_k` (Eq. 9), với `p_k = |D_k|/Σ|D_i|` (Eq. 10).
5. **Cập nhật cục bộ.** Client khôi phục độ thưa bằng `W_k = M_k ⊙ W_global`
   (Eq. 11), lặp lại đến khi hội tụ.

Đầu ra cuối cùng là **một global model** (thưa, tiết kiệm năng lượng).

### 4.3. Bản build lại — giữ gì, đổi gì

| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | TON_IoT, X-IIoTID, IDS-IoT2024 | **CICIoT2023** | **Khác** |
| Model | CNN | **GRU 2 lớp 128 units** | **Khác** |
| Tỷ lệ pruning | agent DRL (ρ ≈ 66–68%) | **giả lập công thức** `0.10 + (cid%4)·0.05 + (r%3)·0.02`, cap 0,40 | **Khác (thay DRL)** |
| Loại pruning | unstructured magnitude | unstructured magnitude (ngưỡng quantile) | Giữ nguyên |
| Mask tính | round 1, dùng lại | **mỗi round mỗi client tính lại** | **Khác** |
| Customized aggregation | chỉ gộp trọng số chưa cắt, trọng số `p_k` | chia trung bình theo mask tích lũy `Σw·p / Σm·p` | Tinh chỉnh |
| FedAvg trọng số | `p_k = n_k/n` | `n_k/n` | Giữ nguyên |
| Số round | tới hội tụ | **10** | **Khác (chuẩn hoá)** |

**Điểm lệch đáng chú ý so với bài gốc:**

- Bài gốc dùng **agent DRL thật** (huấn luyện reward để chọn `ρ`); bản build lại
  **mô phỏng** tỷ lệ pruning bằng một công thức phụ thuộc `client_id` và `round`.
  Vì vậy con số tỷ lệ ở đây **không phản ánh khả năng của DRL**.
- Bài gốc tính mask **một lần ở round 1**; bản build lại tính lại mỗi round, làm
  thay đổi bản chất "one-time pruning" của bài báo.

### 4.4. Có dùng được cho mô hình không đồng nhất không?

> **Ghi chú nguồn gốc:** giới hạn "chỉ dị thể mask, không dị thể kiến trúc" **khớp
> với bài báo gốc**. Trong OptiFLIDS, chữ "heterogeneous" mà bài báo dùng chỉ là
> *"structurally different pruned models"* — tức các mô hình CNN **giống hệt kiến
> trúc** nhưng có mask cắt khác nhau do non-IID; bài báo chưa bao giờ xử lý client
> dùng kiến trúc khác nhau.

**Kết luận: **Một phần** — xử lý được mask thưa khác nhau, nhưng không khác kiến trúc.**

- **Khác kích thước mô hình:** không. Mọi client vẫn phải cùng kiến trúc GRU — chỉ
  khác **mask** (vị trí trọng số bị cắt), và đây chính là thứ customized aggregation
  giải quyết: server chỉ gộp phần trọng số mà mọi client cùng giữ lại.
- **Khác loại mô hình:** không.

Về **dị thể tài nguyên**, cơ chế này có ý nghĩa thực tế: mỗi client có thể cắt ở
tỷ lệ khác nhau (client yếu cắt nhiều hơn), giảm tải tính toán và truyền thông cục bộ.

### 4.5. Kết quả đo được

- Thư mục run: [`GRU_Lightweight_CustomAggregation_UnstructuredMagnitude/output`](GRU_Lightweight_CustomAggregation_UnstructuredMagnitude/output)
- Checkpoint round 1: `optiflids_gru_r1.pth`, **655.154 byte (~640 KB)**
- File kết quả: [`federated_learning_results.csv`](GRU_Lightweight_CustomAggregation_UnstructuredMagnitude/output/federated_learning_results.csv)

*Bảng — CustomAgg + Unstructured Mag. · GRU · 10 metrics theo từng round*

| Round | accuracy | macro_P | micro_P | weighted_P | macro_R | micro_R | weighted_R | macro_F1 | micro_F1 | weighted_F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5247 | 0.2653 | 0.5247 | 0.4261 | 0.2351 | 0.5247 | 0.5247 | 0.1998 | 0.5247 | 0.4308 |
| 2 | 0.5859 | 0.2815 | 0.5859 | 0.4923 | 0.3316 | 0.5859 | 0.5859 | 0.2709 | 0.5859 | 0.4899 |
| 3 | 0.6058 | 0.3743 | 0.6058 | 0.5436 | 0.3750 | 0.6058 | 0.6058 | 0.3211 | 0.6058 | 0.5158 |
| 4 | 0.6141 | 0.3976 | 0.6141 | 0.6471 | 0.4130 | 0.6141 | 0.6141 | 0.3588 | 0.6141 | 0.5282 |
| 5 | 0.6164 | 0.4076 | 0.6164 | 0.6581 | 0.4273 | 0.6164 | 0.6164 | 0.3692 | 0.6164 | 0.5293 |
| 6 | 0.6177 | 0.4393 | 0.6177 | 0.6587 | 0.4309 | 0.6177 | 0.6177 | 0.3706 | 0.6177 | 0.5312 |
| 7 | 0.6279 | 0.4741 | 0.6279 | 0.7694 | 0.4320 | 0.6279 | 0.6279 | 0.3802 | 0.6279 | 0.5582 |
| 8 | 0.6141 | 0.4685 | 0.6141 | 0.7414 | 0.4277 | 0.6141 | 0.6141 | 0.3741 | 0.6141 | 0.5360 |
| 9 | 0.6111 | 0.4557 | 0.6111 | 0.6961 | 0.4297 | 0.6111 | 0.6111 | 0.3708 | 0.6111 | 0.5248 |
| 10 | 0.6284 | 0.5139 | 0.6284 | 0.7888 | 0.4423 | 0.6284 | 0.6284 | 0.3915 | 0.6284 | 0.5573 |

---

## 5. Hetero-FedDistillation

### 5.1. Bài báo

- **Tên:** Adaptive personalized federated learning with lightweight depthwise convolutional bottleneck network for novel IDS in internet of vehicles
- **Tác giả:** Fanghui Wang, Tao Cheng, Mingmin Zhao, Fengming Liu
- **Năm:** 2025
- **Nơi xuất bản:** Scientific Reports, vol. 15, art. 35604 (2025)
- **File trong repo:** [`GRU_Lightweight_Hetero_FedDistillation/s41598-025-17699-3.pdf`](GRU_Lightweight_Hetero_FedDistillation/s41598-025-17699-3.pdf)

### 5.2. Quy trình huấn luyện của bài báo gốc

Bài báo kết hợp hai ý tưởng: (i) **mô hình lightweight** LDwCBN
(Lightweight-Depthwise-Convolution-Bottleneck-Network) cho phân loại lưu lượng
mạng xe, và (ii) **feddistillation + cá nhân hoá** để các client có mô hình khác
nhau vẫn chia sẻ kiến thức được:

1. **Mô hình LDwCBN.** Gồm nhánh lightweight CNN (phần deep): 1 tầng depthwise
   conv (DWconv) + 1 tầng bottleneck 1×1, kết hợp nhánh skip connection; bổ sung
   nhánh biểu diễn tuần tự (phần temporal) để bắt nhịp thời gian của gói tin.
2. **Huấn luyện cục bộ.** Mỗi client huấn luyện trên dữ liệu riêng bằng loss kép:
   `L = L_CE(y_true, p) + λ · L_KL(p_pred · p_gdist)` — vừa khớp nhãn thật vừa
   khớp kiến thức nhận được từ global.
3. **FedDistillation.** Thay vì gửi tham số, client gửi **logits dự đoán trên một
   public dataset chung**. Server tổng hợp thành **global logits / global distribution**
   `p_gdist` rồi gửi lại cho mọi client.
4. **Personalization.** Tầng classifier `p` được tách riêng khỏi phần feature
   extractor: phần dùng chung học global, phần `p` giữ cá nhân hoá cho từng client
   (adaptive personalized FL).

Nhờ trao đổi logits thay vì tham số, phương pháp này **hỗ trợ client có kiến trúc
khác nhau**. Nó cũng có chế độ aggregation tùy biến: client chỉ đóng góp logits trên
tập dữ liệu public đã định nghĩa trước.

### 5.3. Bản build lại — giữ gì, đổi gì

| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | dữ liệu IoV | **CICIoT2023**, 25 feature, 34 lớp | **Khác** |
| Model | LDwCBN (depthwise conv + bottleneck + temporal) | **GRU thuần 2 lớp 128 units** | **Khác** |
| Loại model | đồng nhất | **HETEROGENEOUS**: 5 client GRU thuần + 5 client **GRU + LDwCBN-lite** | **Đúng tinh thần dị thể** |
| Public dataset | đã định nghĩa sẵn | **5000 mẫu lấy từ `global_test_data.csv`** | **⚠️ Rò rỉ dữ liệu** |
| FedDistillation | gửi logits, tổng hợp `p_gdist` | gửi logits, tổng hợp logits, **chưng cất lại** | Giữ nguyên |
| Loss | CE + λ·KL | `0.5·CE + 0.5·KL·T²`, **T = 2,0** | Giữ nguyên |
| Personalization | tách classifier riêng | không tách | **Bỏ** |
| Quantization | — | **dynamic INT8** | **Thêm** |
| Số round | tới hội tụ | **10** | **Khác (chuẩn hoá)** |

**Điểm lệch đáng chú ý so với bài gốc:**

- **Rò rỉ dữ liệu (data leakage).** Public dataset dùng cho chưng cất được lấy
  trực tiếp từ chính tập test (`global_test_data.csv`, `sample(n=5000,
  random_state=42)`). Kết quả của phương pháp này **bị lạm phát** vì quá trình
  distillation đã nhìn thấy dữ liệu test. Bất kỳ so sánh nào lấy số của phương pháp
  5 làm "thắng" đều không công bằng với các phương pháp còn lại.
- Personalization (tách classifier riêng) bị bỏ trong bản build lại.

### 5.4. Có dùng được cho mô hình không đồng nhất không?

**Kết luận: **Có** — đây là phương pháp duy nhất trong số năm phương pháp hỗ trợ
đầy đủ dị thể.**

Vì server chỉ nhận **logits trên public data** (cùng số lớp, cùng số mẫu) chứ không
nhận tham số, nên:

- **Khác kích thước mô hình:** ✓ có. Client to nhỏ thế nào cũng chỉ cần output logits.
- **Khác loại mô hình:** ✓ có. Trong bản build lại, một nửa client là GRU thuần,
  một nửa là GRU + LDwCBN-lite, vẫn chưng cất chéo bình thường.

### 5.5. Kết quả đo được

- Thư mục run: [`GRU_Lightweight_Hetero_FedDistillation/output`](GRU_Lightweight_Hetero_FedDistillation/output)
- Checkpoint round 1: `lightweight_heterogeneous_quantized_r1.pth`, **43.549 byte (~42,5 KB)** — INT8
- File kết quả: [`federated_learning_results.csv`](GRU_Lightweight_Hetero_FedDistillation/output/federated_learning_results.csv)

*Bảng — Hetero-FedDistillation · GRU (dị thể 5+5) · 10 metrics theo từng round*

| Round | accuracy | macro_P | micro_P | weighted_P | macro_R | micro_R | weighted_R | macro_F1 | micro_F1 | weighted_F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2662 | 0.0383 | 0.2662 | 0.1085 | 0.0605 | 0.2662 | 0.2662 | 0.0331 | 0.2662 | 0.1451 |
| 2 | 0.3389 | 0.0448 | 0.3389 | 0.1763 | 0.0843 | 0.3389 | 0.3389 | 0.0507 | 0.3389 | 0.2096 |
| 3 | 0.4734 | 0.1150 | 0.4734 | 0.3003 | 0.1337 | 0.4734 | 0.4734 | 0.0987 | 0.4734 | 0.3443 |
| 4 | 0.5174 | 0.1809 | 0.5174 | 0.4053 | 0.1606 | 0.5174 | 0.5174 | 0.1323 | 0.5174 | 0.4071 |
| 5 | 0.5824 | 0.1674 | 0.5824 | 0.4585 | 0.1879 | 0.5824 | 0.5824 | 0.1623 | 0.5824 | 0.4863 |
| 6 | 0.6033 | 0.1990 | 0.6033 | 0.4923 | 0.2127 | 0.6033 | 0.6033 | 0.1844 | 0.6033 | 0.5156 |
| 7 | 0.6211 | 0.2116 | 0.6211 | 0.5243 | 0.2267 | 0.6211 | 0.6211 | 0.1943 | 0.6211 | 0.5394 |
| 8 | 0.6298 | 0.2183 | 0.6298 | 0.5464 | 0.2360 | 0.6298 | 0.6298 | 0.1998 | 0.6298 | 0.5559 |
| 9 | 0.6336 | 0.2297 | 0.6336 | 0.5607 | 0.2408 | 0.6336 | 0.6336 | 0.2015 | 0.6336 | 0.5611 |
| 10 | 0.6422 | 0.2722 | 0.6422 | 0.5953 | 0.2502 | 0.6422 | 0.6422 | 0.2134 | 0.6422 | 0.5722 |

> **⚠️ Cảnh báo:** kết quả trên bị ảnh hưởng bởi việc public set lấy từ test set
> (xem 5.3). Cột `accuracy` cao nhất bảng (0.6422) **không thể coi là thắng thật sự**
> khi so với các phương pháp khác.

---

## 6. Zero-shot Pruning + PTD Quantization

### 6.1. Bài báo

- **Tên:** Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS)
- **Tác giả:** — (chi tiết trong bài báo)
- **Năm:** 2024
- **Nơi xuất bản:** IEEE Access, vol. 12, pp. 173251–173264 (2024)
- **File trong repo:** [`GRU_Lightweight_Zero-shot Pruning + PTD Quantization/Lightweight_Federated_Learning_for_Efficient_Network_Intrusion_Detection (1).pdf`](GRU_Lightweight_Zero-shot%20Pruning%20%2B%20PTD%20Quantization/Lightweight_Federated_Learning_for_Efficient_Network_Intrusion_Detection%20(1).pdf)

### 6.2. Quy trình huấn luyện của bài báo gốc

Lightweight-Fed-NIDS kết hợp **pruning + quantization** ngay trong pipeline FL để
giảm kích thước mô hình truyền qua mạng:

1. **Pruning.** Trọng số gần 0 bị đặt bằng 0 theo một tỷ lệ đặt trước (chỉ giữ lại
   các trọng số quan trọng nhất theo trị tuyệt đối). Layer cuối (classifier) được
   giữ nguyên để không mất nhãn phân loại.
2. **Zero-shot / post-hoc.** Việc nén được làm **sau khi mô hình đã huấn luyện xong**
   (không cần huấn luyện lại), đúng nghĩa zero-shot.
3. **Quantization (PTD).** Post-training dynamic quantization: chuyển trọng số
   FP32 → INT8 động, tính lại scale/zero-point cho từng tensor khi forward.
4. **Tổng hợp FL.** Server vẫn dùng FedAvg trọng số theo `n_k/n` trên mô hình đã
   prune/quantize của từng client.

### 6.3. Bản build lại — giữ gì, đổi gì

| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | IDS (lightweight NIDS) | **CICIoT2023**, 25 feature, 34 lớp | **Khác** |
| Model | — | **GRU 2 lớp 128 units** | **Khác** |
| Pruning | magnitude pruning, giữ FC đầu ra | MagnitudePruner ratio **0.20**, **giữ FC đầu ra** | Giữ nguyên |
| Quantization | PTD | **dynamic INT8** (FP32→INT8) | Giữ nguyên |
| Tổng hợp | FedAvg trọng số | FedAvg trọng số `n_k/n` | Giữ nguyên |
| Số round | tới hội tụ | **10** | **Khác (chuẩn hoá)** |

### 6.4. Có dùng được cho mô hình không đồng nhất không?

> **Ghi chú nguồn gốc:** giống FedAvg, đây là hạn chế **có sẵn trong bài báo gốc**.
> Lightweight-Fed-NIDS nhắc "heterogeneous hardware and software" chỉ về **phần
> cứng triển khai** (bộ nhớ, sức mạnh xử lý khác nhau giữa các node), còn mô hình là
> **một cấu trúc dùng chung** cho mọi client với tổng hợp FedAvg — bài gốc không hỗ
> trợ client dùng kiến trúc khác nhau.

**Kết luận: **Không**.

- **Khác kích thước mô hình:** không. Mọi client phải có cùng shape tham số để FedAvg
  cộng được.
- **Khác loại mô hình:** không.

Điểm mạnh của phương pháp nằm ở **mặt kích thước**, không phải dị thể: model gửi lên
server chỉ ~170 KB (INT8), nhỏ hơn baseline ~3,8 lần.

### 6.5. Kết quả đo được

- Thư mục run: [`GRU_Lightweight_Zero-shot Pruning + PTD Quantization/output`](GRU_Lightweight_Zero-shot%20Pruning%20%2B%20PTD%20Quantization/output)
- Checkpoint round 1: `lightweight_gru_quantized_r1.pth`, **173.945 byte (~170 KB)** — INT8
- File kết quả: [`federated_learning_results.csv`](GRU_Lightweight_Zero-shot%20Pruning%20%2B%20PTD%20Quantization/output/federated_learning_results.csv)

*Bảng — Zero-shot Pruning + PTD Quant. · GRU · 10 metrics theo từng round*

| Round | accuracy | macro_P | micro_P | weighted_P | macro_R | micro_R | weighted_R | macro_F1 | micro_F1 | weighted_F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5454 | 0.2521 | 0.5454 | 0.4123 | 0.2120 | 0.5454 | 0.5454 | 0.1827 | 0.5454 | 0.4495 |
| 2 | 0.5897 | 0.2954 | 0.5897 | 0.5016 | 0.3340 | 0.5897 | 0.5897 | 0.2759 | 0.5897 | 0.4974 |
| 3 | 0.6064 | 0.3217 | 0.6064 | 0.5220 | 0.3696 | 0.6064 | 0.6064 | 0.3089 | 0.6064 | 0.5167 |
| 4 | 0.6078 | 0.3433 | 0.6078 | 0.5396 | 0.3791 | 0.6078 | 0.6078 | 0.3175 | 0.6078 | 0.5175 |
| 5 | 0.6128 | 0.3940 | 0.6128 | 0.5657 | 0.3987 | 0.6128 | 0.6128 | 0.3425 | 0.6128 | 0.5235 |
| 6 | 0.6224 | 0.4097 | 0.6224 | 0.5843 | 0.4266 | 0.6224 | 0.6224 | 0.3751 | 0.6224 | 0.5352 |
| 7 | 0.6258 | 0.4131 | 0.6258 | 0.5976 | 0.4427 | 0.6258 | 0.6258 | 0.3880 | 0.6258 | 0.5396 |
| 8 | 0.6281 | 0.4202 | 0.6281 | 0.5931 | 0.4538 | 0.6281 | 0.6281 | 0.4021 | 0.6281 | 0.5418 |
| 9 | 0.6295 | 0.4445 | 0.6295 | 0.6770 | 0.4585 | 0.6295 | 0.6295 | 0.4055 | 0.6295 | 0.5428 |
| 10 | 0.6314 | 0.4601 | 0.6314 | 0.6699 | 0.4642 | 0.6314 | 0.6314 | 0.4113 | 0.6314 | 0.5452 |

---

## 7. Zero-shot Pruning

### 7.1. Bài báo

- **Tên:** Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS) — *cùng bài với mục 6*
- **Năm:** 2024
- **Nơi xuất bản:** IEEE Access, vol. 12, pp. 173251–173264 (2024)
- **File trong repo:** [`GRU_Lightweight_Zero-shot Pruning/Lightweight_Federated_Learning_for_Efficient_Network_Intrusion_Detection (1).pdf`](GRU_Lightweight_Zero-shot%20Pruning/Lightweight_Federated_Learning_for_Efficient_Network_Intrusion_Detection%20(1).pdf)

> **Cảnh báo:** phương pháp này và phương pháp 4 dùng chung một bài báo nguồn.
> Đây là **biến thể pruning-only**, không có bước quantization như mục 6.

### 7.2. Quy trình huấn luyện của bài báo gốc

Như mục 6.2 — pipeline **magnitude pruning** sau khi huấn luyện (zero-shot), giữ
lại classifier, **không** kèm quantization:

1. **Pruning.** Trọng số gần 0 bị đặt bằng 0 theo trị tuyệt đối với tỷ lệ cố định
   (giữ FC đầu ra nguyên vẹn).
2. **Zero-shot / post-hoc.** Áp dụng sau huấn luyện, không cần fine-tune lại.
3. **Tổng hợp FL.** FedAvg trọng số theo `n_k/n` trên mô hình đã prune (vẫn FP32).

### 7.3. Bản build lại — giữ gì, đổi gì

| Thành phần | Bài gốc | Bản build lại | Đánh giá |
|---|---|---|---|
| Bộ dữ liệu | IDS (lightweight NIDS) | **CICIoT2023**, 25 feature, 34 lớp | **Khác** |
| Model | — | **GRU 2 lớp 128 units** | **Khác** |
| Pruning | magnitude pruning, giữ FC đầu ra | MagnitudePruner ratio **0.20**, **giữ FC đầu ra** | Giữ nguyên |
| Quantization | PTD | **không có** | **Bỏ (đây là điểm khác biệt vs mục 6)** |
| Tổng hợp | FedAvg trọng số | FedAvg trọng số `n_k/n` | Giữ nguyên |
| Số round | tới hội tụ | **10** | **Khác (chuẩn hoá)** |

### 7.4. Có dùng được cho mô hình không đồng nhất không?

**Kết luận: **Không**.**

Giống hệt mục 6.4 — và cùng lý do: hạn chế này **đã có trong bài báo gốc**
(Lightweight-Fed-NIDS, xem ghi chú mục 6.4). Vẫn là FedAvg trên tham số cùng shape
— không khác kích thước, không khác loại mô hình. Lưu ý: vì prune theo cấu trúc mà
file checkpoint vẫn lưu **FP32 đầy đủ**, nên phương pháp này gần như **không giảm
dung lượng truyền thông** (checkpoint 656 KB ≈ baseline 655 KB).

### 7.5. Kết quả đo được

- Thư mục run: [`GRU_Lightweight_Zero-shot Pruning/output`](GRU_Lightweight_Zero-shot%20Pruning/output)
- Checkpoint round 1: `lightweight_gru_r1.pth`, **656.437 byte (~641 KB)** — FP32
- File kết quả: [`federated_learning_results.csv`](GRU_Lightweight_Zero-shot%20Pruning/output/federated_learning_results.csv)

*Bảng — Zero-shot Pruning · GRU · 10 metrics theo từng round*

| Round | accuracy | macro_P | micro_P | weighted_P | macro_R | micro_R | weighted_R | macro_F1 | micro_F1 | weighted_F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5370 | 0.2412 | 0.5370 | 0.4326 | 0.2291 | 0.5370 | 0.5370 | 0.1898 | 0.5370 | 0.4318 |
| 2 | 0.5821 | 0.2892 | 0.5821 | 0.4952 | 0.3262 | 0.5821 | 0.5821 | 0.2608 | 0.5821 | 0.4838 |
| 3 | 0.6050 | 0.3233 | 0.6050 | 0.5327 | 0.3720 | 0.6050 | 0.6050 | 0.3110 | 0.6050 | 0.5139 |
| 4 | 0.6104 | 0.4019 | 0.6104 | 0.6194 | 0.3929 | 0.6104 | 0.6104 | 0.3291 | 0.6104 | 0.5200 |
| 5 | 0.6181 | 0.4292 | 0.6181 | 0.6680 | 0.4241 | 0.6181 | 0.6181 | 0.3608 | 0.6181 | 0.5302 |
| 6 | 0.6199 | 0.4522 | 0.6199 | 0.6812 | 0.4331 | 0.6199 | 0.6199 | 0.3768 | 0.6199 | 0.5355 |
| 7 | 0.6241 | 0.4348 | 0.6241 | 0.6751 | 0.4486 | 0.6241 | 0.6241 | 0.3896 | 0.6241 | 0.5400 |
| 8 | 0.6265 | 0.4740 | 0.6265 | 0.6869 | 0.4580 | 0.6265 | 0.6265 | 0.3999 | 0.6265 | 0.5430 |
| 9 | 0.6291 | 0.4774 | 0.6291 | 0.6904 | 0.4639 | 0.6291 | 0.6291 | 0.4062 | 0.6291 | 0.5462 |
| 10 | 0.6298 | 0.4763 | 0.6298 | 0.6904 | 0.4679 | 0.6298 | 0.6298 | 0.4086 | 0.6298 | 0.5470 |

---

## 8. So sánh chéo năm phương pháp

### 8.1. Bảng so sánh tổng hợp — round cuối (round 10)

| Phương pháp | accuracy | macro_F1 | weighted_F1 | Checkpoint round 1 | Về bản chất |
|---|---:|---:|---:|---|---|
| FedAvg (baseline) | **0.6326** | **0.4155** | 0.5518 | 655.281 B (~640 KB) | chuẩn, không nén |
| CustomAgg + Unstruct. Mag. | 0.6284 | 0.3915 | **0.5573** | 655.154 B (~640 KB) | cắt rời rạc, tổng hợp theo mask |
| Hetero-FedDistillation | 0.6422 ⚠️ | 0.2134 | 0.5722 | 43.549 B (~42,5 KB) | chưng cất logits, **dò rỉ test** |
| ZS Prune + PTD Quant. | 0.6314 | 0.4113 | 0.5452 | 173.945 B (~170 KB) | prune 20% + INT8 |
| ZS Prune | 0.6298 | 0.4086 | 0.5470 | 656.437 B (~641 KB) | prune 20%, không quantize |

*⚠️ = con số bị lạm phát do dò rỉ dữ liệu (mục 5.3).*

**Đọc nhanh (bỏ qua Hetero vì bị dò rỉ):**

- **accuracy cao nhất:** FedAvg 0.6326, sát sau là ZS+PTD 0.6314 (chênh 0,0012).
- **macro_F1 cao nhất:** FedAvg 0.4155 (ZS+PTD 0.4113 — chênh 0,004).
- **weighted_F1 cao nhất:** CustomAgg 0.5573, nhưng kèm **bất ổn định** (round 8→9 tụt).
- **Nhỏ nhất về dung lượng:** Hetero 42,5 KB (bị dò rỉ) → tiếp là ZS+PTD 170 KB.

### 8.2. Đường cong hội tụ (accuracy theo round)

| Round | FedAvg | CustomAgg | Hetero | ZS+PTD | ZS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5417 | 0.5247 | 0.2662 | 0.5454 | 0.5370 |
| 2 | 0.5795 | 0.5859 | 0.3389 | 0.5897 | 0.5821 |
| 3 | 0.6022 | 0.6058 | 0.4734 | 0.6064 | 0.6050 |
| 4 | 0.6111 | 0.6141 | 0.5174 | 0.6078 | 0.6104 |
| 5 | 0.6192 | 0.6164 | 0.5824 | 0.6128 | 0.6181 |
| 6 | 0.6223 | 0.6177 | 0.6033 | 0.6224 | 0.6199 |
| 7 | 0.6224 | 0.6279 | 0.6211 | 0.6258 | 0.6241 |
| 8 | 0.6265 | 0.6141 | 0.6298 | 0.6281 | 0.6265 |
| 9 | 0.6311 | 0.6111 | 0.6336 | 0.6295 | 0.6291 |
| 10 | 0.6326 | 0.6284 | 0.6422 | 0.6314 | 0.6298 |

**Quan sát:**

- **5 phương pháp hội tụ về cùng một vùng** (0.62–0.64) ở cuối quá trình — nén nhẹ
  không làm mất accuracy, điều khác biệt nằm ở **chi phí gửi lên server**.
- **Hetero khởi động rất chậm** (0.27 → 0.64), do chưng cất mềm (T=2) từ mô hình
  chưa hội tụ; nhưng tăng nhanh từ round 5 và vượt cả baseline — con số này không
  đáng tin do dò rỉ.
- **CustomAgg có biến động kỳ lạ:** từ round 7 (0.6279) tụt xuống round 8–9
  (0.6141, 0.6111) rồi bật lại round 10 (0.6284). Đường không ổn định như các
  phương pháp còn lại.
- **ZS+PTD và ZS gần như trùng nhau** về accuracy — quantization INT8 không hại gì
  hiệu năng nhưng giúp file nhỏ ~3,8 lần. Điều này chứng minh **bước quantization
  là lãi toàn phần**.

---

## 9. Phân tích điểm mạnh — điểm yếu

> Phân tích dưới đây được viết **từ chính số liệu đo được** (5 CSV), không phải từ
> tuyên bố của bài báo gốc. Mọi con số đối chiếu đều lấy tại **round 10**.
>
> **Chú giải chỉ số cho toàn bộ phần này:**
> - `acc − macro_F1` = khoảng cách giữa accuracy và macro-F1 → **đo độ thiên vị lớp
>   đa số**. Càng lớn, model càng "ăn điểm" bằng các lớp đông, càng mù lớp hiếm.
> - `weighted_F1 − macro_F1` = mức model thắng ở lớp phổ biến so với lớp hiếm.
> - Truyền thông ước lượng = kích thước checkpoint × 10 client × 10 round
>   (chỉ tính chiều gửi lên, để so sánh tương đối giữa các phương pháp).

### FedAvg (baseline)

**Điểm mạnh**

- **Hội tụ ổn định nhất, đơn điệu tăng.** Accuracy tăng đều **từng round, không một
  lần tụt** (0.5417 → 0.6326). Không có *model drift* dù dữ liệu 34 lớp, non-IID.
- **Cân bằng nhất giữa lớp đông và lớp hiếm.** macro_F1 = 0.4155 là cao nhất trong
  nhóm không bị dò rỉ; `acc − macro_F1` = 0.217 là **nhỏ nhất nhóm** — nghĩa là dù
  vẫn thiên vị lớp đa số, FedAvg là phương pháp ít thiên vị nhất.
- **Là chuẩn vàng.** Mọi phương pháp nén đều phải đo mức mất mát so với
  0.6326 / 0.4155 / 0.5518. Không có kỹ thuật nén nào "ăn gian" được ở đây.

**Điểm yếu**

- **Truyền thông tối đa: 640 KB × 10 client × 10 round ≈ 65,5 MB.** Gửi toàn bộ
  tham số FP32 mỗi round, cả hai chiều, không có khái niệm nén.
- **Không hỗ trợ dị thể.** Mọi client bắt buộc cùng kiến trúc, cùng số tham số.
- **Vẫn mù lớp hiếm ở mức nhất định:** `acc − macro_F1` = 0.217 là nhỏ nhất nhưng
  chưa hề nhỏ (0.63 vs 0.42) — vấn đề cố hữu của phân loại 34 lớp trên dữ liệu lệch.

### CustomAggregation + Unstructured Magnitude

**Điểm mạnh**

- **weighted_F1 cao nhất nhóm: 0.5573**, với **weighted_precision cực cao 0.7888**
  ở round 10 — model tự tin và chính xác trên các lớp phổ biến (nhưng xem điểm yếu).
- **Cơ chế mask linh hoạt cho thiết bị yếu.** Tỷ lệ cắt thay đổi theo client
  (0.10–0.25) và theo round (+0–0.04): client "yếu" cắt nhiều hơn, client khỏe giữ
  nhiều hơn. Tổng hợp chỉ gộp trọng số **chưa bị cắt ở mọi client** → mỗi client vẫn
  có thể học thêm từ global phần tham số nó giữ lại.
- **Ý nghĩa thực tế về năng lượng.** Cắt rời rạc đặt ~10–29% trọng số về 0 giúp
  giảm phép toán nhân nếu phần cứng hỗ trợ thưa — giảm năng lượng trên IoT.

**Điểm yếu**

- **Bất ổn định nghiêm trọng — phương pháp duy nhất có chuỗi giảm 2 round liên
  tiếp:** accuracy tụt 0.6279 (r7) → 0.6141 (r8) → 0.6111 (r9) rồi bật lại 0.6284
  (r10). Nguyên nhân nằm ở chính cơ chế: mask **tính lại mỗi round** thay đổi vị trí
  trọng số được cắt, nên server tổng hợp trên các bộ "lỗ" khác nhau giữa các round —
  kiến thức học ở round này bị vứt bỏ ở round sau.
- **Không tiết kiệm truyền thông thực tế.** Checkpoint 655 KB ≈ baseline. Prune
  *unstructured* chỉ đặt trọng số về 0 nhưng `.pth` vẫn lưu **đủ tensor FP32 + mask**
  (không lưu dạng thưa), nên băng thông ≈ 65,5 MB như FedAvg — nén "trên giấy".
- **Cap 0.40 là ảo.** Công thức `0.10 + (cid%4)·0.05 + (r%3)·0.02` thực tế chỉ đạt
  tối đa `0.25 + 0.04 = 0.29`, không bao giờ chạm 0.40. Hơn nữa DRL thật của bài báo
  (chọn ρ tối ưu 65,8–68,4%) bị thay bằng công thức giả lập → con số không phản ánh
  khả năng của OptiFLIDS.
- **macro_F1 thấp hơn baseline (0.3915 vs 0.4155):** cắt theo quantile làm mất
  thông tin phân biệt của lớp hiếm; `acc − macro_F1` = 0.237 lớn hơn FedAvg.
- **Không hỗ trợ dị thể kiến trúc** (chỉ khác mask, không khác model).

### Hetero-FedDistillation

**Điểm mạnh**

- **Duy nhất hỗ trợ dị thể đầy đủ.** Đúng 5 client **LDwCBN+GRU** (stem Conv1d →
  BottleneckBlock có residual → GRU 64 units 1 lớp) + 5 client **GRU thuần** (GRU 64
  units 2 lớp) vẫn chưng cất chéo bình thường vì server chỉ trao đổi logits, không
  chạm tham số. Server cũng là LDwCBN+GRU.
- **Checkpoint nhỏ nhất tuyệt đối: 43,5 KB (INT8)** — nhỏ ~15× so với baseline.
- **Accuracy round 10 cao nhất toàn bộ thí nghiệm: 0.6422** và weighted_F1 cao nhất
  0.5722 (nhưng bị dò rỉ, xem dưới).

**Điểm yếu**

- **Rò rỉ dữ liệu nghiêm trọng.** Public set 5000 mẫu được lấy thẳng từ
  `global_test_data.csv` (`sample(n=5000, random_state=42)`) → global logits được
  chưng cất từ **chính dữ liệu test**. Mọi con số của phương pháp này bị lạm phát;
  không được dùng để đối sánh accuracy.
- **macro_F1 thảm họa: 0.2134 — thấp nhất toàn bộ.** `acc − macro_F1` = **0.429**
  gần gấp đôi các phương pháp khác → model gần như **mù với lớp hiếm**, chỉ "ăn
  điểm" nhờ lớp đông. Nguyên nhân: chưng cất về logits trung bình của nhiều client
  làm mờ nhạt phân bố của lớp hiếm (chúng chỉ chiếm ít trong soft-target), và bản
  build lại bỏ bước personalization vốn giúp giữ kiến thức cục bộ.
- **Hội tụ chậm nhất nhóm.** Chỉ đạt 0.60 ở round 6 (các phương pháp khác round 3);
  round 1 chỉ 0.2662. Logits mềm T=2 từ model chưa hội tụ tạo nhãn mờ, khởi động rất
  chậm.
- **Lợi thế truyền thông bị thổi phồng bởi chi phí logits.** Con số 43,5 KB là
  checkpoint INT8, nhưng bản chất feddistillation phải **gửi logits trên 5000 mẫu ×
  34 lớp** mỗi client mỗi round (≈680 KB FP32, hay 340 KB FP16) — ngang ngửa chính
  model nhỏ. Trong code các bước này chạy cục bộ nên không thấy trong log, nhưng khi
  triển khai thật đây là tải truyền thông chính.

### Zero-shot Pruning + PTD Quantization

**Điểm mạnh**

- **Cân bằng tốt nhất giữa hiệu năng và truyền thông** trong nhóm không dò rỉ:
  accuracy 0.6314 (chênh baseline chỉ **0.0012**), macro_F1 0.4113 (chênh **0.0042**),
  nhưng checkpoint chỉ **170 KB — giảm 3,77×** (17,4 MB vs 65,5 MB truyền thông).
- **Hội tụ ổn định, đơn điệu tăng**, không round nào tụt (0.5454 → 0.6314).
- **Zero-shot:** prune + quantize áp dụng sau huấn luyện, **không cần fine-tune lại**
  → chi phí triển khai rẻ, không tốn thêm vòng FL.
- **Quantization là "free lunch":** so với ZS prune-only cùng tỷ lệ 0.20, bản +PTD
  không hề kém điểm (0.6314 vs 0.6298) mà file nhỏ 3,77×.

**Điểm yếu**

- **Không hỗ trợ dị thể** (FedAvg thuần trên tham số cùng shape).
- **Tỷ lệ prune còn thấp (0.20)** — chưa khai thác hết tiềm năng bài gốc; macro_F1
  (0.4113) vẫn hơi thấp hơn baseline (0.4155).
- **`acc − macro_F1` = 0.220** — độ thiên vị lớp đa số gần bằng baseline, tức nén
  không giúp gì cho bài toán lớp hiếm, cũng chẳng làm hại thêm.

### Zero-shot Pruning

**Điểm mạnh**

- **Chứng minh prune 20% vô hại về hiệu năng:** accuracy 0.6298 ≈ baseline 0.6326
  (chênh 0.0028), hội tụ đơn điệu ổn định.
- **Là thí nghiệm đối chứng hoàn hảo cho mục 6:** nhờ có bản prune-only này, ta tách
  được rõ ràng tác động của pruning (vô hại) và quantization (mang lại toàn bộ lợi
  ích về kích thước).

**Điểm yếu**

- **Không giảm truyền thông gì cả — checkpoint còn LỚN HƠN baseline: 656.437 B vs
  655.281 B.** Lý do: prune đặt trọng số về 0 nhưng state_dict vẫn lưu **đủ tensor
  FP32** (không lưu dạng thưa) → truyền thông ≈ 65,6 MB, ngang FedAvg.
- **Vô nghĩa về mặt lightweight nếu đứng riêng:** 90% trọng số bằng 0 nhưng không
  đổi gì về băng thông, và cũng không nhanh hơn nếu phần cứng không có tính năng
  thưa. "Nén trên giấy" đúng nghĩa.
- **Không hỗ trợ dị thể.**

### Bảng tổng kết đối chiếu

| Tiêu chí | FedAvg | CustomAgg | Hetero ⚠ | ZS+PTD | ZS |
|---|---|---|---|---|---|
| accuracy round 10 | 0.6326 | 0.6284 | 0.6422 ⚠ | 0.6314 | 0.6298 |
| macro_F1 round 10 | **0.4155** | 0.3915 | 0.2134 | 0.4113 | 0.4086 |
| weighted_F1 round 10 | 0.5518 | **0.5573** | 0.5722 ⚠ | 0.5452 | 0.5470 |
| `acc − macro_F1` (thiên vị lớp đa số) | **0.217** | 0.237 | **0.429** | 0.220 | 0.221 |
| Checkpoint round 1 | 655 KB | 655 KB | **43,5 KB** ⚠ | 170 KB | 656 KB |
| Truyền thông ước lượng (10 client × 10 round) | 65,5 MB | 65,5 MB | ~15× nhỏ hơn nhưng +logits ⚠ | **17,4 MB** | 65,6 MB |
| Round đầu đạt accuracy ≥ 0.60 | r3 | r3 | **r6** | r3 | r3 |
| Hội tụ đơn điệu | ✓ | **✗** (tụt r7→r9) | ✗ | ✓ | ✓ |
| Dị thể kích thước model | ✗ | ✗ | **✓** | ✗ | ✗ |
| Dị thể loại model | ✗ | ✗ | **✓** | ✗ | ✗ |
| Có global model dùng được | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rò rỉ dữ liệu | không | không | **CÓ** | không | không |

*⚠ = con số bị ảnh hưởng bởi rò rỉ dữ liệu (mục 5.3), không dùng để đối sánh trực tiếp.*

---

## 10. Kết luận

Trên cùng nền GRU + CICIoT2023 (34 lớp, 10 client, 10 round):

1. **Nén không làm mất hiệu năng.** Pruning 20% và quantization INT8 giữ accuracy
   gần như nguyên vẹn (0.6314 so với baseline 0.6326). Đây là kết quả quan trọng
   nhất: **Lightweight-Fed-NIDS (mục 6) là lựa chọn đáng tin nhất** khi cần giảm
   dung lượng truyền — model 170 KB, giảm ~3,8 lần so với 640 KB, gần như không mất
   điểm.
2. **Quantization mới là bước quyết định.** So sánh mục 6 vs mục 7 cho thấy: prune
   đơn thuần (mục 7) chẳng giảm kích thước file nào (656 KB — thậm chí còn lớn hơn
   baseline 655 KB), còn +PTD (mục 6) giảm mạnh mà không đổi accuracy → bước INT8 là
   "free lunch". Một phương pháp "prune mà không quantize, không lưu dạng thưa" sẽ
   **không giảm được băng thông thực tế** dù điểm số không tệ — đây là cái bẫy cần
   tránh khi đánh giá lightweight.
3. **Dị thể thực sự chỉ đến từ distillation.** Nếu nhiệm vụ yêu cầu client dùng
   kiến trúc khác nhau, chỉ Hetero-FedDistillation đáp ứng được (5 client LDwCBN+GRU
   + 5 client GRU thuần) — nhưng phải sửa lỗi dò rỉ dữ liệu và chấp nhận macro_F1
   yếu trước khi dùng làm phương án chính. Lưu ý thêm: chi phí truyền thông của nó
   là **logits trên public set** (~680 KB FP32 mỗi client mỗi round), không phải
   checkpoint 43,5 KB như vẻ ngoài.
4. **Không phương pháp nào giải quyết được bài toán lớp hiếm.** Cả năm đều có
   `acc − macro_F1` ≥ 0.217; Hetero còn lên tới 0.429. Nếu mục tiêu là phát hiện tấn
   công dạng hiếm (thường là đáng quan tâm nhất trong NIDS), cả năm đều thiếu —
   đây là giới hạn chung của thiết lập thí nghiệm, không phải của riêng phương pháp
   nén nào.
5. **Baseline vẫn là chuẩn vàng.** FedAvg ổn định nhất, đứng đầu về macro_F1 và
   ít thiên vị lớp đa số nhất. Bất kỳ phương pháp nén nào muốn thuyết phục đều phải
   chứng minh được (i) không tụt accuracy quá mức chấp nhận được, (ii) giảm được
   truyền thông thực tế, và (iii) không làm thiên vị lớp đa số tệ hơn — điều mà
   mục 7 (prune-only) không đạt (ii), CustomAgg không đạt (i) do bất ổn định.
6. **Hai hướng đi tiếp có giá trị nhất:**
   - Nghiên cứu nâng tỷ lệ prune (0.2 → 0.4+ như OptiFLIDS đề xuất ~66%) để xem mức
     nén tối đa trước khi accuracy sụt; nếu dùng mask như CustomAgg thì **đừng tính
     lại mask mỗi round** (nguyên nhân bất ổn định) — cố định như bài gốc.
   - Với Hetero: tách public set ra khỏi test (dùng tập riêng) rồi đánh giá lại;
     thêm bước personalization để cải thiện macro_F1.



