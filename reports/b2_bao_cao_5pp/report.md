# Báo cáo tái hiện năm phương pháp Federated Learning trên CICIoT2023

## Tóm tắt điều hành

Báo cáo này tổng hợp **15 lần chạy đã hoàn tất**, tương ứng với năm phương pháp và ba kịch bản cho mỗi phương pháp. Toàn bộ **111 ảnh** trong các thư mục `output/.../artifacts` đã được sao chép vào `report/images/` và được nhúng theo bố cục dọc, **mỗi ảnh nằm trên một hàng riêng**.

Các kết luận chính:

1. Nếu xét **global model** trên tập test chung, kết quả mạnh nhất về accuracy là **PerFed-SKD – 10 GRU: 72,5690%**; kết quả mạnh nhất về macro-F1 là **FD-IDS – 10 Transformer: 40,4408%**. Hai tiêu chí dẫn đến hai lựa chọn khác nhau vì dữ liệu mất cân bằng mạnh.
2. Trong các phương pháp có personalized model, **PerFed-SKD – 10 GRU** nổi bật nhất với mean personalized accuracy **66,9950%**, mean personalized macro-F1 **30,9894%**, đồng thời có độ lệch giữa client thấp nhất trong nhóm này. Tuy nhiên, PerFed-SKD đã pretrain trên `global_train_data.csv`, nên lợi thế này không thể xem là một so sánh FL thuần túy với các phương pháp không dùng dữ liệu train tập trung ở server.
3. **pFedES** có chi phí truyền thông thấp nhất, chỉ **0,04349 MiB/run**, do chỉ trao đổi proxy extractor 57 tham số. Đổi lại, mean personalized macro-F1 chỉ đạt **8,47–11,95%** sau 10 round.
4. **FedCAPS** giảm số feature từ 25 xuống 7–10 và chỉ truyền **0,1377–0,1434 MiB**, nhưng kết quả test personalized còn thấp. Không có feature nào xuất hiện trong cả ba subset cuối, cho thấy lựa chọn feature nhạy với cách gán kiến trúc cho client.
5. Phương pháp **adaptive mutual distillation với proxy CNN-1D** cho global proxy khá ổn định ở cả ba kịch bản: accuracy **61,83–62,19%**, macro-F1 **35,38–35,66%**. Tuy nhiên, personalized model chỉ đạt mean macro-F1 **16,52–17,27%** và có chênh lệch client lớn.
6. Binary attack F1 đều rất cao, khoảng **98,4–99,5%**, nhưng không phản ánh đầy đủ chất lượng phân loại 34 lớp. Macro-F1 và per-class F1 cho thấy nhiều lớp hiếm vẫn có F1 bằng 0.

> **Lưu ý về cách đọc bảng:** các giá trị tỷ lệ được đổi từ `[0,1]` sang phần trăm. “Điểm phần trăm” là hiệu tuyệt đối giữa hai tỷ lệ phần trăm. Các phương pháp không cùng một đối tượng đánh giá: có phương pháp báo global model, có phương pháp báo global proxy, weighted mean personalized, unweighted mean personalized hoặc pooled personalized. Vì vậy, báo cáo chỉ xếp hạng trực tiếp trong cùng ngữ nghĩa metric; các so sánh khác được ghi rõ là tham khảo.

## 1. Phạm vi, nguồn và kiểm tra tính đầy đủ

### 1.1. Năm bài báo/phương pháp

| Thư mục | Bài báo gốc | Tên dùng trong báo cáo |
|---|---|---|
| `fd_ids_noniid` | [FD-IDS: Federated Learning with Knowledge Distillation for Intrusion Detection in Non-IID IoT Environments](../fd_ids_noniid/sensors-25-04309.pdf) | FD-IDS |
| `perfesskd` | [Personalized Federated Learning for Heterogeneous Edge Device: Self-Knowledge Distillation Approach](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf) | PerFed-SKD |
| `permutation_feature_importance` | [Permutation-Invariant Representation Learning for Robust and Privacy-Preserving Feature Selection](../permutation_feature_importance/2510.05535v3.pdf) | FedCAPS |
| `pfedes` | [pFedES: Generalized Proxy Feature Extractor Sharing for Model Heterogeneous Personalized Federated Learning](../pfedes/00121-YiL.pdf) | pFedES |
| `proxymodel` | [Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf) | Adaptive mutual distillation với proxy CNN-1D |

Phần mô tả rút gọn dùng trong báo cáo chính được đối chiếu theo từng trang/phần ở [methodology_research.md](methodology_research.md).

### 1.2. Ma trận nghiệm thu output

| Phương pháp | Kịch bản 1 | Kịch bản 2 | Kịch bản 3 | Trạng thái | Số ảnh |
|---|---|---|---|---:|---:|
| FD-IDS | 10 CNN-1D | 10 GRU | 10 Transformer | Đủ 3/3, completed | 21 |
| PerFed-SKD | 10 CNN-1D | 10 GRU | 10 Transformer | Đủ 3/3, completed | 21 |
| FedCAPS | 4 GRU + 3 Transformer + 3 CNN-1D | 6 CNN-1D + 3 GRU + 1 Transformer | 5 Transformer + 5 GRU | Đủ 3/3, complete | 27 |
| pFedES | 10 GRU | 10 Transformer | 5 GRU + 5 Transformer | Đủ 3/3, complete | 21 |
| Adaptive mutual proxy | 10 GRU | 10 Transformer | 5 GRU + 5 Transformer | Đủ 3/3, completed | 21 |
| **Tổng** |  |  |  | **15/15 lần chạy** | **111** |

Không phát hiện kịch bản thiếu cần huấn luyện lại. Mỗi lần chạy đều có `summary.json`, `config.json`, history theo round/client/local epoch, classification report, confusion matrix, communication/runtime report, checkpoint và toàn bộ artifact được quy định trong đặc tả tương ứng.

### 1.3. Ngữ nghĩa kết quả chính

| Phương pháp | Đối tượng được dùng làm kết quả chính | Hệ quả khi so sánh |
|---|---|---|
| FD-IDS | Một global model đồng nhất | So sánh trực tiếp được giữa ba kiến trúc của FD-IDS |
| PerFed-SKD | Global model; báo thêm mean personalized | Global và personalized phải tách thành hai bảng |
| FedCAPS | Weighted mean của 10 personalized model; báo thêm unweighted mean và pooled | Weighted mean chịu ảnh hưởng mạnh của client 8 có dữ liệu lớn |
| pFedES | Unweighted mean của 10 personalized model; classification report là pooled trên 10 lượt dự đoán toàn bộ global test | Pooled support bằng 10 lần tập test; pooled macro-F1 khác mean macro-F1 |
| Adaptive mutual proxy | Global proxy CNN-1D; báo thêm mean personalized | Global proxy và personalized model là hai đối tượng khác kiến trúc và mục tiêu |

## 2. Dữ liệu và hợp đồng thí nghiệm chung

### 2.1. CICIoT2023 sau tiền xử lý

Nguồn mô tả dữ liệu: [data_description.md](../fd_ids_noniid/data_description.md).

- Dataset gốc có **45.019.234 mẫu**, 46 feature và 34 nhãn: 33 loại tấn công, 1 lớp `BENIGN`.
- Chia train/test **80/20**: 36.014.594 mẫu train và 9.003.649 mẫu global test.
- Các cột định danh hoặc phương sai rất thấp được lọc trước; XGBoost được fit trên khoảng 10% dữ liệu để chọn **25 feature**.
- `QuantileTransformer` chỉ fit trên train, sau đó dùng lại để transform test, tránh rò rỉ thống kê từ test.
- Train được phân cho 10 client bằng Dirichlet `alpha = 0,2`, đồng thời mỗi client chỉ được phép chứa ngẫu nhiên 5–32 lớp. Kích thước client rất lệch: client nhỏ nhất có 31.520 mẫu, client lớn nhất có 12.759.509 mẫu.
- Không dùng class weighting hoặc resampling trong 15 lần chạy. Đây là nguyên nhân quan trọng khiến accuracy/binary attack F1 cao hơn đáng kể so với macro-F1.

Danh sách 25 feature theo đúng thứ tự:

`ack_flag_number`, `AVG`, `Std`, `UDP`, `fin_count`, `Max`, `TCP`, `syn_count`, `Protocol Type`, `Rate`, `IAT`, `syn_flag_number`, `rst_flag_number`, `Tot sum`, `HTTPS`, `ack_count`, `fin_flag_number`, `HTTP`, `rst_count`, `Header_Length`, `psh_flag_number`, `ICMP`, `Time_To_Live`, `ARP`, `DNS`.

### 2.2. Các kiến trúc downstream dùng chung

| Kiến trúc | Cấu trúc rút gọn | Số tham số |
|---|---|---:|
| GRU | Chuỗi 25×1 → GRU 2 tầng, hidden 64, dropout 0,2 → last step → Linear 64→34 | 40.034 |
| Transformer | Chiếu scalar 1→64 + positional embedding → 2 encoder, 4 head, FFN 128 → mean pool → Linear | 71.010 |
| CNN-1D | Conv 1→32→64→128, BN/ReLU, MaxPool, adaptive average pool → Linear 128→34 | 35.874 |

Tất cả mô hình nhận `[B,25]` hoặc `[B,F]` sau feature selection và trả logits 34 lớp. Seed khởi tạo là 42. Với các đặc tả có hợp đồng khởi tạo liên phương pháp, cùng một họ model bắt đầu từ state xác định và được ghi SHA-256 trong output.

### 2.3. Điểm khác nhau có thể ảnh hưởng tính công bằng

| Yếu tố | FD-IDS | PerFed-SKD | FedCAPS | pFedES | Adaptive mutual proxy |
|---|---|---|---|---|---|
| Local split | 95/5 | 95/5 | 90/10 | 95/5 | 95/5 |
| Optimizer | Adam, 0,001 | SGD, 0,01 | SGD, 0,01 cho classifier | SGD, 0,01 | Adam, 0,001 |
| Rounds/epochs | 10 round × 1 local epoch | Pretrain 1 epoch + 10 round × 1 | Search riêng + final classifier 10 epoch | 10 round; 1 local + 1 proxy epoch/round | 10 round × 1 + final fine-tune 1 epoch |
| Server thấy global train | Không train trên file này | **Có**, pretrain một epoch | Chỉ kiểm tra header/count | Không | Không |
| Multi-GPU | DDP 2 T4 | Client-parallel 2 T4 | Client-parallel 2 T4 | Client-parallel 2 T4 | Client-parallel 2 T4 |
| Aggregation/trao đổi | Full model, weighted FedAvg | Full model của selected clients, unweighted mean | Feature records/feedback | Proxy 57 tham số | Proxy CNN-1D 35.874 tham số |

Do đó, runtime và communication chỉ phản ánh đúng **hợp đồng triển khai hiện tại**, không phải một benchmark phần cứng hoàn toàn cô lập.

## 3. Phương pháp 1 — FD-IDS

Nguồn triển khai: [ARCHITECTURE_AND_OUTPUT_SPEC.md](../fd_ids_noniid/ARCHITECTURE_AND_OUTPUT_SPEC.md).

### 3.1. Phương pháp trong bài báo gốc

FD-IDS được thiết kế cho **phát hiện xâm nhập trong horizontal federated learning**: các client có cùng không gian feature nhưng giữ những tập mẫu khác nhau, phân phối nhãn có thể non-IID. Bài báo giả định server và các client tham gia là tin cậy; mục tiêu riêng tư ở đây là không chuyển raw traffic ra khỏi client, chưa phải một bảo đảm hình thức bằng differential privacy hoặc secure aggregation.

**Chuẩn bị dữ liệu và mô hình.** Bài báo loại các trường IP/port có nguy cơ làm mô hình ghi nhớ định danh, xử lý giá trị lỗi, one-hot feature định danh và Z-score feature số. Mutual Information (MI) giữa từng feature và nhãn được dùng để xếp hạng rồi giữ 25/95 feature của Edge-IIoT và 71/115 feature của N-BaIoT. Classifier chung cho mọi client là DNN gồm năm hidden layer `32→64→128→64→32`, ReLU, đầu ra softmax. Việc dùng cùng kiến trúc và cùng số chiều sau MI là điều kiện để server có thể trung bình trực tiếp tham số.

**Ba tín hiệu tối ưu hóa tại client.** Ở round `t`, server phát global model `w_G^t`. Client `k` sao chép nó thành local student, đồng thời giữ một bản global model đóng băng làm teacher. Student được tối ưu bởi:

$$L_{hard}=CE(y,z_s),$$

$$L_{soft}=T^2 KL\!\left(softmax(z_t/T)\parallel softmax(z_s/T)\right),$$

$$L_{prox}=\frac{\mu}{2}\|w_k-w_G^t\|_2^2,$$

$$L_{total}=\lambda L_{hard}+(1-\lambda)L_{soft}+\beta L_{prox}.$$

Trong đó, `L_hard` học trực tiếp từ nhãn; `L_soft` truyền phân bố xác suất lớp của global teacher thay vì chỉ truyền nhãn đúng/sai; hệ số `T²` bù lại độ co gradient khi tăng temperature. `L_prox` hạn chế local model trôi quá xa điểm khởi tạo toàn cục khi dữ liệu client lệch mạnh. `lambda` cân bằng nhãn cứng và tri thức mềm, còn `beta` điều chỉnh tác động của proximal regularization.

**Một communication round diễn ra như sau:**

1. Server gửi `w_G^t` cho chín client.
2. Mỗi client khởi tạo student từ `w_G^t`, đóng băng teacher và train trên dữ liệu local bằng `L_total`.
3. Client chỉ gửi local weights và số mẫu lên server; raw traffic không được gửi.
4. Server thực hiện sample-weighted FedAvg,

   $$w_G^{t+1}=\sum_{k=1}^{K}\frac{n_k}{\sum_j n_j}w_k^{t+1},$$

   rồi dùng model vừa tổng hợp làm teacher ở round tiếp theo.

Hai cơ chế FedProx và distillation xử lý non-IID theo hai hướng bổ sung: FedProx ràng buộc **không gian tham số**, còn teacher truyền tri thức ở **không gian đầu ra**. Tuy nhiên, FD-IDS vẫn truyền toàn bộ classifier mỗi round, yêu cầu kiến trúc đồng nhất và làm tăng tính toán client do phải chạy thêm teacher/KL. Paper chưa xử lý concept drift, poisoning/membership inference và mới kiểm chứng trên hai dataset IDS.

**Thiết lập gốc.** Bài báo dùng Edge-IIoT và N-BaIoT, 9 client, hai mức Dirichlet `theta=1` (lệch nhẹ) và `theta=0,1` (lệch mạnh), 40 communication round, 2 local epoch/round, batch 128, Adam `lr=0,001`; các hệ số là `mu=0,01`, `lambda=0,5`, `beta=0,1`, `T=3`.

### 3.2. Những thay đổi khi build lại

- Dataset đổi sang CICIoT2023, 10 client, Dirichlet `alpha=0,2`, 34 lớp.
- MI không được chạy lại vì input đã chỉ còn 25 feature; pipeline upstream dùng XGBoost thay MI.
- DNN gốc được thay bằng ba kịch bản homogeneous: 10 CNN-1D, 10 GRU, 10 Transformer.
- Giảm từ 40 xuống 10 round và từ 2 xuống 1 local epoch; tăng batch từ 128 lên 1024.
- Giữ Adam `0,001`, `mu=0,01`, `lambda=0,5`, `beta=0,1`, `T=3`.
- Chạy DDP trên hai T4; batch toàn cục 1024 được chia 512/GPU. Với CNN-1D, BatchNorm được đổi sang SyncBatchNorm.
- Checkpoint chọn theo validation macro-F1, không dùng test để chọn mô hình.

### 3.3. So sánh ba kịch bản

| Kịch bản | Best round | Test accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Binary attack F1 | Runtime | FL communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 CNN-1D | 10 | 59,1893% | 39,6575% | 33,4445% | 29,9830% | 50,9519% | 99,2237% | 114,3 phút | 27,7161 MiB |
| 10 GRU | 10 | 60,1705% | 40,8016% | 37,8727% | 33,6860% | 52,0735% | 99,0162% | 95,0 phút | 30,5435 MiB |
| **10 Transformer** | **10** | **64,2810%** | **45,5235%** | **43,1865%** | **40,4408%** | **56,7152%** | 99,1608% | 178,2 phút | 54,1763 MiB |

Transformer là kịch bản tốt nhất về mọi metric đa lớp chính. So với GRU, Transformer tăng **6,7548 điểm phần trăm macro-F1** nhưng runtime gấp khoảng **1,87 lần** và communication gấp **1,77 lần**. Cả ba best checkpoint đều ở round 10, vì vậy chưa có bằng chứng rằng 10 round đã hội tụ hoàn toàn; đây là nhóm đáng ưu tiên chạy thêm round nếu mục tiêu là tối đa hóa chất lượng.

Ở cấp lớp, Transformer có 16/34 lớp đạt F1 từ 0,5 trở lên, so với 13 lớp ở GRU và 10 lớp ở CNN-1D. Tuy nhiên, cả ba vẫn có 11–12 lớp F1 bằng 0, chủ yếu là các lớp hiếm như `BACKDOOR_MALWARE`, `BROWSERHIJACKING`, `COMMANDINJECTION`, `RECON-PINGSWEEP`, `SQLINJECTION`, `UPLOADING_ATTACK`, `XSS`. Điều này giải thích vì sao binary attack F1 xấp xỉ 99% nhưng macro-F1 chỉ 30–40%.

### 3.4. Toàn bộ ảnh — 10 CNN-1D

**Phân phối lớp**

![FD-IDS CNN-1D class distribution](images/fd_ids_noniid/cnn1d_10_clients/class_distribution.png)

**Accuracy và F1 theo round**

![FD-IDS CNN-1D accuracy F1](images/fd_ids_noniid/cnn1d_10_clients/accuracy_f1_curves.png)

**Loss**

![FD-IDS CNN-1D loss](images/fd_ids_noniid/cnn1d_10_clients/loss_curves.png)

**Confusion matrix**

![FD-IDS CNN-1D confusion matrix](images/fd_ids_noniid/cnn1d_10_clients/confusion_matrix.png)

**Per-class F1**

![FD-IDS CNN-1D per class F1](images/fd_ids_noniid/cnn1d_10_clients/per_class_f1.png)

**Runtime theo round**

![FD-IDS CNN-1D runtime](images/fd_ids_noniid/cnn1d_10_clients/runtime_per_round.png)

**Communication tích lũy**

![FD-IDS CNN-1D communication](images/fd_ids_noniid/cnn1d_10_clients/communication_cumulative.png)

### 3.5. Toàn bộ ảnh — 10 GRU

**Phân phối lớp**

![FD-IDS GRU class distribution](images/fd_ids_noniid/gru_10_clients/class_distribution.png)

**Accuracy và F1 theo round**

![FD-IDS GRU accuracy F1](images/fd_ids_noniid/gru_10_clients/accuracy_f1_curves.png)

**Loss**

![FD-IDS GRU loss](images/fd_ids_noniid/gru_10_clients/loss_curves.png)

**Confusion matrix**

![FD-IDS GRU confusion matrix](images/fd_ids_noniid/gru_10_clients/confusion_matrix.png)

**Per-class F1**

![FD-IDS GRU per class F1](images/fd_ids_noniid/gru_10_clients/per_class_f1.png)

**Runtime theo round**

![FD-IDS GRU runtime](images/fd_ids_noniid/gru_10_clients/runtime_per_round.png)

**Communication tích lũy**

![FD-IDS GRU communication](images/fd_ids_noniid/gru_10_clients/communication_cumulative.png)

### 3.6. Toàn bộ ảnh — 10 Transformer

**Phân phối lớp**

![FD-IDS Transformer class distribution](images/fd_ids_noniid/transformer_10_clients/class_distribution.png)

**Accuracy và F1 theo round**

![FD-IDS Transformer accuracy F1](images/fd_ids_noniid/transformer_10_clients/accuracy_f1_curves.png)

**Loss**

![FD-IDS Transformer loss](images/fd_ids_noniid/transformer_10_clients/loss_curves.png)

**Confusion matrix**

![FD-IDS Transformer confusion matrix](images/fd_ids_noniid/transformer_10_clients/confusion_matrix.png)

**Per-class F1**

![FD-IDS Transformer per class F1](images/fd_ids_noniid/transformer_10_clients/per_class_f1.png)

**Runtime theo round**

![FD-IDS Transformer runtime](images/fd_ids_noniid/transformer_10_clients/runtime_per_round.png)

**Communication tích lũy**

![FD-IDS Transformer communication](images/fd_ids_noniid/transformer_10_clients/communication_cumulative.png)

## 4. Phương pháp 2 — PerFed-SKD

Nguồn triển khai: [ARCHITECTURE_AND_OUTPUT_SPEC.md](../perfesskd/ARCHITECTURE_AND_OUTPUT_SPEC.md).

### 4.1. Phương pháp trong bài báo gốc

PerFed-SKD kết hợp **personalization theo lịch sử của từng client**, **self-knowledge distillation** và **chọn thiết bị có điều kiện**. Khác FD-IDS, teacher của client `m` không phải global model hiện tại mà là personalized model tốt đã được client đó lưu từ lịch sử, ký hiệu `V_m`. Vì vậy mỗi client có một quỹ đạo cá nhân, dù server vẫn duy trì một global model để chia sẻ tri thức giữa các thiết bị.

**Kiến trúc logic.** Ở server, Coordinator điều phối round; Teacher Model lưu global weights; Device Selector quyết định client cần nhận model mới; Aggregator tổng hợp upload; Communication Manager quản lý trao đổi. Ở client, Controller/System Monitor quản lý tài nguyên và trạng thái; Student Model Manager giữ current model cùng historical teacher `V_m`; Data Manager giữ dữ liệu local; Training Optimizer thực hiện cập nhật. Phân tách này nhằm cho phép thiết bị không được chọn vẫn tiếp tục cá nhân hóa cục bộ.

**Mục tiêu học.** Bài toán FL toàn cục vẫn có dạng:

$$\min_w F(w)=\sum_{m=1}^{M}\frac{|D_m|}{|D|}F_m(w).$$

Tại client, current student `w_m^t` học từ nhãn thật và đầu ra của historical teacher:

$$\phi_m(w_m^t)=f_m(w_m^t)+\lambda L\!\left(x(V_m),x(w_m^t)\right).$$

Trong đó `f_m` là cross-entropy, `x(·)` là prediction và paper ký hiệu `L` tổng quát là divergence giữa prediction lịch sử với prediction hiện tại; phương trình gốc không quy định riêng KL hay temperature. `lambda` kiểm soát mức current model phải bảo toàn historical personalized knowledge.

Teacher được đóng băng trong lần local update. Sau khi train, current model được lưu lại thành `V_m` cho round kế tiếp. Cơ chế này đóng vai trò temporal regularization: model mới học dữ liệu hiện tại nhưng bị hạn chế quên tri thức đã tích lũy riêng ở client.

**Luồng của một round:**

1. Server có thể pretrain/khởi tạo global teacher trên một tập dữ liệu định trước rồi phân phối model ban đầu.
2. Device Selector so chất lượng local model với ngưỡng toàn cục. Client yếu được chọn để nhận global weights; client đủ tốt tiếp tục từ personalized state của chính nó.
3. Tất cả client thực hiện local learning với historical self-teacher; chỉ nhóm được chọn gửi model cập nhật về server.
4. Server aggregate các upload để tạo global model mới, cập nhật thống kê accuracy/ngưỡng rồi chọn nhóm cho round tiếp theo.

Selective communication tập trung băng thông vào client đang cần tri thức toàn cục, còn SKD bảo vệ kiến thức cá nhân của client qua các round. Đổi lại, thuật toán vẫn đòi hỏi model có cùng cấu trúc ở các client tham gia aggregation; client được chọn vẫn upload toàn bộ tham số. Paper giữ raw data local nhưng không đưa ra cơ chế bảo vệ hình thức đối với rò rỉ từ model update. Pseudo-code gốc cũng có điểm chưa nhất quán giữa tập client được cộng và mẫu số `|S|`; cách xử lý trong bản dựng được công bố riêng ở mục 4.2.

**Thiết lập gốc.** Thí nghiệm dùng MNIST/EMNIST, 20–80 thiết bị, Dirichlet `alpha=0,001`, 200 global iteration, 20 local epoch, learning rate `0,01`, các participation ratio 20%, 60%, 100% và lặp lại thí nghiệm 10 lần.

### 4.2. Những thay đổi và quyết định khi build lại

- Đổi ảnh MNIST/EMNIST thành 25 feature CICIoT2023 và phân loại 34 lớp.
- Cố định 10 client homogeneous cho từng notebook; không dùng kịch bản mixed vì Algorithm 1 phải aggregate trực tiếp trọng số cùng kiến trúc.
- Server pretrain global model **một epoch trên toàn bộ `global_train_data.csv`**. Dữ liệu này trùng miền dữ liệu đã phân cho client; vì vậy đây là một giả định server-data mạnh và phải được công bố khi so sánh privacy.
- Chạy 10 round, 1 local epoch, batch 1024, SGD `lr=0,01`. Bản dựng cụ thể hóa divergence `L` chưa được paper định nghĩa thành KL với `lambda=1`, `T=1`; đây là lựa chọn tái hiện, không phải hyperparameter được bài báo gốc công bố.
- Round 1 chọn đủ 10 client. Từ round sau, bản dựng dùng mean local validation accuracy của đủ 10 client làm ngưỡng và chọn client thấp hơn mean. Đây là thay đổi so với Algorithm 1 gốc, nơi điều kiện là local accuracy `a_m` thấp hơn global-model accuracy `A`.
- Theo cách diễn giải nhất quán với mục tiêu giảm communication, server lấy unweighted mean **chỉ trên selected clients**. Đây là cách giải quyết điểm không nhất quán trong pseudo-code gốc, nơi tổng dùng `M` nhưng mẫu số dùng `|S|`.
- Chạy client-parallel trên hai T4, không DDP. Checkpoint chọn theo global validation macro-F1.

### 4.3. Global model

| Kịch bản | Best round | Test accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Binary attack F1 | Runtime | Communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 CNN-1D | 10 | 65,4886% | 40,7172% | 37,3237% | 32,7390% | 58,1448% | **99,2486%** | 30,4 phút | **9,9778 MiB** |
| **10 GRU** | **1** | **72,5690%** | 40,4520% | 37,8378% | 36,3608% | **68,6109%** | 98,9930% | **28,8 phút** | 14,0500 MiB |
| 10 Transformer | 3 | 69,9675% | **44,9383%** | **43,4471%** | **40,0462%** | 63,8592% | 99,1213% | 80,2 phút | 21,6705 MiB |

GRU cho accuracy và weighted-F1 tốt nhất; Transformer cho macro-F1 tốt nhất. Best round của GRU là 1 và Transformer là 3, trong khi CNN-1D là 10. Điều này cho thấy global model GRU/Transformer đạt đỉnh validation sớm rồi không cải thiện thêm theo tiêu chí macro-F1, dù các personalized state vẫn tiếp tục thay đổi.

Device selection hoạt động đúng mục tiêu giảm truyền thông:

- CNN-1D: tổng 36 lượt client upload qua 10 round, trung bình 3,6 client/round; communication bằng 36% trường hợp full participation cùng kích thước model.
- GRU: 46 lượt, trung bình 4,6 client/round; communication bằng 46% full participation.
- Transformer: 40 lượt, trung bình 4,0 client/round; communication bằng 40% full participation.

### 4.4. Personalized model và độ đồng đều client

| Kịch bản | Mean accuracy | Min–max accuracy | Mean macro-F1 | Min–max macro-F1 | Độ lệch chuẩn macro-F1 |
|---|---:|---:|---:|---:|---:|
| 10 CNN-1D | 46,4799% | 18,0306–64,9344% | 19,7153% | 5,2549–32,7883% | 8,6181% |
| **10 GRU** | **66,9950%** | **60,8185–70,3600%** | **30,9894%** | **28,6051–35,2363%** | **1,9602%** |
| 10 Transformer | 57,9521% | 21,8008–69,7674% | 30,2976% | 4,5349–39,7421% | 9,0783% |

GRU không chỉ có mean cao nhất mà còn ổn định nhất giữa client. CNN-1D và Transformer có một số client suy giảm rất mạnh khi đánh giá trên **toàn bộ global test**, cho thấy personalized model đã thích nghi với local distribution nhưng chưa tổng quát đồng đều ra phân phối toàn cục. Đây không nhất thiết là thất bại trên chính local test distribution, vì output hiện tại không cung cấp một local test riêng có cùng phân phối với từng client.

Classification report của global model vẫn có 11–12 lớp F1 bằng 0. Transformer và GRU đều có 14 lớp đạt F1 ≥ 0,5; CNN-1D có 13 lớp. Vì vậy, lợi thế accuracy của GRU chủ yếu đến từ các lớp có support lớn, còn Transformer cân bằng hơn giữa các lớp theo macro-F1.

### 4.5. Toàn bộ ảnh — 10 CNN-1D

**Phân phối lớp**

![PerFed-SKD CNN-1D class distribution](images/perfesskd/10_clients_cnn1d/class_distribution.png)

**Accuracy và F1 theo round**

![PerFed-SKD CNN-1D accuracy F1](images/perfesskd/10_clients_cnn1d/accuracy_f1_curves.png)

**Loss**

![PerFed-SKD CNN-1D loss](images/perfesskd/10_clients_cnn1d/loss_curves.png)

**Confusion matrix**

![PerFed-SKD CNN-1D confusion matrix](images/perfesskd/10_clients_cnn1d/confusion_matrix.png)

**Per-class F1**

![PerFed-SKD CNN-1D per class F1](images/perfesskd/10_clients_cnn1d/per_class_f1.png)

**Runtime theo round**

![PerFed-SKD CNN-1D runtime](images/perfesskd/10_clients_cnn1d/runtime_per_round.png)

**Communication tích lũy**

![PerFed-SKD CNN-1D communication](images/perfesskd/10_clients_cnn1d/communication_cumulative.png)

### 4.6. Toàn bộ ảnh — 10 GRU

**Phân phối lớp**

![PerFed-SKD GRU class distribution](images/perfesskd/10_clients_gru/class_distribution.png)

**Accuracy và F1 theo round**

![PerFed-SKD GRU accuracy F1](images/perfesskd/10_clients_gru/accuracy_f1_curves.png)

**Loss**

![PerFed-SKD GRU loss](images/perfesskd/10_clients_gru/loss_curves.png)

**Confusion matrix**

![PerFed-SKD GRU confusion matrix](images/perfesskd/10_clients_gru/confusion_matrix.png)

**Per-class F1**

![PerFed-SKD GRU per class F1](images/perfesskd/10_clients_gru/per_class_f1.png)

**Runtime theo round**

![PerFed-SKD GRU runtime](images/perfesskd/10_clients_gru/runtime_per_round.png)

**Communication tích lũy**

![PerFed-SKD GRU communication](images/perfesskd/10_clients_gru/communication_cumulative.png)

### 4.7. Toàn bộ ảnh — 10 Transformer

**Phân phối lớp**

![PerFed-SKD Transformer class distribution](images/perfesskd/10_clients_transformer/class_distribution.png)

**Accuracy và F1 theo round**

![PerFed-SKD Transformer accuracy F1](images/perfesskd/10_clients_transformer/accuracy_f1_curves.png)

**Loss**

![PerFed-SKD Transformer loss](images/perfesskd/10_clients_transformer/loss_curves.png)

**Confusion matrix**

![PerFed-SKD Transformer confusion matrix](images/perfesskd/10_clients_transformer/confusion_matrix.png)

**Per-class F1**

![PerFed-SKD Transformer per class F1](images/perfesskd/10_clients_transformer/per_class_f1.png)

**Runtime theo round**

![PerFed-SKD Transformer runtime](images/perfesskd/10_clients_transformer/runtime_per_round.png)

**Communication tích lũy**

![PerFed-SKD Transformer communication](images/perfesskd/10_clients_transformer/communication_cumulative.png)

## 5. Phương pháp 3 — FedCAPS

Nguồn triển khai: [ARCHITECTURE_AND_OUTPUT_SPEC.md](../permutation_feature_importance/ARCHITECTURE_AND_OUTPUT_SPEC.md).

### 5.1. Phương pháp trong bài báo gốc

FedCAPS là **federated feature selection**, không phải thuật toán FedAvg classifier. Đầu ra liên kết giữa các client là một tập feature dùng chung; mỗi client có thể dùng tập đó để huấn luyện downstream model của mình. Bài toán khó nằm ở chỗ một subset là một **tập không có thứ tự**, số tổ hợp tăng theo cấp số nhân và chất lượng của cùng subset có thể khác nhau giữa các client.

**Dữ liệu trung gian thay cho raw rows.** Mỗi client dùng một local collector (trong paper là MARLFS) để thử các tập feature và tạo record `(f_i,v_i)`: `f_i` là danh sách feature ID, `v_i` là hiệu năng downstream thu được trên dữ liệu local. Client chỉ chuyển record, score và thông tin quy mô mẫu cần cho weighting; raw feature values không rời thiết bị. Đây là ranh giới dữ liệu của thiết kế, không đồng nghĩa với một chứng minh differential privacy vì feature ID và score vẫn là tín hiệu thống kê về dữ liệu local.

**Biểu diễn bất biến hoán vị.** Server huấn luyện encoder–decoder để ánh xạ một subset có độ dài biến đổi thành embedding cố định rồi tái tạo membership của nó. Multihead Attention Block được viết:

$$H=LN(Q+MultiHead(Q,K,V)),\qquad MAB(Q,K,V)=LN(H+rFF(H)).$$

Encoder không dùng positional encoding, nhờ đó hoán vị thứ tự feature không làm thay đổi biểu diễn mong muốn. Induced Set Attention Block (ISAB) đưa vào `M` inducing points để giảm chi phí self-attention từ `O(N^2)` xuống `O(NM)`; paper xếp chồng hai ISAB. Decoder dùng Pooling by Multihead Attention (PMA) với learned seed vectors, sau đó MAB và row-wise feed-forward để dự đoán lại tập feature. Encoder–decoder được tối ưu bằng negative log-likelihood tái tạo subset.

**Tìm kiếm subset bằng PPO.** Những record có chất lượng cao nhất được lấy làm seed trong latent space. Actor nhận embedding trạng thái `E` và đề xuất embedding mới `E'`; decoder biến `E'` thành candidate subset. Critic ước lượng discounted return. Actor dùng clipped PPO surrogate để tránh cập nhật policy quá lớn, còn critic tối thiểu hóa sai số bình phương với return. Reward kết hợp hai mục tiêu: tăng downstream performance và giảm số feature, nên phương pháp có thể ưu tiên một subset gọn hơn dù score tuyệt đối chưa cao nhất.

**Phản hồi liên client và sample-aware aggregation.** Candidate được gửi về các client để đánh giá định kỳ. Nếu client `c` có `|D_c|` mẫu thì:

$$W_c=\frac{|D_c|}{\sum_j|D_j|},\qquad v_{global}(f)=\sum_c W_c\,v_c(f).$$

Weighted score làm tín hiệu reward toàn cục, ngăn client rất nhỏ có ảnh hưởng ngang client chứa phần lớn dữ liệu. Giữa các lần đánh giá thật, critic cung cấp ước lượng để giảm số lượt giao tiếp. Toàn bộ encoder, decoder, actor và critic nằm ở server; classifier weights/gradients không phải đối tượng được aggregate.

**Quy trình tổng thể:** client thu record local → server học set representation → chọn top-`K` seed → PPO sinh candidate → client chấm candidate → server tổng hợp score theo số mẫu → lặp tìm kiếm → phát subset cuối cho client train downstream model. Điểm mạnh là hỗ trợ model phía client không đồng nhất và payload nhỏ; chi phí được chuyển sang giai đoạn thu record, huấn luyện representation và RL search. Chất lượng subset cũng phụ thuộc trực tiếp vào evaluator dùng để tạo score.

**Thiết lập gốc.** CAPS/FedCAPS được đánh giá trên 14 dataset gồm binary classification, multiclass classification và regression; downstream evaluator chính là Random Forest với five-fold cross-validation hoặc holdout. Paper so với 12 baseline feature-selection trong bối cảnh centralized và 4 baseline trong bối cảnh FL. Vì vậy metric cốt lõi của paper là chất lượng/độ gọn của subset, tính bất biến hoán vị và chi phí truyền record/feedback, không phải accuracy của một global neural classifier.

### 5.2. Những thay đổi khi build lại

- Dataset đổi thành CICIoT2023 với đúng 25 feature upstream; FedCAPS tiếp tục chọn subset nhỏ hơn từ 25 feature này.
- Tạo ba kịch bản downstream model không đồng nhất. Classifier parameters không FedAvg và không truyền lên server.
- Mỗi client thu 300 MARLFS record; mỗi record được hoán vị 25 lần.
- Encoder gồm 2 ISAB, 4 head, embedding 128, 32 inducing point, 32 PMA seed, batch 64, learning rate 0,001.
- PPO dùng top 25 seed, 10 search epoch × 1.000 step, actor `3e-4`, critic `1e-3`, reward trade-off `lambda=0,1`.
- Reward triển khai compactness `1-|f|/25`; subset cuối chọn theo reward, không nhất thiết theo weighted Micro-F1 cao nhất.
- Search pool giới hạn 100.000 mẫu/client; candidate evaluator chỉ train 1 epoch. Final personalized classifier train 10 epoch, SGD 0,01, batch 1024; local split 90/10.
- Ba classifier neural thay Random Forest của thí nghiệm gốc. Đây là thay đổi đáng kể vì chất lượng subset phụ thuộc downstream evaluator.

### 5.3. Subset được chọn

| Kịch bản | Phân bổ model | Số feature | Feature cuối | Weighted Micro-F1 khi search | Weighted Macro-F1 khi search |
|---|---|---:|---|---:|---:|
| S1 | C1–4 GRU; C5–7 Transformer; C8–10 CNN-1D | 10 | UDP, Rate, IAT, rst_flag_number, Tot sum, HTTPS, fin_flag_number, Header_Length, Time_To_Live, ARP | **71,1814%** | **6,6425%** |
| S2 | C1–6 CNN-1D; C7–9 GRU; C10 Transformer | 7 | UDP, fin_count, syn_count, Rate, IAT, ICMP, DNS | 66,8681% | 4,8200% |
| S3 | C1–5 Transformer; C6–10 GRU | 7 | AVG, Max, syn_count, syn_flag_number, rst_flag_number, rst_count, ICMP | 60,3019% | 3,4823% |

Không có feature nào nằm trong giao của cả ba subset. Sáu feature xuất hiện trong hai kịch bản là `IAT`, `ICMP`, `Rate`, `UDP`, `rst_flag_number`, `syn_count`. Điều này cho thấy feature selection không chỉ phụ thuộc dữ liệu mà còn phụ thuộc assignment của downstream architecture.

Encoder–decoder hội tụ ở epoch 13 trong cả ba run, validation reconstruction loss xấp xỉ `3,77e-8–4,01e-8`; permutation invariance max absolute error khoảng `4,77e-6–5,72e-6`, nằm trong tolerance `1e-5`. Như vậy thành phần bất biến hoán vị đạt hợp đồng số học.

### 5.4. Kết quả test personalized

Metric chính của FedCAPS là **weighted mean theo số mẫu local-train**.

| Kịch bản | Weighted accuracy | Weighted macro-F1 | Weighted-F1 | Unweighted accuracy | Unweighted macro-F1 | Pooled macro-F1 | Runtime | Communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **S1: 4G+3T+3C** | 34,0503% | **13,2762%** | 24,2026% | 25,7233% | **8,3564%** | **14,9473%** | 56,8 phút | 0,14338 MiB |
| **S2: 6C+3G+1T** | **35,4991%** | 10,4478% | **25,5629%** | **27,9071%** | 7,9262% | 13,9975% | **43,2 phút** | 0,14086 MiB |
| **S3: 5T+5G** | 34,1934% | 11,0488% | 24,3800% | 26,6656% | 8,2356% | 14,9234% | 53,4 phút | **0,13766 MiB** |

S2 có weighted/unweighted accuracy cao nhất và nhanh nhất; S1 có macro-F1 cao nhất. Chênh lệch 7–8 điểm phần trăm giữa weighted và unweighted accuracy cho thấy các client lớn có performance cao hơn hoặc có ảnh hưởng mạnh hơn. Client-level accuracy trải từ **0,21% đến 55,47%** tùy kịch bản; mean personalized macro-F1 chỉ 7,93–8,36%.

Kết quả search validation cao hơn rõ rệt so với global test. Các nguyên nhân có thể đọc trực tiếp từ hợp đồng thí nghiệm gồm: candidate evaluator chỉ train 1 epoch trên search pool tối đa 100.000 mẫu/client; reward đặt 90% trọng số lên compactness khi `lambda=0,1`; subset được chọn theo reward thay vì weighted Micro-F1 tối đa; và local non-IID validation không đại diện đầy đủ cho global test. Đây là các giả thuyết từ cấu hình, chưa phải kết luận nhân quả vì chưa có ablation.

### 5.5. Toàn bộ ảnh — S1: 4 GRU + 3 Transformer + 3 CNN-1D

**Phân phối lớp**

![FedCAPS S1 class distribution](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/class_distribution.png)

**Accuracy và F1**

![FedCAPS S1 accuracy F1](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/accuracy_f1_curves.png)

**Loss**

![FedCAPS S1 loss](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/loss_curves.png)

**Confusion matrix**

![FedCAPS S1 confusion matrix](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/confusion_matrix.png)

**Per-class F1**

![FedCAPS S1 per class F1](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/per_class_f1.png)

**Runtime**

![FedCAPS S1 runtime](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/runtime_per_round.png)

**Communication**

![FedCAPS S1 communication](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/communication_cumulative.png)

**Độ quan trọng feature đã chọn**

![FedCAPS S1 feature importance](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/selected_feature_importance.png)

**Tìm kiếm kích thước subset**

![FedCAPS S1 subset search](images/permutation_feature_importance/01_4gru_3transformer_3cnn1d/subset_size_search.png)

### 5.6. Toàn bộ ảnh — S2: 6 CNN-1D + 3 GRU + 1 Transformer

**Phân phối lớp**

![FedCAPS S2 class distribution](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/class_distribution.png)

**Accuracy và F1**

![FedCAPS S2 accuracy F1](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/accuracy_f1_curves.png)

**Loss**

![FedCAPS S2 loss](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/loss_curves.png)

**Confusion matrix**

![FedCAPS S2 confusion matrix](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/confusion_matrix.png)

**Per-class F1**

![FedCAPS S2 per class F1](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/per_class_f1.png)

**Runtime**

![FedCAPS S2 runtime](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/runtime_per_round.png)

**Communication**

![FedCAPS S2 communication](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/communication_cumulative.png)

**Độ quan trọng feature đã chọn**

![FedCAPS S2 feature importance](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/selected_feature_importance.png)

**Tìm kiếm kích thước subset**

![FedCAPS S2 subset search](images/permutation_feature_importance/02_6cnn1d_3gru_1transformer/subset_size_search.png)

### 5.7. Toàn bộ ảnh — S3: 5 Transformer + 5 GRU

**Phân phối lớp**

![FedCAPS S3 class distribution](images/permutation_feature_importance/03_5transformer_5gru/class_distribution.png)

**Accuracy và F1**

![FedCAPS S3 accuracy F1](images/permutation_feature_importance/03_5transformer_5gru/accuracy_f1_curves.png)

**Loss**

![FedCAPS S3 loss](images/permutation_feature_importance/03_5transformer_5gru/loss_curves.png)

**Confusion matrix**

![FedCAPS S3 confusion matrix](images/permutation_feature_importance/03_5transformer_5gru/confusion_matrix.png)

**Per-class F1**

![FedCAPS S3 per class F1](images/permutation_feature_importance/03_5transformer_5gru/per_class_f1.png)

**Runtime**

![FedCAPS S3 runtime](images/permutation_feature_importance/03_5transformer_5gru/runtime_per_round.png)

**Communication**

![FedCAPS S3 communication](images/permutation_feature_importance/03_5transformer_5gru/communication_cumulative.png)

**Độ quan trọng feature đã chọn**

![FedCAPS S3 feature importance](images/permutation_feature_importance/03_5transformer_5gru/selected_feature_importance.png)

**Tìm kiếm kích thước subset**

![FedCAPS S3 subset search](images/permutation_feature_importance/03_5transformer_5gru/subset_size_search.png)

## 6. Phương pháp 4 — pFedES

Nguồn triển khai: [ARCHITECTURE_AND_OUTPUT_SPEC.md](../pfedes/ARCHITECTURE_AND_OUTPUT_SPEC.md).

### 6.1. Phương pháp trong bài báo gốc

pFedES giải quyết **model-heterogeneous personalized FL (MHPFL)**. Mỗi client `k` được phép giữ classifier riêng `F_k(w_k)` có kiến trúc khác client khác; thay vì aggregate các classifier không tương thích, hệ thống chèn một feature enhancer/proxy đồng nhất và nhỏ `G(\theta)` trước chúng. Proxy nhận và trả tensor cùng kích thước, nên enhanced sample `\hat{x}=G(\theta;x)` vẫn có thể đi vào bất kỳ local model nào.

**Hai tầng tri thức.** Local model giữ tri thức đặc thù của dữ liệu/kiến trúc client và tồn tại xuyên suốt các round. Global proxy là kênh tri thức chung: nó học thông qua từng local model rồi được FedAvg ở server. Trong thiết kế ảnh gốc, proxy chỉ gồm hai convolution same-padding `3→8→3`, nhỏ hơn rất nhiều local CNN và không phải classifier độc lập.

**Pha 1 — cập nhật personalized model.** Client nhận global proxy của round hiện tại và đóng băng proxy. Với cùng minibatch, local model dự đoán cả ảnh gốc và ảnh đã enhance:

$$y_1=F_k(w_k;G(\theta;x)),\qquad y_2=F_k(w_k;x),$$

$$\ell_w=\mu\,\ell(y_1,y)+(1-\mu)\,\ell(y_2,y),\qquad 0<\mu\leq 0,5.$$

Nhánh dữ liệu gốc giữ quá trình học bám nhiệm vụ cá nhân; nhánh enhanced đưa tri thức được proxy tích lũy từ các client khác vào classifier. Giới hạn `mu≤0,5` khiến dữ liệu gốc vẫn là tín hiệu chủ đạo.

**Pha 2 — cập nhật proxy.** Client đóng băng local model vừa cập nhật, sau đó tối ưu proxy qua local model:

$$\hat{y}=F_k(w_k;G(\theta_k;x)),\qquad \ell_\theta=\ell(\hat{y},y).$$

“Đóng băng” local model nghĩa là không cập nhật `w_k`, nhưng gradient vẫn đi xuyên qua `F_k` tới `G`. Nhờ đó proxy học cách biến đổi đầu vào sao cho classifier riêng của client dự đoán tốt hơn. Paper tách hai pha thay vì cập nhật đồng thời để tri thức global không bị local optimization lấn át ngay từ minibatch đầu và để giảm áp lực bộ nhớ.

**Trao đổi server–client:**

1. Server chọn client và phát global proxy `\theta^t`; local model không bị thay thế.
2. Client lần lượt chạy pha cập nhật `w_k` rồi pha cập nhật `\theta_k`.
3. Client chỉ upload `\theta_k` cùng số mẫu; không upload `w_k`.
4. Server tính

   $$\theta^{t+1}=\sum_{k\in S_t}\frac{n_k}{\sum_{j\in S_t}n_j}\theta_k^{t+1}.$$

5. Khi huấn luyện kết thúc, client dùng **chỉ local personalized model** để inference; proxy là công cụ chuyển giao tri thức trong training.

Thiết kế này hỗ trợ model khác kiến trúc, che cấu trúc/weights của local classifier khỏi server và làm payload phụ thuộc kích thước proxy thay vì local model. Dưới các giả định trơn, gradient không chệch và phương sai bị chặn được nêu trong paper, phân tích lý thuyết cho tốc độ hội tụ non-convex `O(1/T)`. Tuy nhiên, đây là privacy theo ranh giới kiến trúc, không phải bảo đảm DP; proxy update vẫn có thể mang thông tin. Proxy cũng phải bảo toàn shape đầu vào và đủ năng lực để chuyển tri thức hữu ích giữa những local model rất khác nhau.

**Thiết lập gốc.** Paper dùng MNIST, CIFAR-10, CIFAR-100; phân hoạch non-IID 2/10 lớp hoặc 10/100 lớp mỗi client; CNN-1 đến CNN-5 trong cả trường hợp homogeneous và heterogeneous; 10/50/100 client với participation 100%/20%/10%; SGD `0,01`, batch 64–512, 1 hoặc 10 local epoch và 100 hoặc 500 communication round. Paper báo cáo mức tăng accuracy tối đa 1,29 điểm phần trăm, đồng thời giảm tới 99,6% communication và 82,9% computation so với FedGH tại mục tiêu được khảo sát.

### 6.2. Những thay đổi khi build lại

- Đổi ảnh thành 25 feature CICIoT2023, 34 lớp; local model là GRU/Transformer.
- Proxy ảnh `Conv 3→8→3` được đổi thành proxy 1D `Conv 1→8→1`, kernel 3, padding same, giữ input/output `[B,25]`; tổng cộng **57 tham số**.
- Ba kịch bản: 10 GRU; 10 Transformer; client 1–5 GRU và 6–10 Transformer.
- Dùng đủ 10 client mỗi round, 10 round, batch 1024, SGD 0,01, không scheduler; local/proxy mỗi pha một epoch; `mu=0,1` cho enhanced-data loss.
- Không server pretrain, KD, mutual distillation hoặc FedProx.
- Chạy client-parallel hai T4. Checkpoint chọn theo mean personalized validation macro-F1.
- Mỗi personalized model đánh giá toàn bộ global test. Metric chính là unweighted mean của 10 model; confusion matrix/classification report cộng 10 lượt dự đoán nên support gấp 10 lần global test.

### 6.3. So sánh ba kịch bản

| Kịch bản | Best round | Mean accuracy | Mean macro-F1 | Mean weighted-F1 | Pooled macro-F1 | Binary attack F1 | Runtime | Communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 GRU | 10 | 29,4082% | 8,4737% | 19,4591% | 15,0637% | 98,8039% | **43,8 phút** | 0,04349 MiB |
| **10 Transformer** | **10** | **32,3831%** | **11,9493%** | **22,5547%** | **18,8191%** | **98,8690%** | 129,3 phút | 0,04349 MiB |
| 5 GRU + 5 Transformer | 10 | 31,6232% | 10,0468% | 21,4700% | 16,9671% | 98,8624% | 135,9 phút | 0,04349 MiB |

Transformer tốt nhất trên toàn bộ metric đa lớp nhưng tốn runtime gần ba lần GRU. Communication giống hệt vì cả ba chỉ trao đổi proxy 57 tham số: tổng **45.600 byte** cho 10 round.

Best checkpoint của cả ba là round 10, nên proxy/local iterative training vẫn đang cải thiện theo validation criterion ở biên ngân sách hiện tại. Tuy nhiên, độ lệch client rất lớn:

| Kịch bản | Min–max accuracy | Min–max macro-F1 | Độ lệch chuẩn macro-F1 |
|---|---:|---:|---:|
| 10 GRU | 0,2087–57,8400% | 0,0123–21,6502% | 6,8174% |
| 10 Transformer | 0,2099–59,1546% | 0,0148–26,3818% | 8,8931% |
| Mixed | 0,2087–59,1627% | 0,0123–26,3734% | 7,8626% |

Một số personalized model gần như dự đoán một lớp khi đánh giá trên global test. Đây là dấu hiệu local non-IID personalization quá mạnh so với khả năng chuyển global knowledge của proxy 57 tham số trong 10 round. Dù vậy, không thể kết luận proxy quá nhỏ là nguyên nhân duy nhất nếu chưa chạy ablation theo số round, `mu`, kích thước proxy và standalone baseline.

### 6.4. Toàn bộ ảnh — 10 GRU

**Phân phối lớp**

![pFedES GRU class distribution](images/pfedes/01_10_clients_gru/class_distribution.png)

**Accuracy và F1 theo round**

![pFedES GRU accuracy F1](images/pfedes/01_10_clients_gru/accuracy_f1_curves.png)

**Loss**

![pFedES GRU loss](images/pfedes/01_10_clients_gru/loss_curves.png)

**Confusion matrix**

![pFedES GRU confusion matrix](images/pfedes/01_10_clients_gru/confusion_matrix.png)

**Per-class F1**

![pFedES GRU per class F1](images/pfedes/01_10_clients_gru/per_class_f1.png)

**Runtime theo round**

![pFedES GRU runtime](images/pfedes/01_10_clients_gru/runtime_per_round.png)

**Communication tích lũy**

![pFedES GRU communication](images/pfedes/01_10_clients_gru/communication_cumulative.png)

### 6.5. Toàn bộ ảnh — 10 Transformer

**Phân phối lớp**

![pFedES Transformer class distribution](images/pfedes/02_10_clients_transformer/class_distribution.png)

**Accuracy và F1 theo round**

![pFedES Transformer accuracy F1](images/pfedes/02_10_clients_transformer/accuracy_f1_curves.png)

**Loss**

![pFedES Transformer loss](images/pfedes/02_10_clients_transformer/loss_curves.png)

**Confusion matrix**

![pFedES Transformer confusion matrix](images/pfedes/02_10_clients_transformer/confusion_matrix.png)

**Per-class F1**

![pFedES Transformer per class F1](images/pfedes/02_10_clients_transformer/per_class_f1.png)

**Runtime theo round**

![pFedES Transformer runtime](images/pfedes/02_10_clients_transformer/runtime_per_round.png)

**Communication tích lũy**

![pFedES Transformer communication](images/pfedes/02_10_clients_transformer/communication_cumulative.png)

### 6.6. Toàn bộ ảnh — 5 GRU + 5 Transformer

**Phân phối lớp**

![pFedES mixed class distribution](images/pfedes/03_5_gru_5_transformer/class_distribution.png)

**Accuracy và F1 theo round**

![pFedES mixed accuracy F1](images/pfedes/03_5_gru_5_transformer/accuracy_f1_curves.png)

**Loss**

![pFedES mixed loss](images/pfedes/03_5_gru_5_transformer/loss_curves.png)

**Confusion matrix**

![pFedES mixed confusion matrix](images/pfedes/03_5_gru_5_transformer/confusion_matrix.png)

**Per-class F1**

![pFedES mixed per class F1](images/pfedes/03_5_gru_5_transformer/per_class_f1.png)

**Runtime theo round**

![pFedES mixed runtime](images/pfedes/03_5_gru_5_transformer/runtime_per_round.png)

**Communication tích lũy**

![pFedES mixed communication](images/pfedes/03_5_gru_5_transformer/communication_cumulative.png)

## 7. Phương pháp 5 — Lightweight FL với adaptive mutual distillation và proxy model

Nguồn triển khai: [ARCHITECTURE_AND_OUTPUT_SPEC.md](../proxymodel/ARCHITECTURE_AND_OUTPUT_SPEC.md) và [CONTEXT.md](../proxymodel/CONTEXT.md).

### 7.1. Phương pháp trong bài báo gốc

Bài báo xây dựng pipeline nhẹ cho **Non-Intrusive Load Monitoring (NILM)**: từ tổng công suất của một hộ gia đình, model hồi quy công suất từng thiết bị. Hai khó khăn được xử lý đồng thời là phần cứng client khác nhau và kiến trúc cá nhân hóa không thể FedAvg trực tiếp. Pipeline gồm (1) tìm local architecture phù hợp phần cứng bằng memory-efficient NAS và (2) liên kết các architecture đó bằng adaptive federated mutual learning qua một proxy đồng nhất.

**Memory-efficient NAS (MNAS).** Search space là một DAG 5 node; mỗi cạnh có 8 lựa chọn `{conv7, conv5, conv3, conv1, max pool, avg pool, skip, identity}` và cuối quá trình giữ hai path mạnh nhất tại mỗi node. Paper giảm bộ nhớ tìm kiếm bằng hai kỹ thuật:

- Các convolution kernel lồng nhau dùng chung tham số: kernel nhỏ là phần con của kernel lớn, nên không cần lưu một tensor độc lập cho mỗi operation.
- Single-path binarization chỉ kích hoạt một operation/path ở mỗi bước. Gumbel–Softmax tạo xấp xỉ khả vi để vẫn cập nhật architecture parameters.

Ngoài validation loss, search objective có hardware-aware latency penalty dạng:

$$\min_\alpha\;L_{val}(w,\alpha)+\left(\frac{T(\alpha)}{T_0}\right)^\lambda,
\qquad \text{s.t. } w=\arg\min_w L_{train}(w,\alpha),$$

với `T(alpha)` là tổng latency operation có trọng số, `T0` là độ dài sliding window và `lambda>1` là hệ số phạt. Vì vậy hai client có dữ liệu tương tự nhưng giới hạn phần cứng khác nhau vẫn có thể nhận personalized architecture khác nhau.

**Adaptive federated mutual learning.** Sau NAS, client có personalized model `s_k` khác kiến trúc và một proxy `r_k` cùng cấu trúc trên mọi client. Cả hai nhận cùng minibatch, có label loss riêng `\ell_s`, `\ell_r` và một discrepancy giữa hai đầu ra `\ell(y_s,y_r)`. Loss được viết:

$$L_s=\ell_s+\ell_d,\qquad L_r=\ell_r+\ell_d,$$

$$\ell_d=\frac{\ell(y_s,y_r)}{\ell_s+\ell_r}.$$

Đây là học hai chiều: personalized model truyền local knowledge cho proxy, còn proxy—sau khi đã aggregate tri thức từ các client—dạy ngược personalized model. Mẫu số làm hệ số thích nghi: ở đầu training, khi hai label loss còn lớn và dự đoán chưa đáng tin, distillation bị giảm; khi hai model học tốt hơn, ảnh hưởng tương đối của việc khớp đầu ra tăng lên. Cần phân biệt cơ chế này với teacher–student một chiều có teacher cố định: ở đây cả hai model đều nhận gradient và cùng thay đổi.

**Một chu kỳ federated:**

1. Server gửi global proxy cho client; personalized architecture/weights vẫn nằm local.
2. Client jointly train personalized model và proxy bằng label supervision cộng adaptive mutual loss.
3. Client upload chỉ proxy; server lấy trung bình `1/K` các proxy cùng kiến trúc để tạo proxy round mới.
4. Server phân phối proxy mới và lặp lại. Sau round cuối, personalized model được fine-tune thêm trên dữ liệu local rồi dùng inference.

Proxy gốc gồm ba convolution kernel 5 và một dense layer; fixed centralized/federated baseline lớn hơn gồm năm convolution kernel 7 và dense. Payload vì thế độc lập với kích thước personalized model, đồng thời NAS cho phép tối ưu latency riêng cho thiết bị. Raw data và personalized weights không rời client, nhưng paper giả định server/network đáng tin cậy và không đưa ra phòng vệ với client/server độc hại. Tác giả cũng ghi nhận NAS vẫn phát sinh compute phụ và communication còn có thể giảm thêm bằng các kỹ thuật như quantization.

**Thiết lập gốc.** Paper dùng REFIT (20 hộ tại Anh, chu kỳ 8 giây) và REDD (6 hộ tại Mỹ, chu kỳ 3 giây), đánh giá hồi quy bằng MAE/SAE. Các benchmark bao gồm centralized learning, local learning, NAS, MNAS, vanilla federated learning và pipeline đề xuất; do đó cả độ chính xác, kích thước model, latency và chi phí communication đều là mục tiêu đánh giá.

### 7.2. Những thay đổi khi build lại

- Đổi từ hồi quy NILM sang phân loại IDS 34 lớp trên CICIoT2023; MAE/SAE đổi thành accuracy, macro/weighted metrics, FPR/FNR.
- **Không chạy lại MNAS.** Personalized architecture được cố định thành GRU hoặc Transformer; vì vậy run hiện tại chỉ tái hiện phần adaptive federated mutual learning, không phải toàn bộ pipeline search của paper.
- Proxy đổi thành CNN-1D ba block 35.874 tham số; personalized/proxy label loss dùng cross-entropy.
- Distillation discrepancy dùng MSE giữa hai softmax. Mẫu số `CE_s+CE_r` được detach và cộng `1e-8`; một backward chung bảo đảm mỗi model nhận label gradient riêng và một lần distillation gradient.
- Original paper dùng average `1/K`; bản dựng dùng sample-weighted FedAvg theo local-train rows để xử lý client-size imbalance.
- Ba kịch bản: 10 GRU; 10 Transformer; client 1–5 GRU và 6–10 Transformer.
- 10 round × 1 local epoch, batch 1024, Adam 0,001; all-client participation; client-parallel hai T4; sau best checkpoint, personalized model fine-tune thêm một epoch chỉ bằng cross-entropy.
- Feature selection MI/NAS của paper không chạy; input dùng 25 feature XGBoost upstream.

### 7.3. Global proxy

| Kịch bản personalized | Best round | Proxy accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 | Binary attack F1 | Runtime | Communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 GRU | 8 | 61,8285% | 38,8190% | 41,4994% | 35,6114% | 52,3778% | 99,4250% | **31,7 phút** | 27,7161 MiB |
| 10 Transformer | 10 | 62,1071% | **39,1184%** | **41,5992%** | **35,6648%** | 53,1571% | 99,4580% | 65,0 phút | 27,7161 MiB |
| Mixed | 10 | **62,1917%** | 38,6356% | 41,4423% | 35,3779% | **53,5797%** | **99,4625%** | 56,7 phút | 27,7161 MiB |

Kết quả global proxy gần như không đổi giữa ba kịch bản: biên độ accuracy chỉ 0,3632 điểm phần trăm và macro-F1 chỉ 0,2869 điểm. Điều này hợp lý với thiết kế: cùng một proxy CNN-1D, cùng dữ liệu và FedAvg; kiến trúc personalized chỉ tác động gián tiếp qua distillation. Transformer cho macro-F1 cao nhất, mixed cho accuracy/weighted-F1 cao nhất.

Trong các round cuối, adaptive distillation loss chỉ khoảng `0,0009–0,0010`, trong khi personalized label loss xấp xỉ `0,10`. Sự chênh lệch quy mô cho thấy tín hiệu mutual transfer ở cuối quá trình khá nhỏ so với label supervision. Đây là quan sát định lượng, không đủ để kết luận nên tăng trọng số distillation nếu chưa làm sensitivity test.

Ở per-class output của global proxy, 14/34 lớp có F1 bằng 0, nhưng 14–15 lớp có F1 ≥ 0,5. Mô hình rất mạnh ở một số lớp DDoS lớn (`DDOS-RSTFINFLOOD`, `DDOS-PSHACK_FLOOD`, `DDOS-ICMP_FLOOD`) và bỏ sót nhiều lớp hiếm, tạo ra macro-F1 trung bình dù binary attack F1 rất cao.

### 7.4. Personalized model

| Kịch bản | Mean accuracy | Min–max accuracy | Mean macro-F1 | Min–max macro-F1 | Độ lệch chuẩn macro-F1 |
|---|---:|---:|---:|---:|---:|
| 10 GRU | 34,7448% | 1,4983–60,6892% | 16,5224% | 0,2507–34,9635% | 10,6070% |
| **10 Transformer** | **35,8243%** | **8,3889–61,8225%** | **17,2663%** | **2,0736–36,5897%** | 10,7436% |
| Mixed | 35,1813% | 2,5340–61,6891% | 16,8358% | 0,8637–36,7743% | 10,6422% |

Transformer cải thiện client yếu nhất so với GRU/mixed, nhưng độ phân tán vẫn lớn. Global proxy tốt hơn mean personalized model khoảng 26–27 điểm phần trăm accuracy. Điều này cho thấy trong cấu hình hiện tại, phần global knowledge trong proxy tổng quát ra global test tốt hơn các personalized state đã học mạnh theo local non-IID distribution.

### 7.5. Toàn bộ ảnh — 10 GRU

**Phân phối lớp**

![Adaptive mutual GRU class distribution](images/proxymodel/scenario_1_10_clients_gru/class_distribution.png)

**Accuracy và F1 theo round**

![Adaptive mutual GRU accuracy F1](images/proxymodel/scenario_1_10_clients_gru/accuracy_f1_curves.png)

**Loss**

![Adaptive mutual GRU loss](images/proxymodel/scenario_1_10_clients_gru/loss_curves.png)

**Confusion matrix**

![Adaptive mutual GRU confusion matrix](images/proxymodel/scenario_1_10_clients_gru/confusion_matrix.png)

**Per-class F1**

![Adaptive mutual GRU per class F1](images/proxymodel/scenario_1_10_clients_gru/per_class_f1.png)

**Runtime theo round**

![Adaptive mutual GRU runtime](images/proxymodel/scenario_1_10_clients_gru/runtime_per_round.png)

**Communication tích lũy**

![Adaptive mutual GRU communication](images/proxymodel/scenario_1_10_clients_gru/communication_cumulative.png)

### 7.6. Toàn bộ ảnh — 10 Transformer

**Phân phối lớp**

![Adaptive mutual Transformer class distribution](images/proxymodel/scenario_2_10_clients_transformer/class_distribution.png)

**Accuracy và F1 theo round**

![Adaptive mutual Transformer accuracy F1](images/proxymodel/scenario_2_10_clients_transformer/accuracy_f1_curves.png)

**Loss**

![Adaptive mutual Transformer loss](images/proxymodel/scenario_2_10_clients_transformer/loss_curves.png)

**Confusion matrix**

![Adaptive mutual Transformer confusion matrix](images/proxymodel/scenario_2_10_clients_transformer/confusion_matrix.png)

**Per-class F1**

![Adaptive mutual Transformer per class F1](images/proxymodel/scenario_2_10_clients_transformer/per_class_f1.png)

**Runtime theo round**

![Adaptive mutual Transformer runtime](images/proxymodel/scenario_2_10_clients_transformer/runtime_per_round.png)

**Communication tích lũy**

![Adaptive mutual Transformer communication](images/proxymodel/scenario_2_10_clients_transformer/communication_cumulative.png)

### 7.7. Toàn bộ ảnh — 5 GRU + 5 Transformer

**Phân phối lớp**

![Adaptive mutual mixed class distribution](images/proxymodel/scenario_3_5_gru_5_transformer/class_distribution.png)

**Accuracy và F1 theo round**

![Adaptive mutual mixed accuracy F1](images/proxymodel/scenario_3_5_gru_5_transformer/accuracy_f1_curves.png)

**Loss**

![Adaptive mutual mixed loss](images/proxymodel/scenario_3_5_gru_5_transformer/loss_curves.png)

**Confusion matrix**

![Adaptive mutual mixed confusion matrix](images/proxymodel/scenario_3_5_gru_5_transformer/confusion_matrix.png)

**Per-class F1**

![Adaptive mutual mixed per class F1](images/proxymodel/scenario_3_5_gru_5_transformer/per_class_f1.png)

**Runtime theo round**

![Adaptive mutual mixed runtime](images/proxymodel/scenario_3_5_gru_5_transformer/runtime_per_round.png)

**Communication tích lũy**

![Adaptive mutual mixed communication](images/proxymodel/scenario_3_5_gru_5_transformer/communication_cumulative.png)

## 8. So sánh liên phương pháp

### 8.1. Global/global-proxy metrics

Bảng này chỉ gồm các output có một model dùng chung. “Adaptive proxy” là CNN-1D proxy, không phải GRU/Transformer personalized model.

| Phương pháp | Bối cảnh kiến trúc | Đối tượng | Accuracy | Macro-F1 | Weighted-F1 | Best round |
|---|---|---|---:|---:|---:|---:|
| FD-IDS | 10 CNN-1D | Global CNN-1D | 59,1893% | 29,9830% | 50,9519% | 10 |
| PerFed-SKD | 10 CNN-1D | Global CNN-1D | **65,4886%** | **32,7390%** | **58,1448%** | 10 |
| FD-IDS | 10 GRU | Global GRU | 60,1705% | 33,6860% | 52,0735% | 10 |
| **PerFed-SKD** | **10 GRU** | **Global GRU** | **72,5690%** | 36,3608% | **68,6109%** | 1 |
| Adaptive proxy | Personalized 10 GRU | Global proxy CNN-1D | 61,8285% | **35,6114%** | 52,3778% | 8 |
| **FD-IDS** | **10 Transformer** | **Global Transformer** | 64,2810% | **40,4408%** | 56,7152% | 10 |
| PerFed-SKD | 10 Transformer | Global Transformer | **69,9675%** | 40,0462% | **63,8592%** | 3 |
| Adaptive proxy | Personalized 10 Transformer | Global proxy CNN-1D | 62,1071% | 35,6648% | 53,1571% | 10 |
| Adaptive proxy | Personalized mixed | Global proxy CNN-1D | 62,1917% | 35,3779% | 53,5797% | 10 |

Các nhận định có thể bảo vệ từ bảng:

- Trong kịch bản CNN-1D đồng nhất, PerFed-SKD cao hơn FD-IDS **6,2993 điểm accuracy** và **2,7560 điểm macro-F1**.
- Trong kịch bản GRU đồng nhất, PerFed-SKD cao hơn FD-IDS **12,3985 điểm accuracy** và **2,6748 điểm macro-F1**. Tuy nhiên, PerFed-SKD có server pretrain trên global train và dùng checkpoint round 1, nên không thể quy toàn bộ chênh lệch cho SKD/device selection.
- Trong kịch bản Transformer đồng nhất, PerFed-SKD cao hơn FD-IDS **5,6865 điểm accuracy**, nhưng FD-IDS cao hơn **0,3946 điểm macro-F1**. Nếu mục tiêu là bao phủ đều 34 loại tấn công, FD-IDS Transformer nhỉnh hơn; nếu ưu tiên tổng số dự đoán đúng, PerFed-SKD Transformer nhỉnh hơn.
- Global proxy của adaptive mutual learning ít nhạy với kiến trúc personalized; cả ba run gần như tạo cùng một operating point.

### 8.2. Personalized metrics

| Phương pháp | Kịch bản | Mean accuracy | Mean macro-F1 | Min macro-F1 | Max macro-F1 |
|---|---|---:|---:|---:|---:|
| PerFed-SKD | 10 CNN-1D | 46,4799% | 19,7153% | 5,2549% | 32,7883% |
| **PerFed-SKD** | **10 GRU** | **66,9950%** | **30,9894%** | **28,6051%** | 35,2363% |
| PerFed-SKD | 10 Transformer | 57,9521% | 30,2976% | 4,5349% | **39,7421%** |
| FedCAPS | 4G+3T+3C | 25,7233% | 8,3564% | 0,0123% | 22,1613% |
| FedCAPS | 6C+3G+1T | 27,9071% | 7,9262% | 1,2418% | 20,2800% |
| FedCAPS | 5T+5G | 26,6656% | 8,2356% | 0,0123% | 16,4236% |
| pFedES | 10 GRU | 29,4082% | 8,4737% | 0,0123% | 21,6502% |
| pFedES | 10 Transformer | 32,3831% | 11,9493% | 0,0148% | 26,3818% |
| pFedES | 5G+5T | 31,6232% | 10,0468% | 0,0123% | 26,3734% |
| Adaptive mutual | 10 GRU | 34,7448% | 16,5224% | 0,2507% | 34,9635% |
| Adaptive mutual | 10 Transformer | 35,8243% | 17,2663% | 2,0736% | 36,5897% |
| Adaptive mutual | 5G+5T | 35,1813% | 16,8358% | 0,8637% | 36,7743% |

PerFed-SKD dẫn đầu personalized metrics, đặc biệt là 10 GRU. Adaptive mutual distillation đứng thứ hai ở hai kịch bản homogeneous. pFedES và FedCAPS hy sinh chất lượng test để đạt communication rất thấp hoặc feature subset nhỏ trong ngân sách hiện tại.

Không nên dùng hàng mixed FedCAPS để so trực tiếp với mixed pFedES/adaptive: FedCAPS gán **client 1–5 Transformer, 6–10 GRU**, trong khi hai phương pháp còn lại gán **client 1–5 GRU, 6–10 Transformer**. Do client 8 có hơn 12,7 triệu mẫu và mỗi client có label distribution khác nhau, việc đảo assignment làm thay đổi cả workload lẫn weighted performance.

### 8.3. Communication

| Phương pháp | Communication/run | Nội dung được tính | Nhận xét |
|---|---:|---|---|
| **pFedES** | **0,04349 MiB** | Proxy extractor download + upload qua 10 round | Thấp nhất; proxy chỉ 57 tham số |
| FedCAPS | 0,13766–0,14338 MiB | Feature-selection record và PPO feedback | Không truyền classifier; khác bản chất model FL |
| PerFed-SKD | 9,9778–21,6705 MiB | Full model của selected clients | Giảm còn 36–46% số lượt full participation |
| FD-IDS | 27,7161–54,1763 MiB | Full model download/upload của 10 client × 10 round | Tăng theo kích thước CNN/GRU/Transformer |
| Adaptive mutual proxy | 27,7161 MiB | Proxy CNN-1D download/upload của đủ 10 client | Personalized model bị loại khỏi logical cost |

Theo logical accounting hiện tại, pFedES dùng ít hơn khoảng **637 lần** so với adaptive proxy và ít hơn khoảng **702 lần** so với FD-IDS GRU. Các tỷ lệ này không bao gồm protocol, serialization, retry, checkpoint hoặc inter-process payload. FedCAPS có boundary khác hẳn vì truyền record/feature ID thay vì model state.

### 8.4. Runtime

| Phương pháp | Nhanh nhất | Chậm nhất | Thành phần chi phối |
|---|---:|---:|---|
| FD-IDS | 95,0 phút (GRU) | 178,2 phút (Transformer) | DDP full participation trên 36 triệu mẫu × 10 round |
| PerFed-SKD | 28,8 phút (GRU) | 80,2 phút (Transformer) | Server pretrain + client-parallel; selected upload không đồng nghĩa selected training |
| FedCAPS | 43,2 phút (S2) | 56,8 phút (S1) | MARLFS, encoder–decoder, PPO và final training |
| pFedES | 43,8 phút (GRU) | 135,9 phút (mixed) | Hai pha local/proxy trên mỗi client mỗi round |
| Adaptive mutual proxy | 31,7 phút (GRU) | 65,0 phút (Transformer) | Hai model học đồng thời + final fine-tune |

Runtime không thể dùng làm kết luận thuật toán tuyệt đối: FD-IDS dùng DDP và global batch chia đôi, các phương pháp còn lại dùng client-parallel; FedCAPS còn thực hiện feature search; số phase/epoch khác nhau. Bảng chỉ hữu ích để lập ngân sách chạy lại trong đúng môi trường Kaggle T4×2 này.

## 9. Phân tích tổng hợp

### 9.1. Tác động của mất cân bằng lớp

Tập test có hơn 1,37 triệu mẫu `DDOS-ICMP_FLOOD` nhưng chỉ 239 mẫu `UPLOADING_ATTACK` và 432 mẫu `RECON-PINGSWEEP`. Vì không dùng class weighting/resampling:

- Accuracy và weighted-F1 bị chi phối bởi các lớp lớn.
- Binary attack F1 cao vì 33/34 nhãn được gộp thành “attack”; một model nhận biết attack nói chung vẫn có thể không phân biệt đúng loại tấn công.
- Macro-F1 mới phản ánh đều 34 lớp. Kết quả tốt nhất chỉ 40,4408%, và ngay cả model tốt vẫn có 11 lớp F1 bằng 0.
- Không nên tuyên bố hệ thống IDS “99% chính xác” từ binary attack F1. Kết luận đúng là nhận biết attack/benign tốt, còn phân loại chi tiết 34 lớp vẫn hạn chế.

### 9.2. Global generalization và personalization

PerFed-SKD 10 GRU là trường hợp personalized thành công nhất: mean gần global result và độ lệch client nhỏ. Ngược lại, pFedES/adaptive/FedCAPS có một số client gần như sụp đổ khi đánh giá trên global test.

Cần phân biệt mục tiêu: personalized FL thường đánh giá mỗi model trên test data cùng local distribution. Ở đây mọi personalized model được đánh giá trên toàn bộ global test. Phép đo này rất tốt để kiểm tra **khả năng tổng quát toàn cục**, nhưng có thể đánh giá thấp lợi ích cá nhân hóa cho client. Muốn kết luận đầy đủ, cần bổ sung local test split độc lập theo từng client.

### 9.3. Dung lượng kênh chia sẻ

- Full-model sharing (FD-IDS, PerFed-SKD) tạo global model mạnh hơn nhưng tốn communication.
- Proxy CNN-1D 35.874 tham số của adaptive mutual learning mang global knowledge tốt hơn proxy 57 tham số của pFedES trong ngân sách 10 round, nhưng tốn khoảng 637 lần communication.
- FedCAPS truyền rất ít nhưng chỉ chia sẻ feature-selection knowledge; classifier vẫn hoàn toàn local. Vì vậy nó không nhận cùng loại regularization/global representation như model aggregation.
- Communication–quality trade-off hiện tạo một Pareto rõ: pFedES/FedCAPS ở đầu cực tiết kiệm, PerFed-SKD ở vùng giữa, FD-IDS/adaptive proxy ở đầu cực nhiều byte hơn.

### 9.4. Độ hội tụ

- FD-IDS: cả ba best round = 10.
- PerFed-SKD: CNN-1D best = 10; GRU = 1; Transformer = 3.
- pFedES: cả ba best = 10.
- Adaptive mutual: GRU = 8; Transformer/mixed = 10.
- FedCAPS: encoder reconstruction hội tụ sớm ở epoch 13, nhưng search/final classifier là pipeline khác nên không có `best_round` tương đương.

Các run best ở round cuối chưa đủ chứng cứ hội tụ. Các run PerFed-SKD best sớm có dấu hiệu global validation macro-F1 plateau/giảm sau khi device selection tiếp tục; cần xem đây là đặc tính của checkpoint selection, không phải lỗi chạy.

### 9.5. Mức độ tái hiện bài báo gốc

| Phương pháp | Thành phần tái hiện tốt | Thành phần đã lược bỏ/thay đổi lớn |
|---|---|---|
| FD-IDS | FedProx + round-wise KD + weighted FedAvg | MI thay bằng XGBoost upstream; DNN/dataset/round budget đổi |
| PerFed-SKD | Historical teacher, selective communication, personalized state | Dataset/model/budget đổi; ngưỡng và aggregation phải giải quyết ambiguity; server pretrain dùng full global train |
| FedCAPS | MARLFS records, permutation-invariant encoder, PPO, sample-aware weighting | Random Forest/14 datasets/baselines bỏ; neural evaluator và giới hạn search pool thêm vào |
| pFedES | Proxy extractor sharing và two-phase iterative training | Image CNN đổi GRU/Transformer; round giảm 100/500 xuống 10; proxy cực nhỏ 1D |
| Adaptive mutual proxy | Proxy aggregation + bidirectional adaptive distillation + fine-tune | **Không chạy MNAS**; NILM regression đổi IDS classification; proxy và FedAvg weighting đổi |

Vì vậy, các kết quả là **tái hiện phương pháp cốt lõi trên một bài toán mới**, không phải reproduction định lượng để đối chiếu trực tiếp con số trong paper gốc.

## 10. Hạn chế và các lần chạy bổ sung nên thực hiện

### 10.1. Hạn chế hiện tại

1. Mỗi cấu hình chỉ có một run với seed 42; chưa có mean ± standard deviation hoặc kiểm định thống kê.
2. Training budget, optimizer, local split, execution topology và server-data assumption khác nhau giữa phương pháp.
3. Không có baseline chung như standalone local, centralized, FedAvg và FedProx chạy cùng kiến trúc/budget.
4. Mixed assignment không đồng nhất giữa FedCAPS và hai phương pháp proxy.
5. Personalized model chỉ được đánh giá trên global test, chưa có local test độc lập.
6. Không phương pháp nào xử lý imbalance; nhiều lớp hiếm có F1 bằng 0.
7. Communication là logical payload estimate, chưa phải network trace thực tế.

### 10.2. Thứ tự ưu tiên huấn luyện lại

Các output bắt buộc hiện **không thiếu**, nên không cần huấn luyện lại chỉ để hoàn thiện báo cáo. Nếu mục tiêu tiếp theo là một so sánh khoa học chặt chẽ, nên chạy theo thứ tự:

1. **Lặp 3–5 seed** cho mọi cấu hình, giữ nguyên data split hoặc ghi rõ split seed, báo cáo mean ± std.
2. **Kéo dài round với early stopping** cho FD-IDS, pFedES, PerFed-SKD CNN-1D và adaptive Transformer/mixed vì best checkpoint đang ở round cuối.
3. **Ablation PerFed-SKD không server pretrain** và/hoặc chỉ pretrain trên một public/server subset tách biệt, để đo đúng đóng góp SKD/device selection.
4. **Ablation FedCAPS**: all-25-feature baseline; chọn candidate theo weighted Micro-F1 thay vì reward; thử `lambda ∈ {0,1; 0,5; 0,9}`; tăng candidate evaluator epoch; giữ cùng final-training budget.
5. **Ablation pFedES**: 100/500 round gần paper gốc; proxy rộng hơn; sweep `mu`; thêm standalone local baseline để biết proxy có thực sự giúp từng client hay không.
6. **Đồng bộ mixed assignment** theo cùng client ID và kiến trúc cho mọi phương pháp.
7. **Thêm local test theo client** song song với global test, đồng thời giữ global test để đo generalization.
8. Tạo một nhánh thí nghiệm xử lý imbalance bằng class-weighted CE, focal loss hoặc balanced sampling; không trộn các kết quả này vào benchmark hiện tại nếu chưa chạy lại đủ năm phương pháp.

## 11. Kết luận

Trong ngân sách hiện tại, **PerFed-SKD 10 GRU** là lựa chọn tốt nhất nếu ưu tiên accuracy/global và personalized consistency, nhưng phải chấp nhận giả định server pretrain trên global train. **FD-IDS 10 Transformer** là lựa chọn tốt nhất nếu ưu tiên macro-F1 34 lớp và không dùng server train tập trung, với chi phí runtime/communication cao nhất. **Adaptive mutual proxy** tạo global proxy ổn định và nhanh hơn FD-IDS Transformer, nhưng personalized generalization còn yếu. **pFedES** là lựa chọn cực tiết kiệm communication, còn **FedCAPS** là lựa chọn phù hợp khi mục tiêu chính là chia sẻ tri thức chọn feature thay vì aggregate classifier; cả hai cần thêm round/ablation trước khi dùng cho IDS 34 lớp trong thực tế.

Kết quả quan trọng nhất không phải binary attack F1 gần 99%, mà là khoảng cách lớn giữa chỉ số này và macro-F1. Hệ thống hiện phân biệt attack/benign tốt hơn nhiều so với phân loại đúng từng loại tấn công hiếm. Mọi bước tối ưu tiếp theo nên lấy macro-F1, per-class recall và worst-client performance làm tiêu chí chính.
