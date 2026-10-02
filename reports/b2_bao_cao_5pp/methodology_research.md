# Nghiên cứu đối chiếu phương pháp gốc của năm bài báo

Tài liệu này chỉ mô tả **phương pháp và thí nghiệm trong bài báo gốc**; không sử dụng hoặc phân tích kết quả tái hiện trong workspace. Nguồn chuẩn được ưu tiên là PDF của nhà xuất bản/hội nghị. Bản Markdown chỉ được dùng để tra cứu văn bản; riêng bài FedCAPS hiện là arXiv v3, còn bản Markdown của bài NILM bị thiếu toàn bộ Sections I–III. Số trang dưới đây là **số trang của file PDF**, không phải số trang in trên tạp chí.

## 1. FD-IDS — Federated Learning với FedProx và Knowledge Distillation cho IDS non-IID

**Bài báo:** *FD-IDS: Federated Learning with Knowledge Distillation for Intrusion Detection in Non-IID IoT Environments*, Haonan Peng, Chunming Wu và Yanfeng Xiao, *Sensors* 2025. PDF là nguồn chuẩn về tên tác giả và nhan đề; bản Markdown ghi biến thể khác của tên tác giả. Nguồn: [PDF gốc, tr. 1](../fd_ids_noniid/sensors-25-04309.pdf#page=1), [bản Markdown](../fd_ids_noniid/sensors-25-04309.md).

### 1.1. Bài toán và mục tiêu

FD-IDS nhắm tới IDS đa lớp cho IoT trong bối cảnh dữ liệu lưu lượng của các thiết bị không độc lập, không cùng phân phối. Dữ liệu của từng client có thể chỉ chứa một phần loại tấn công và có tỷ lệ lớp khác nhau, vì vậy local update dễ lệch khỏi hướng tối ưu chung. Paper đặt đồng thời ba mục tiêu: không tập trung raw traffic; giảm số chiều/chi phí bằng chọn đặc trưng; và giảm *model drift* do non-IID bằng cách phối hợp ràng buộc FedProx với tri thức mềm từ global model. [PDF, tr. 6–9, §3.1–3.5](../fd_ids_noniid/sensors-25-04309.pdf#page=6), [Markdown, §3](../fd_ids_noniid/sensors-25-04309.md#3-methodology).

### 1.2. Giả định hệ thống

- Đây là horizontal FL: các client có cùng feature space và bài toán phân loại, nhưng giữ các mẫu khác nhau.
- Hệ thống gồm một central server và nhiều client IoT. Global và local model có cùng kiến trúc để server có thể lấy trung bình tham số.
- Paper giả định server và client đều đáng tin: không cố ý thay dữ liệu hoặc chèn cập nhật độc hại. Giả định này loại poisoning/Byzantine client khỏi phạm vi phương pháp.
- Huấn luyện đồng bộ: server chờ tất cả client hoàn thành local training trước khi cập nhật global model; trong thiết lập gốc cả chín client đều tham gia mỗi round.
- Raw samples ở lại client; đối tượng trao đổi là model parameters. Đây là privacy-by-locality của FL, không phải một cơ chế bảo mật mật mã hay differential privacy.

Nguồn: [PDF, tr. 9–10, §3.5.1](../fd_ids_noniid/sensors-25-04309.pdf#page=9), [PDF, tr. 13–14, §4.1](../fd_ids_noniid/sensors-25-04309.pdf#page=13).

### 1.3. Thành phần mô hình

1. **Tiền xử lý và chọn feature bằng Mutual Information (MI).** Paper bỏ các feature IP/port dễ làm mô hình phụ thuộc định danh, loại missing/infinite, one-hot encode biến phân loại và chuẩn hóa Z-score. Sau tiền xử lý, Edge-IIoT có 95 chiều và N-BaIoT có 115 chiều; MI xếp hạng quan hệ giữa từng feature và nhãn, rồi giữ lần lượt 25 và 71 feature. [PDF, tr. 7–8, §3.3](../fd_ids_noniid/sensors-25-04309.pdf#page=7).
2. **DNN classifier.** Mạng gồm input, năm hidden layer đối xứng `32 → 64 → 128 → 64 → 32`, ReLU ở hidden layers và softmax ở output; optimizer là Adam với learning rate 0,001. [PDF, tr. 8–9, §3.4](../fd_ids_noniid/sensors-25-04309.pdf#page=8).
3. **FedProx local regularization.** Proximal term giữ local weights gần global weights ở đầu round, nhằm giảm client drift.
4. **Global teacher / local student distillation.** Trong mỗi round, bản global model nhận từ server được giữ làm teacher; local model là student. Teacher và student chạy trên cùng local mini-batch, nhưng teacher không được cập nhật bởi local optimizer.
5. **Server-side weighted aggregation.** Sau local training, server lấy trung bình local weights theo số mẫu của client, cùng nguyên tắc sample-weighted FedAvg.

Nguồn cho ba thành phần FL: [PDF, tr. 9–12, §3.5 và Algorithm 1](../fd_ids_noniid/sensors-25-04309.pdf#page=9), [Markdown, §3.5](../fd_ids_noniid/sensors-25-04309.md#35-federated-learning-process-with-knowledge-distillation).

### 1.4. Thuật toán client–server theo từng bước

**Khởi tạo tại server**

1. Server khởi tạo global weights (w_G^0).
2. Ở round (t), server gửi (w_G^t) tới tất cả (K) client.

**Cập nhật tại client (k)**

3. Client đặt student (w_k \leftarrow w_G^t), đồng thời dùng chính bản (w_G^t) làm teacher cố định.
4. Với mỗi mini-batch local, client tính hard-label classification loss (L_{hard}).
5. Client lấy teacher logits (Z_t) và student logits (Z_s), làm mềm hai phân phối bằng temperature (T), rồi tính KL divergence (L_{soft}).
6. Client tính khoảng cách tham số giữa student hiện tại và global reference (L_{proximal}).
7. Client ghép ba thành phần thành một objective, gradient-descent/Adam trong (E) local epoch và trả (w_k^{t+1}) cho server.

**Tổng hợp tại server**

8. Server lấy weighted mean theo (n_k/n) để tạo (w_G^{t+1}).
9. Lặp lại đến (R) round và trả global model cuối.

Paper gọi việc distill ở mọi communication round là *round-wise KD*; phần thực nghiệm cũng đối chiếu với periodic KD và KD chỉ ở cuối, rồi chọn round-wise KD làm cấu hình chính. [PDF, tr. 10–12, §3.5.2 và Algorithm 1](../fd_ids_noniid/sensors-25-04309.pdf#page=10), [PDF, tr. 17–18, §4.3.2](../fd_ids_noniid/sensors-25-04309.pdf#page=17).

### 1.5. Hàm mục tiêu và công thức quan trọng

MI giữa feature và nhãn được paper viết dưới dạng entropy:

\[
I(V_x;V_y)=E(V_x)+E(V_y)-JE(V_x,V_y).
\]

Weighted aggregation:

\[
w_G^{t+1}=\sum_{k=1}^{K}\frac{n_k}{n}w_k^{t+1},\qquad n=\sum_{k=1}^{K}n_k.
\]

FedProx penalty:

\[
L_{proximal}=\frac{\mu}{2}\lVert w_k-w_G^t\rVert_2^2.
\]

Distillation loss với temperature scaling:

\[
L_{soft}=T^2 KL\!\left(softmax(Z_t/T)\parallel softmax(Z_s/T)\right).
\]

Objective local cuối cùng:

\[
L_{total}=\lambda L_{hard}+(1-\lambda)L_{soft}+\beta L_{proximal}.
\]

Paper đặt $\mu=0{,}01$, $\lambda=0{,}5$, $\beta=0{,}1$, $T=3$. Cần phân biệt $\mu$ nằm bên trong proximal term và $\beta$ là trọng số của toàn proximal term trong tổng loss. [PDF, tr. 7, 9–12, Eqs. (1)–(6)](../fd_ids_noniid/sensors-25-04309.pdf#page=7).

### 1.6. Privacy và communication boundary

- **Không rời client:** raw network traffic, local dataset.
- **Client nhận:** full global DNN weights.
- **Client gửi:** full updated local DNN weights.
- **Server thấy:** model updates và số mẫu dùng làm trọng số aggregation; server không nhận raw samples theo protocol mô tả.
- **Chưa được bảo vệ:** paper không dùng secure aggregation, encryption hoặc differential privacy; đồng thời giả định client/server trusted. Phần hạn chế của paper thừa nhận chưa xử lý poisoning và membership inference.

Nguồn: [PDF, tr. 9–10, §3.5.1](../fd_ids_noniid/sensors-25-04309.pdf#page=9), [PDF, tr. 27–28, §4.4](../fd_ids_noniid/sensors-25-04309.pdf#page=27).

### 1.7. Thiết lập thí nghiệm gốc

- **Dataset:** Edge-IIoT và N-BaIoT; train/test 80/20.
- **Client:** 9 client, synchronous full participation.
- **Non-IID:** chia train bằng Dirichlet với $\theta=1$ cho low non-IID và $\theta=0{,}1$ cho high non-IID.
- **Training:** 40 communication rounds, 2 local epochs/round, batch size 128, Adam, learning rate 0,001.
- **Model/loss:** DNN năm hidden layer; $\mu=0{,}01$, $\lambda=0{,}5$, $\beta=0{,}1$, $T=3$.
- **So sánh/ablation:** các khoảng KD, có/không KD, FedAvg/FedProx/SIM-FED, có/không MI, centralized learning và các IDS trước đó.
- **Metric:** accuracy, precision, recall, F1; paper còn báo parameter count, FLOPs, model size và communication volume.

Nguồn: [PDF, tr. 13–15, §4.1–4.2 và Tables 2–3](../fd_ids_noniid/sensors-25-04309.pdf#page=13), [PDF, tr. 16–26, §4.3](../fd_ids_noniid/sensors-25-04309.pdf#page=16).

### 1.8. Ưu điểm và giới hạn paper tự nêu

**Ưu điểm được paper khẳng định:** MI giảm chiều và chi phí; FedProx hạn chế local drift; global-teacher KD truyền thông tin phân phối lớp toàn cục để local update ít lệch hơn; DNN nhỏ khoảng 22.095 tham số/0,08 MB được paper đánh giá là phù hợp thiết bị biên; raw data không phải tập trung. [PDF, tr. 23–24, §4.3.6](../fd_ids_noniid/sensors-25-04309.pdf#page=23), [PDF, tr. 27–28, §4.4–5](../fd_ids_noniid/sensors-25-04309.pdf#page=27).

**Giới hạn được paper nêu rõ:** distillation làm tăng tính toán tại client; chưa xét concept drift; mới đánh giá trên hai dataset nên chưa bao phủ toàn bộ dị biệt IoT; giả định client an toàn và chưa xử lý poisoning/membership inference. Hướng tiếp theo gồm distillation nhẹ hơn, online/incremental/continual learning, thêm dataset, trust-aware/robust aggregation và differential privacy. [PDF, tr. 27–28, §4.4–5](../fd_ids_noniid/sensors-25-04309.pdf#page=27).

---

## 2. PerFed-SKD — Personalized FL bằng self-knowledge distillation và chọn thiết bị theo accuracy

**Bài báo:** *Personalized Federated Learning for Heterogeneous Edge Device: Self-Knowledge Distillation Approach*, Neha Singh, Jatin Rupchandani và Mainak Adhikari, *IEEE Transactions on Consumer Electronics* 70(1), 2024. Nguồn: [PDF gốc](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf), [bản Markdown](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md).

### 2.1. Bài toán và mục tiêu

PerFed-SKD xử lý hiện tượng global model có thể làm một client non-IID kém hơn chính personalized state trước đó. Paper muốn giữ “historical personalized knowledge” ở từng edge device, đồng thời giảm số lần truyền global/local model cho thiết bị hạn chế tài nguyên. Hai cơ chế cốt lõi là: dùng personalized model của round trước làm teacher cho model hiện tại; và chỉ cấp global model mới cho các client có local accuracy thấp hơn global accuracy threshold. [PDF, tr. 1–2, §I](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=1), [PDF, tr. 4–5, §IV](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=4).

### 2.2. Giả định và mô hình hệ thống

- Có (M) edge devices nối với một cloud server; client (m) giữ private dataset (D_m).
- Các client cùng tối ưu một model có thể aggregate trực tiếp bằng trọng số; “heterogeneous” trong bài chủ yếu chỉ dị biệt dữ liệu/tài nguyên, không mô tả model-architecture heterogeneity.
- Server có một **predefined dataset** để khởi tạo/train teacher/global dense model trước khi phân phối. Đây là giả định mạnh hơn FL thuần túy chỉ có dữ liệu tại client.
- Mỗi client lưu persistent personalized model (V_m) qua các round.
- Server biết global accuracy (A) và local accuracies (a_m) để chọn thiết bị; do đó metric/metadata của client cũng đi qua trust boundary.

Nguồn: [PDF, tr. 3–4, §III](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=3), [Markdown, §III](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md#iii-system-model-and-problem-formulation).

### 2.3. Thành phần mô hình

**Phía cloud:** Coordinator điều phối round; Teacher Model khởi tạo dense/global model; Aggregator hợp nhất local parameters; Device Selector so local accuracy với global accuracy; Communication Manager truyền request/model/metadata. **Phía client:** Controller nhận lệnh; System Monitor theo dõi tải và pin; Student Model Manager khởi tạo hoặc khôi phục local/personalized weights; Data Manager giữ local data; Training Optimizer cập nhật model. Paper mô tả các vai trò logic này nhưng không cung cấp chi tiết layer-by-layer của dense/student network. [PDF, tr. 3, §III-A, Fig. 1](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=3).

SKD trong bài là *self*-distillation theo thời gian: teacher (V_m) là bản personalized model đã lưu của chính client (m), còn student là local model đang train. Teacher không cần là một kiến trúc lớn khác student. [PDF, tr. 4–5, §IV](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=4).

### 2.4. Thuật toán client–server theo từng bước

1. Server train/khởi tạo (omega^0) bằng predefined dataset.
2. Ở round $t$, đặt threshold $\tau=A$, tức accuracy của global model hiện tại.
3. Server so $a_m$ của từng client với $\tau$. Client có $a_m<\tau$ được đưa vào selected set $S$.
4. Server chỉ gửi (omega^t) cho client trong (S). Client được chọn khởi tạo local model bằng global weights; client không được chọn tiếp tục từ weights đang có.
5. Mỗi client train song song trên (D_m), với local classification loss cộng divergence giữa prediction của historical teacher (V_m) và current student.
6. Sau local update, client lưu weights mới thành (V_m) để làm teacher ở round sau, đồng thời trả local weights và local accuracy.
7. Server nhận model của selected clients và aggregate thành (omega^{t+1}); global accuracy (A) được cập nhật từ local accuracies.
8. Lặp lại (T) round; đầu ra chính của thuật toán là tập personalized models (P_m), không chỉ một global model.

Nguồn: [PDF, tr. 4–5, §IV, Algorithms 1–2](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=4), [Markdown, Algorithms 1–2](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md#algorithm-1-perfed-skd-global-model-training).

### 2.5. Hàm mục tiêu và điểm cần giữ fidelity

Standard FL objective được paper viết:

\[
\min_\omega F(\omega)=\sum_{m=1}^{M}\frac{|D_m|}{|D|}F_m(\omega).
\]

Local SKD objective:

\[
\phi_m(\omega_m^t)=f_m(\omega_m^t)+\lambda\,L\!\left(x(V_m)\,\|\,x(\omega_m^t)\right),
\]

trong đó (f_m) là cross-entropy, (L) là divergence giữa prediction cũ và mới, và (lambda) điều khiển SKD. Local SGD update:

\[
\omega_m^t\leftarrow\omega_m^t-\eta\nabla\phi_m(\omega_m^t,V_m).
\]

Paper không định nghĩa rõ loại divergence, temperature hoặc giá trị (lambda) trong phần phương pháp/thí nghiệm. Ngoài ra, văn bản nói aggregate selected clients nhưng dòng 10 của Algorithm 1 ghi tổng trên (m\in M) rồi chia (|S|); đây là một bất nhất ký hiệu cần được nêu khi tái hiện, không nên tự xem pseudo-code là một weighted FedAvg đầy đủ. Phần System Model có nhắc proximal term cho standard FL, nhưng objective PerFed-SKD ở Eq. (2) chỉ có cross-entropy và SKD divergence. [PDF, tr. 3–5, Eqs. (1)–(3), Algorithms 1–2](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=3).

### 2.6. Privacy và communication boundary

- **Không rời client:** raw (D_m), persistent teacher/personalized state (V_m) (trừ bản weights được upload khi client được chọn).
- **Client nhận:** initial/global model; các selected clients tiếp tục nhận global update ở round kế tiếp.
- **Client gửi:** local model parameters và local accuracy khi tham gia selection/aggregation.
- **Server giữ:** initial teacher/global model, selected set, global/local accuracy metadata và aggregated model.
- **Communication saving:** đến từ việc không gửi global model và không nhận local update từ mọi client ở mọi round; paper không dùng nén, secure aggregation hoặc DP.

Nguồn: [PDF, tr. 3–5, §III-A và §IV](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=3).

### 2.7. Thiết lập thí nghiệm gốc

- **Dataset:** MNIST và EMNIST; non-IID bằng Dirichlet. Kết quả chính dùng (alpha=0{,}001), sensitivity dùng (alpha\in\{0{,}001,0{,}01,0{,}1\}).
- **Quy mô:** 20 đến 80 edge devices; participation ratios 20%, 60%, 100%.
- **Training:** 200 global iterations, 20 local epochs, local learning rate 0,01; toàn bộ quy trình lặp 10 lần rồi báo trung bình accuracy/loss.
- **Môi trường:** Dell Vostro, Intel Core i7-12700, Spyder/PyTorch.
- **Baselines:** FedAvg, FedProx, FedEnsemble, PerFedAvg.
- **Đánh giá:** personalized average test accuracy/loss, độ lệch chuẩn giữa client, communication/convergence curves, user training time và server aggregation time.

Lưu ý fidelity: phần mô tả EMNIST trong paper tự mâu thuẫn — Table II ghi 60.000 samples/26 classes, trong khi đoạn văn ghi 814.255 images/47 balanced classes. Không nên lấp khoảng trống này bằng suy đoán khi trình bày thí nghiệm gốc. [PDF, tr. 5–8, §V](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=5), [Markdown, §V](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.md#v-experimental-analysis).

### 2.8. Ưu điểm và giới hạn paper tự nêu

**Ưu điểm paper nêu:** historical personalized teacher làm giảm quên tri thức local và cân bằng generalization–personalization; device selection giảm trao đổi không cần thiết; phương pháp hướng tới thiết bị hạn chế tài nguyên; paper báo convergence nhanh hơn và personalized accuracy cao hơn các baseline. [PDF, tr. 6–8, §V-C và §VI](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=6).

**Giới hạn/hướng tiếp theo paper nêu:** implementation hiện chưa semi-synchronous; tác giả đề xuất semi-synchronous edge mechanism để giảm tổng thời gian train và tăng communication efficiency. Paper không có một mục limitations đầy đủ và không đánh giá adversarial/privacy attacks. [PDF, tr. 8, §VI](../perfesskd/Personalized_Federated_Learning_for_Heterogeneous_Edge_Device_Self-Knowledge_Distillation_Approach.pdf#page=8).

---

## 3. FedCAPS — Federated feature selection bằng permutation-invariant embedding và PPO search

**Bài báo:** *Permutation-Invariant Representation Learning for Robust and Privacy-Preserving Feature Selection*, Rui Liu và cộng sự, arXiv:2510.05535v3. **FedCAPS** là phiên bản federated; **CAPS** là backbone centralized. Đây là author manuscript/arXiv v3 trong workspace, chưa phải publisher PDF. Nguồn: [PDF arXiv v3](../permutation_feature_importance/2510.05535v3.pdf), [bản Markdown](../permutation_feature_importance/2510.05535v3.md).

### 3.1. Bài toán và mục tiêu

FedCAPS không phải một thuật toán train classifier toàn cục theo FedAvg. Nó giải bài toán **federated feature selection**: từ các client có private dataset, học một representation space chung cho các feature subsets và tìm subset (f^*) vừa tốt cho downstream task trên toàn federation, vừa nhỏ. Hai vấn đề chính là: feature subset là một tập nên thứ tự feature không được làm đổi embedding; và data/client không cân bằng nên đánh giá từ client nhỏ có thể nhiễu, trong khi raw data không thể gom về server. [PDF, tr. 1–3, §I–III-A](../permutation_feature_importance/2510.05535v3.pdf#page=1), [Markdown, §II](../permutation_feature_importance/2510.05535v3.md#ii-problem-statement).

### 3.2. Giả định hệ thống

- Client (c) giữ (D_c=(X_c,y_c)), có thể khác nhau về số mẫu/phân phối, nhưng framework biểu diễn subset bằng **feature ID sequence chung**. Việc broadcast một candidate sequence cho mọi client ngầm yêu cầu các feature ID đó có ý nghĩa/được ánh xạ nhất quán ở các bên tham gia.
- Mỗi client có thể chạy một data collector/feature-selection evaluator cục bộ và trả score hiệu năng của một subset.
- Encoder, decoder, PPO actor và critic đều chạy tại central server; client không aggregate classifier parameters.
- Server biết (|D_c|) để tính sample-aware weights.
- Privacy claim giới hạn ở việc raw (X_c,y_c) không rời client; paper không mô tả adversarial server, DP hoặc secure computation.

Nguồn: [PDF, tr. 2, §II](../permutation_feature_importance/2510.05535v3.pdf#page=2), [PDF, tr. 5–6, §III, “Privacy-preserving Global Knowledge Aggregation” và “Sample-Aware Weighted Aggregation”](../permutation_feature_importance/2510.05535v3.pdf#page=5).

### 3.3. Thành phần mô hình

1. **Local feature-selection record collector.** Một record là ((f_i,v_i)): chuỗi feature indices của subset và downstream performance trên local dataset. Paper dùng MARLFS để thu record.
2. **Permutation-invariant encoder (omega).** Multihead Attention Block (MAB) không positional encoding/dropout; hai Induced Set Attention Blocks (ISAB) dùng learnable inducing points để giảm pairwise-attention cost từ (O(N^2)) xuống (O(NM)), với (M\ll N).
3. **Decoder (psi).** Pooling by Multihead Attention (PMA), MAB và row-wise feed-forward reconstruct feature-index sequence từ continuous embedding.
4. **PPO actor–critic.** Actor dịch chuyển top-(K) seed embeddings sang candidate embeddings tốt hơn; decoder biến chúng lại thành subsets. Critic ước lượng reward, nhờ đó không phải hỏi client ở mọi policy step.
5. **Sample-aware evaluator.** Candidate subsets được client đánh giá local; server tổng hợp score theo tỷ lệ số mẫu và định kỳ dùng feedback thật để hiệu chỉnh critic.

Nguồn: [PDF, tr. 3–6, §III](../permutation_feature_importance/2510.05535v3.pdf#page=3), [Markdown, §III](../permutation_feature_importance/2510.05535v3.md#iii-methodology).

### 3.4. Thuật toán client–server theo từng bước

**Giai đoạn A — thu tri thức chọn feature**

1. Mỗi client dùng local data để sinh nhiều candidate subsets và đo downstream score (v_i).
2. Client upload record ((f_i,v_i)), không upload raw features/samples.
3. Server gộp records và tăng dữ liệu bằng các hoán vị của cùng feature-index set.

**Giai đoạn B — học unified embedding space tại server**

4. Encoder đưa từng sequence (f_i) qua hai ISAB để tạo embedding (E_i); không dùng positional encoding để giảm permutation bias.
5. Decoder PMA–MAB–rFF reconstruct sequence; server train encoder–decoder bằng negative log-likelihood reconstruction loss.

**Giai đoạn C — policy-guided search**

6. Server xếp hạng historical records theo performance và lấy top-(K) làm seed.
7. PPO actor biến (E_i) thành (E_i^+); decoder tạo candidate subset (f_i^+).
8. Server broadcast candidate indices tới mọi participating client.
9. Client áp dụng (f_i^+) lên (X_c), train/evaluate downstream model local và trả score (v_{c,i}).
10. Server tính weighted global score (hat v_i), dùng feedback định kỳ để calibrate critic; phần lớn PPO updates dựa trên critic để giảm communication.
11. Actor tối ưu clipped PPO objective; critic tối ưu sai số giữa predicted value và discounted return.
12. Sau search, server chọn candidate có (hat v_i) lớn nhất làm (f^*) và phân phối feature indices cuối tới client.

Nguồn: [PDF, tr. 3–6, §III-A và Fig. 2](../permutation_feature_importance/2510.05535v3.pdf#page=3).

### 3.5. Hàm mục tiêu và công thức quan trọng

Mục tiêu federated feature selection:

\[
f^*=\psi(E^*)=\arg\max_{E\in\mathcal E}\sum_{c=1}^{C}W_c\,\mathcal M\!\left(X_c[\psi(E)]\right).
\]

Hai ISAB tạo encoder:

\[
ISAB_M(f)=MAB(f,H,H),\quad H=MAB(I,f,f),\qquad
\omega(f)=ISAB_M(ISAB_M(f)).
\]

Reconstruction objective:

\[
\mathcal L_{rec}=-\log P_\psi(f\mid E)
=-\sum_{n=1}^{N}\log P_\psi(f_n\mid h_n).
\]

Critic và clipped actor objectives:

\[
\mathcal L_{critic}=\frac1T\sum_{t=1}^{T}\big(V(s_t)-G_t\big)^2,
\]

\[
\mathcal L_{actor}=\hat{\mathbb E}_t\left[
\min\left(r_t(\theta)\hat A_t,
clip(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right].
\]

Sample-aware aggregation:

\[
W_c=\frac{|D_c|}{\sum_{j=1}^{C}|D_j|},\qquad
\hat v_i=\sum_{c=1}^{C}W_c v_{c,i}.
\]

Paper còn định nghĩa reward đa mục tiêu để cân bằng tăng downstream performance với giảm subset length qua hệ số (lambda); do biểu thức reward trong manuscript có ký hiệu dấu/ngoặc chưa thật sáng rõ, khi tái hiện nên bám code hoặc đặc tả đã chốt thay vì tự sửa công thức. [PDF, tr. 2–6, Eqs. (1)–(11)](../permutation_feature_importance/2510.05535v3.pdf#page=2).

### 3.6. Privacy và communication boundary

- **Không rời client:** raw (X_c,y_c), dữ liệu sau chọn feature và downstream training samples.
- **Client → server lúc khởi tạo:** feature ID sequences và local performance scores; server cũng cần số mẫu (|D_c|).
- **Server → client khi search:** candidate feature-index sequences.
- **Client → server khi search:** local performance của từng candidate; feedback có thể thưa vì critic thay thế nhiều lượt đánh giá.
- **Không truyền:** full downstream model weights/gradients theo protocol FedCAPS.

Do đó “privacy-preserving” trong paper có nghĩa raw-data locality, không có tuyên bố formal DP/security guarantee. Feature IDs, sample count và performance scores vẫn nằm phía server theo đúng protocol. [PDF, tr. 5–6, §III](../permutation_feature_importance/2510.05535v3.pdf#page=5).

### 3.7. Thiết lập thí nghiệm gốc

- **Dataset:** 14 bộ public, gồm SpectF, SVMGuide3, German Credit, Credit Default, SpamBase, Megawatt, Innosphere, Mice-Protein, Coil-20, Fashion-MNIST, UrbanSound, OpenML_589, OpenML_616 và IQ-Dataset; phủ binary classification, multiclass classification và regression.
- **Downstream evaluator:** Random Forest; five-fold cross-validation và holdout setting.
- **Metric:** classification dùng F1/precision/recall/ROC-AUC; multiclass dùng micro-F1/precision/recall/macro-F1; regression dùng (1-)MAE, (1-)MSE, (1-)RAE, (1-)RMSE.
- **Record collection:** MARLFS 300 epochs; hoán vị mỗi record 25 lần.
- **Encoder/decoder:** 2 ISAB; decoder PMA+MAB+rFF; 4 attention heads; embedding 128; batch 64; step size 0,001; seed/inducing-point dimension 32.
- **Search:** top 25 seeds; 10 search epochs; batch 512; actor LR 0,0003; critic LR 0,001; reward trade-off 0,1; discount 0,99; 1.000 search steps; PPO clip 0,2.
- **Baselines:** 12 feature-selection baselines cho CAPS; FedAvg, FedNTD, FedProx, MOON cho FedCAPS; thêm ablation collector/encoder/search, permutation sensitivity, seed sensitivity, downstream-model robustness, subset-size study, (lambda)-sensitivity và case study.
- **Hardware/software:** Windows 11, Ryzen 5 5600X, RTX 3070 Ti, Python 3.10.15, PyTorch 2.5.1.

Nguồn: [PDF, tr. 6–11, §IV](../permutation_feature_importance/2510.05535v3.pdf#page=6), [Markdown, §IV](../permutation_feature_importance/2510.05535v3.md#iv-experiments).

### 3.8. Ưu điểm và giới hạn paper tự nêu

**Ưu điểm paper nêu:** permutation-invariant set encoder tránh permutation bias; inducing points hạ attention complexity; PPO không phụ thuộc giả định embedding space lồi; top-(K) seeds tăng ổn định/tốc độ; critic với sparse feedback giảm communication; sample-size weighting giảm nhiễu từ client nhỏ; feature subset cuối nhỏ hơn nhưng vẫn duy trì/cải thiện downstream performance. [PDF, tr. 3–11, §III–IV](../permutation_feature_importance/2510.05535v3.pdf#page=3).

**Giới hạn paper nêu:** manuscript không có mục limitations/future work riêng. Các hạn chế được nói trực tiếp chủ yếu là của thiết kế trước khi được khắc phục: full pairwise attention có (O(N^2)), embedding search phi lồi và random seed dễ đi vào vùng kém. Vì vậy không nên gán thêm cho tác giả các bảo đảm mà paper không nêu, đặc biệt không nên mô tả FedCAPS là DP, secure aggregation hoặc chống inference attack. [PDF, tr. 3–5, §III-A](../permutation_feature_importance/2510.05535v3.pdf#page=3), [PDF, tr. 11, §VI](../permutation_feature_importance/2510.05535v3.pdf#page=11).

---

## 4. pFedES — chia sẻ proxy homogeneous feature extractor cho personalized model-heterogeneous FL

**Bài báo:** *pFedES: Generalized Proxy Feature Extractor Sharing for Model Heterogeneous Personalized Federated Learning*, Liping Yi và cộng sự, AAAI-25. Nguồn: [PDF gốc](../pfedes/00121-YiL.pdf), [bản Markdown](../pfedes/00121-YiL.md).

### 4.1. Bài toán và mục tiêu

pFedES giải bài toán MHPFL: mỗi client muốn giữ một personalized model khác kiến trúc/kích thước vì phân phối dữ liệu và tài nguyên riêng, nên không thể FedAvg trực tiếp full model. Phương pháp thêm cùng một proxy feature extractor nhỏ trước mọi local heterogeneous model. Chỉ proxy đồng nhất được chia sẻ/aggregate để chuyên chở global knowledge; full local model vẫn riêng tư và dùng cho inference. [PDF, tr. 1–3, Abstract, Introduction và Preliminaries](../pfedes/00121-YiL.pdf#page=1), [Markdown, “The Proposed pFedES Approach”](../pfedes/00121-YiL.md#the-proposed-pfedes-approach).

### 4.2. Giả định hệ thống

- Một server, (N) client, mỗi round chọn (K=C\cdot N) client.
- Mọi client làm cùng supervised image-classification task và có input/output semantics tương thích, nhưng local architectures (F_k(\omega_k)) có thể khác nhau.
- Client data non-IID; local train/test cùng phân phối của client trong thí nghiệm.
- Mọi client phải chạy cùng proxy extractor (G(\theta)); proxy giữ nguyên input dimension ở output để enhanced data có thể đưa vào từng local model.
- Server chỉ aggregate proxy parameters; personalized model không rời client.

Mục tiêu personalized gốc là tối thiểu tổng local losses, sau đó paper viết lại khi thêm proxy:

\[
\min_{\omega_0,\ldots,\omega_{N-1}}\sum_{k=0}^{N-1}\mathcal L_k(F_k(\omega_k);D_k),
\]

\[
\min_{\theta,\omega_0,\ldots,\omega_{N-1}}
\sum_{k=0}^{N-1}\mathcal L_k(\{G(\theta),F_k(\omega_k)\};D_k).
\]

Nguồn: [PDF, tr. 3–4, Preliminaries và “The Proposed pFedES Approach”](../pfedes/00121-YiL.pdf#page=3).

### 4.3. Thành phần mô hình

1. **Global/local proxy extractor (G(\theta)).** Cùng kiến trúc cho mọi client; paper dùng CNN hai convolution layers với `padding=same`, nhỏ hơn nhiều local model.
2. **Personalized heterogeneous model (F_k(\omega_k)).** Kiến trúc, capacity và weights riêng của client (k); dùng chính model này cho inference cuối.
3. **Hai nhánh input khi train local model.** Original sample (x) và enhanced sample (hat x=G(\theta;x)) cùng đi qua (F_k); hai hard losses được trộn để tránh proxy kém ổn định ở round đầu làm hỏng local learning.
4. **Alternating freeze/train.** Pha 1 freeze proxy để truyền global knowledge vào local model; pha 2 freeze local model để truyền personalized knowledge ngược vào proxy.
5. **Weighted proxy aggregation.** Server chỉ lấy trung bình proxy weights theo local sample count.

Nguồn: [PDF, tr. 3–5, Fig. 1 và “Iterative Training and Model Aggregation”](../pfedes/00121-YiL.pdf#page=3).

### 4.4. Thuật toán client–server theo từng bước

1. Round (t): server chọn (S^t) gồm (K) client và broadcast (G(\theta^{t-1})).
2. Client (k) nhận proxy và giữ local model (F_k(\omega_k^{t-1})) từ trạng thái cá nhân trước đó.
3. **Pha local A:** freeze proxy. Với batch ((x,y)), tạo (hat x=G(\theta^{t-1};x)). Local model dự đoán trên cả (hat x) và (x), tạo (hat y_1,hat y_2); trộn hai cross-entropy losses rồi chỉ cập nhật (omega_k).
4. **Pha local B:** freeze local model vừa cập nhật. Chạy $x\to G\to \hat x\to F_k$, tính supervised loss với $y$, rồi chỉ cập nhật proxy từ $\theta^{t-1}$ thành $\theta_k^t$.
5. Client upload (G(\theta_k^t)); giữ (F_k(\omega_k^t)) local.
6. Server weighted-average proxy parameters để tạo $\theta^t$.
7. Lặp đến khi personalized local models hội tụ. Khi inference, bỏ proxy và chỉ dùng (F_k(\omega_k)), theo mô tả của paper.

Nguồn: [PDF, tr. 3–5, “Overview” và “Iterative Training and Model Aggregation”](../pfedes/00121-YiL.pdf#page=3), [Markdown, cùng phần](../pfedes/00121-YiL.md#iterative-training-and-model-aggregation).

### 4.5. Hàm mục tiêu và công thức quan trọng

Hai prediction branch:

\[
\hat x=G(\theta^{t-1};x),\qquad
\hat y_1=F_k(\omega_k^{t-1};\hat x),\quad
\hat y_2=F_k(\omega_k^{t-1};x).
\]

Hai hard losses và integrated local-model loss:

\[
\ell_1=\ell(\hat y_1,y),\quad \ell_2=\ell(\hat y_2,y),\qquad
\ell_\omega=\mu\ell_1+(1-\mu)\ell_2,\;\mu\in(0,0.5].
\]

Local-model update:

\[
\omega_k^t\leftarrow\omega_k^{t-1}-\eta_\omega\nabla\ell_\omega.
\]

Proxy-training loss/update khi local model bị freeze:

\[
\hat y=F_k(\omega_k^t;G(\theta^{t-1};x)),\quad
\ell_\theta=\ell(\hat y,y),\quad
\theta_k^t\leftarrow\theta^{t-1}-\eta_\theta\nabla\ell_\theta.
\]

Server aggregation:

\[
\theta^t=\sum_{k\in S^t}\frac{n_k}{n}\theta_k^t.
\]

Paper còn chứng minh non-convex convergence rate (O(1/T)) dưới gradient Lipschitz smoothness, unbiased stochastic gradients và bounded variance; đây là kết quả có điều kiện trên các Assumptions 1–2, không phải bảo đảm vô điều kiện. [PDF, tr. 5, “Convergence Analysis,” Eqs. (12)–(17)](../pfedes/00121-YiL.pdf#page=5).

### 4.6. Privacy và communication boundary

- **Không rời client:** raw (D_k); full heterogeneous architecture/weights (F_k(\omega_k)); enhanced samples (hat x).
- **Server → client:** small global proxy extractor weights (G(\theta)).
- **Client → server:** updated local proxy extractor weights (G(\theta_k)), cùng sample count cần cho weighting.
- **Inference:** personalized (F_k) local; paper nói proxy chỉ phục vụ train.

Paper lập luận chỉ có proxy, không có enhanced data, thì không thể đảo ngược raw input; và không gửi full local model giúp bảo vệ model intellectual property. Đây là lập luận boundary của tác giả, không phải đánh giá tấn công thực nghiệm hoặc formal DP guarantee. [PDF, tr. 5, “Discussion”](../pfedes/00121-YiL.pdf#page=5).

### 4.7. Thiết lập thí nghiệm gốc

- **Dataset:** MNIST, CIFAR-10, CIFAR-100.
- **Non-IID:** mỗi client chỉ có 2/10 classes cho MNIST/CIFAR-10 và 10/100 classes cho CIFAR-100; local split train/test 8:2, cùng local distribution.
- **Model:** homogeneous dùng CNN-1; heterogeneous gán đều một trong CNN-1…CNN-5; proxy là CNN hai layer (MNIST giảm filters ở layer hai).
- **Quy mô:** (N\in\{10,50,100\}), participation lần lượt gồm các cấu hình 100%, 20%, 10%.
- **Training:** SGD, (eta=\eta_\omega=\eta_\theta=0,01); grid (E\in\{1,10\}), batch (B\in\{64,128,256,512\}); tune (mu) và proxy epochs (E_{fe}); (T\in\{100,500\}) rounds; mỗi experiment chạy 3 trials.
- **Baselines:** Standalone, LG-FedAvg, FedGH, FML, FedKD, FedAPEN, FD, FedProto, FedTGP.
- **Metric:** mean individual client accuracy, số transmitted parameters và FLOPs tại target accuracy.
- **Hardware:** PyTorch trên 4 RTX 3090 24 GB.

Nguồn: [PDF, tr. 5–7, “Experimental Evaluation”](../pfedes/00121-YiL.pdf#page=5), [Markdown, “Experiment Setup”](../pfedes/00121-YiL.md#experiment-setup).

### 4.8. Ưu điểm và giới hạn paper tự nêu

**Ưu điểm paper nêu:** hỗ trợ full model heterogeneity mà không cần public data; proxy nhỏ làm communication thấp hơn truyền full model; alternating training trao đổi tri thức hai chiều; giữ kín local model structure/IP; paper báo trade-off accuracy–communication–computation tốt và convergence (O(1/T)) dưới giả định. [PDF, tr. 5–7, Discussion, Convergence Analysis và Results](../pfedes/00121-YiL.pdf#page=5).

**Giới hạn paper nêu:** bài không có limitations/future-work section rõ ràng. Phần Motivation chỉ ra joint end-to-end training proxy+local model gây global knowledge phai sau batch đầu và tăng memory, nên paper thay bằng alternating freeze/train; đó là giới hạn của phương án trực giác mà pFedES xử lý. Phạm vi được tác giả xác định là supervised image classification với shared proxy có output cùng kích thước input. [PDF, tr. 3–4, “Motivation”](../pfedes/00121-YiL.pdf#page=3), [PDF, tr. 7, “Conclusion”](../pfedes/00121-YiL.pdf#page=7).

---

## 5. Lightweight Federated Learning for On-Device NILM — MNAS và adaptive mutual distillation qua proxy

**Bài báo:** *Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring*, Yehui Li, Ruiyang Yao, Dalin Qin và Yi Wang, *IEEE Transactions on Smart Grid* 16(2), 2025. Paper không đặt acronym riêng cho toàn framework; thành phần là memory-efficient NAS (MNAS) cộng adaptive federated mutual learning. Bản Markdown trong workspace bắt đầu từ Section IV, vì vậy Sections I–III dưới đây được đối chiếu trực tiếp từ PDF. Nguồn: [PDF gốc](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf), [Markdown phần IV–V](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.md).

### 5.1. Bài toán và mục tiêu

NILM ước lượng công suất từng thiết bị từ tổng công suất hộ gia đình. Với (N) appliance:

\[
P_t=\sum_{n=1}^{N}P_{n,t}+X_t+e_t,\qquad
\hat P_{n,t}=f_n(w_n,P_t),
\]

trong đó (X_t) là công suất của appliance không được mô hình hóa và (e_t) là noise. Paper xử lý hai nút thắt on-device: model lớn vượt giới hạn memory, còn model nhỏ train từ ít dữ liệu local dễ overfit. Mục tiêu là tự tìm personalized architecture phù hợp mỗi appliance/household rồi dùng FL để hấp thụ tri thức từ nhiều hộ mà không chia raw high-frequency meter data. [PDF, tr. 1–3, §I–II](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=1).

### 5.2. Giả định hệ thống

- Có (K) households/end devices và một aggregation server; mỗi thiết bị có labeled aggregate/appliance power data local.
- Personalized architectures có thể khác nhau, nhưng mỗi client có thêm một proxy model **cùng kiến trúc** để aggregate.
- Raw load data và personalized weights ở lại device; proxy weights được upload.
- Real-time NILM cung cấp latency limit (T_0) bằng sliding-window duration để NAS phạt kiến trúc quá chậm.
- Paper giả định communication network và aggregation server đáng tin/có an toàn; malicious attacker ngoài phạm vi.

Nguồn: [PDF, tr. 2–3, §II–III-A](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=2), [PDF, tr. 10, §V](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=10).

### 5.3. Thành phần mô hình

1. **Compressed NAS search space.** Các convolution kernels lồng nhau chia sẻ weights; device chỉ lưu largest kernel thay vì toàn bộ kernels độc lập.
2. **Single-path search.** Mỗi search step chỉ kích hoạt một operation path qua binary gate; Gumbel-Softmax làm relaxation khả vi để cập nhật architecture parameters.
3. **Hardware-aware objective.** Measured latency của operation được đưa vào loss, phạt khi network latency vượt sliding-window limit.
4. **Personalized student (f_s(w_s)).** Kiến trúc do MNAS tìm riêng cho appliance/household.
5. **Unified proxy (f_r(w_r)).** Kiến trúc đồng nhất trên mọi device, làm cầu nối để server aggregate. Trong experiment proxy có ba 1-D convolution layers kernel 5 và một dense layer.
6. **Adaptive mutual distillation.** Personalized và proxy model học hai chiều trên cùng local data; distillation tự yếu đi khi cả hai model còn dự đoán kém.

Nguồn: [PDF, tr. 3–6, §III-A–C](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=3).

### 5.4. Thuật toán client–server theo từng bước

**Pha 1 — MNAS độc lập trên từng device**

1. Device dựng compressed supernet từ candidate operations.
2. Mỗi step sample một path qua binary gates/Gumbel-Softmax.
3. Cập nhật weights (w) bằng training loss và architecture parameters (alpha) bằng validation loss cộng latency penalty.
4. Khi search xong, giữ operations có architecture score lớn nhất để tạo personalized model (w_s).

**Pha 2 — adaptive federated mutual learning**

5. Mỗi device có personalized model (w_s) và copy proxy (w_r) từ global proxy.
6. Trên local batch, cả hai model dự đoán; mỗi model có supervised label loss riêng.
7. Tính disagreement giữa hai predictions, chia cho tổng supervised losses để tạo adaptive distillation term. Khi cả hai đang sai/loss lớn, distillation contribution nhỏ; khi đã đáng tin hơn, mutual transfer mạnh hơn.
8. Cập nhật song song (w_s) và (w_r) theo respective total losses.
9. Device giữ personalized (w_s), upload chỉ proxy (w_r).
10. Server lấy trung bình proxy weights $\bar w_r=\frac1K\sum_k w_r^k$, rồi phân phối lại cho client.
11. Lặp mutual-learning/aggregation trong (R) rounds.
12. Sau federation, mỗi device fine-tune personalized model bằng local data để khớp local distribution; personalized model dùng cho NILM inference.

Nguồn: [PDF, tr. 3 và 6, §III-A, §III-C và Algorithm 1](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=3).

### 5.5. Hàm mục tiêu và công thức quan trọng

NAS bi-level objective:

\[
\min_\alpha L_{val}(w^*(\alpha),\alpha),\qquad
w^*(\alpha)=\arg\min_w L_{train}(w,\alpha).
\]

Latency của architecture và hardware-aware search objective:

\[
T(\alpha)=\sum_{e=1}^{E}\sum_{i=1}^{M}g_{e,i}(\alpha)L(o_i),
\]

\[
\min_\alpha L_{val}(w,\alpha)+\left(\frac{T(\alpha)}{T_0}\right)^\lambda,
\quad w=\arg\min_w L_{train}(w,\alpha).
\]

Mutual-learning losses:

\[
L_{train,s}=\ell_s(w_s,x)+\ell_d(w_s,w_r,x),\qquad
L_{train,r}=\ell_r(w_r,x)+\ell_d(w_s,w_r,x),
\]

với (y_s=f_s(w_s,x)), (y_r=f_r(w_r,x)), (ell_s=\ell(y,y_s)), (ell_r=\ell(y,y_r)), và:

\[
\ell_d(w_s,w_r,x)=\frac{\ell(y_s,y_r)}{\ell_s(w_s,x)+\ell_r(w_r,x)}.
\]

Ở đây (ell) có thể là (L_2). Dạng chuẩn hóa này chính là adaptive weight: denominator lớn khi label performance còn kém nên làm nhỏ mutual-distillation signal. [PDF, tr. 2–6, Eqs. (3)–(19)](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=2).

### 5.6. Privacy và communication boundary

- **Không rời device:** raw aggregate/appliance power sequences, personalized architecture và (w_s), architecture-search states.
- **Server → client:** aggregated unified proxy weights $\bar w_r$.
- **Client → server:** updated unified proxy weights (w_r^k).
- **Server aggregate:** unweighted average (1/K) trong Algorithm 1, không phải sample-size weighted average.
- **Không có trong paper:** secure aggregation, encryption, DP hoặc formal inference-attack analysis.

Nguồn: [PDF, tr. 3, §III-A](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=3), [PDF, tr. 6, §III-C và Algorithm 1](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=6).

### 5.7. Thiết lập thí nghiệm gốc

- **Dataset:** REFIT — 20 UK houses, aggregate + 9 appliance monitors, 8-second resolution; REDD — 6 US houses, 9–24 appliances/house, 3-second resolution.
- **Appliances:** REFIT gồm fridge, washing machine, dishwasher, microwave, kettle; REDD báo fridge, dishwasher, microwave.
- **Benchmarks:** centralized, local, NAS, MNAS, standard federated/FedAvg và proposed MNAS+adaptive federated mutual learning.
- **Backbone/search:** 1-D CNN; fixed baseline gồm 5 conv layers kernel 7 + dense. Search space có 5 nodes, mỗi path có 8 operations: conv 7/5/3/1, max pool, average pool, skip, identity; giữ hai paths có architecture parameter lớn nhất tại mỗi node. Proxy: 3 conv layers kernel 5 + dense.
- **Metric:** MAE và signal aggregate error (SAE); thêm model size, inference time, search/training time, time/space complexity.
- **Robustness:** imbalance bằng chuyển 30% data giữa một tỷ lệ devices; device heterogeneity bằng kéo dài gấp đôi training time ở một tỷ lệ devices.

Nguồn: [PDF, tr. 6–10, §IV](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=6), [Markdown, §IV](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.md#iv-case-studies).

### 5.8. Ưu điểm và giới hạn paper tự nêu

**Ưu điểm paper nêu:** compressed kernels và single-path activation đưa search memory về gần compact-model training; personalized architecture thích nghi appliance/household; latency penalty phù hợp real-time NILM; proxy giải quyết model-architecture heterogeneity; adaptive distillation tránh lan truyền prediction kém ở giai đoạn đầu; raw load data và personalized model được giữ local. [PDF, tr. 3–6, §III](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=3).

**Giới hạn paper nêu rõ:** MNAS vẫn tăng nhẹ compute vì device phải học architecture; giả định network và aggregation server reliable/secure, bỏ qua hackers/malicious attackers; chưa tự quyết định trước liệu local hay global knowledge có lợi cho từng task; communication overhead cần được giảm thêm, tác giả đề xuất gradient quantization. [PDF, tr. 10, §V](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.pdf#page=10), [Markdown, §V](../proxymodel/2025%20Lightweight_Federated_Learning_for_On-Device_Non-Intrusive_Load_Monitoring.md#v-conclusion-and-future-works).

---

## 6. Bảng phân biệt nhanh năm phương pháp gốc

| Phương pháp | Thứ thật sự được cộng tác | Thành phần cá nhân hóa | Thứ client gửi | Cách xử lý heterogeneity chính |
|---|---|---|---|---|
| FD-IDS | Full classifier weights | Không; một global architecture | Full local weights | FedProx + global-teacher KD |
| PerFed-SKD | Selected local model weights | Historical local teacher (V_m) | Weights + local accuracy nếu được chọn | Self-KD theo thời gian + accuracy-based device selection |
| FedCAPS | Feature-selection knowledge/scores | Local evaluation trên private data | Feature IDs, scores, sample count | Permutation-invariant embedding + sample-aware reward |
| pFedES | Small homogeneous feature extractor | Full heterogeneous local model | Proxy extractor weights | Alternating proxy/local training |
| Lightweight federated NILM | Unified proxy model | MNAS-searched appliance/household model | Proxy model weights | Adaptive mutual distillation + memory/latency-aware NAS |

Bảng này chỉ tóm tắt đối tượng trao đổi; chi tiết và các ngoại lệ phải đọc theo từng mục ở trên.
