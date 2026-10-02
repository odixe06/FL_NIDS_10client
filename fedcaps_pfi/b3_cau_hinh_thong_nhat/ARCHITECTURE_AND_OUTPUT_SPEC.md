# FedCAPS task10: kiến trúc và output

Hợp đồng chung nằm tại [`../ARCHITECTURE_AND_OUTPUT_SPEC.md`](../ARCHITECTURE_AND_OUTPUT_SPEC.md).

- Ba notebook đồng nhất: 10 GRU, 10 Transformer, 10 CNN-1D.
- Dữ liệu đã dùng chính sách chuẩn hóa chung upstream; notebook không fit lại
  scaler.
- Feature search dùng search pool phân tầng tối đa 100.000 dòng/client và
  deterministic stratified 5-fold CV. Global test tuyệt đối không thuộc search.
- Final personalized classifier dùng 100% dữ liệu local, không local
  validation, học 10 round × một epoch và giữ state riêng qua round.
- Không FedAvg classifier; server là feature-selection stack nên classification
  `not_applicable`.
- Đánh giá 10 client trên global test sau từng round; `evaluation_metrics` có
  100 hàng và `final_round_metrics` có 10 hàng.
- Lưu server feature stack và đủ 10 client checkpoint tại từng round; không có
  checkpoint “best”.

Ngoài output chung, run phải có `feature_selection_records`, `encoder_history`,
`ppo_history`, `candidate_evaluations`, `selected_features`, raw confusion và
personalized test report.
