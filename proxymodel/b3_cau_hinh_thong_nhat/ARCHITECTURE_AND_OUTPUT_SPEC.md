# ProxyModel task10: kiến trúc và output

Hợp đồng chung nằm tại [`../ARCHITECTURE_AND_OUTPUT_SPEC.md`](../ARCHITECTURE_AND_OUTPUT_SPEC.md).

- Ba notebook: 10 GRU, 10 Transformer, 10 CNN-1D.
- 10 round, 100% dữ liệu local, không local validation.
- Personalized classifier và local CNN-1D proxy tối ưu bằng adaptive mutual
  distillation; chỉ proxy được FedAvg có trọng số.
- Kịch bản CNN-1D vẫn giữ local CNN và proxy CNN thành hai model/state riêng.
- Round 10 thực hiện final personalized CE fine-tune của phương pháp trước khi
  checkpoint và đánh giá.
- Đánh giá global CNN-1D proxy và đủ 10 client sau từng round.
- Lưu `server.pt` và đủ 10 client checkpoint ở từng round; không checkpoint
  “best”.
- `evaluation_metrics` có 110 hàng; `final_round_metrics` có 11 hàng.

Kiến trúc classifier/proxy và cây output bắt buộc theo mục 3, 7 và 8 của hợp
đồng chung.
