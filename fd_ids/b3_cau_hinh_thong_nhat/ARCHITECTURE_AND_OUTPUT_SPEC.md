# FD-IDS task10: kiến trúc và output

Hợp đồng chung nằm tại [`../ARCHITECTURE_AND_OUTPUT_SPEC.md`](../ARCHITECTURE_AND_OUTPUT_SPEC.md).

- Ba notebook: 10 GRU, 10 Transformer, 10 CNN-1D.
- 10 round, một local epoch/round, 100% dữ liệu từng client, không local validation.
- Mỗi client reset từ global classifier ở đầu round; loss gồm CE, KD teacher
  global và FedProx; server FedAvg có trọng số.
- Client và server có cùng state sau aggregation, vì vậy chỉ server được đánh
  giá trên global test sau từng round.
- Lưu `server.pt` ở đủ 10 round, không lưu client duplicate và không có
  checkpoint “best”.
- `evaluation_metrics.csv/json` có 10 hàng, `final_round_metrics` có 1 hàng,
  mỗi hàng chứa đúng 10 metrics chung.

Kiến trúc classifier và cây output bắt buộc theo mục 3, 7 và 8 của hợp đồng
chung.
