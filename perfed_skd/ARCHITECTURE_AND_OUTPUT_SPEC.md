# PerFed-SKD task10: kiến trúc và output

Hợp đồng chung nằm tại [`../ARCHITECTURE_AND_OUTPUT_SPEC.md`](../ARCHITECTURE_AND_OUTPUT_SPEC.md).

- Ba notebook: 10 GRU, 10 Transformer, 10 CNN-1D.
- Split phân tầng cố định 90% local train / 10% local validation.
- Chỉ validation accuracy chọn client cho round tiếp theo; validation vẫn ghi
  đủ 10 metrics sau mỗi round. Global test không tham gia selection.
- Server pretrain một epoch; personalized client state tồn tại qua 10 round;
  selected clients nhận global state và server tổng hợp các selected update.
- Đánh giá server và đủ 10 client trên global test sau từng round.
- Lưu `server.pt` và 10 client checkpoint ở đủ 10 round; không checkpoint
  “best”.
- `evaluation_metrics` có 110 hàng, `validation_metrics` có 100 hàng và
  `final_round_metrics` có 11 hàng.

Kiến trúc classifier và cây output bắt buộc theo mục 3, 7 và 8 của hợp đồng
chung.
