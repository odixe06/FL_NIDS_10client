# pFedES task10: kiến trúc và output

Hợp đồng chung nằm tại [`../ARCHITECTURE_AND_OUTPUT_SPEC.md`](../ARCHITECTURE_AND_OUTPUT_SPEC.md).

- Ba notebook: 10 GRU, 10 Transformer, 10 CNN-1D.
- 10 round, 100% dữ liệu local, không local validation.
- Personalized classifier giữ state riêng; proxy feature extractor 25→25 có
  đúng 57 tham số và được FedAvg có trọng số.
- Server proxy không phát logits 34 lớp nên server evaluation là
  `not_applicable`; đánh giá đủ 10 client sau từng round.
- Lưu `server.pt` và đủ 10 client checkpoint ở từng round; không checkpoint
  “best”.
- `evaluation_metrics` có 100 hàng; `final_round_metrics` có 10 hàng.

Kiến trúc classifier, proxy và cây output bắt buộc theo mục 3, 7 và 8 của hợp
đồng chung.
