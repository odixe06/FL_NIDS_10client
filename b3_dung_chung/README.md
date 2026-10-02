# Task10 federated-learning notebooks

Thư mục này chứa 15 notebook Kaggle: năm phương pháp × ba kịch bản 10 client
GRU/Transformer/CNN-1D.

- Hợp đồng đã chốt: [`ARCHITECTURE_AND_OUTPUT_SPEC.md`](ARCHITECTURE_AND_OUTPUT_SPEC.md)
- Runtime dùng chung cho FD-IDS, PerFed-SKD, pFedES và ProxyModel:
  `scripts/common_client_parallel_runtime.py`
- Generator dùng chung:
  `python scripts/build_common_notebooks.py`
- Runtime/generator FedCAPS riêng:
  `permutation_feature_importance/scripts/`

Mỗi method folder cũng giữ một entry-point generator tương thích với tên script
cũ. Notebook được tạo theo kiểu self-contained: runtime đầy đủ được nhúng vào
notebook và ghi ra `/kaggle/temp` trước khi tạo hai persistent GPU worker.

Kiểm tra cấu trúc toàn bộ notebook:

```bash
python ../.agents/skills/kaggle-training-notebook/scripts/validate_client_parallel_notebook.py \
  $(find . -name '*.ipynb' -type f | sort)
```
