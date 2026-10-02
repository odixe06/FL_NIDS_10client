# FL_NIDS_10client — Federated NIDS trên CICIoT2023 (10 client non-IID)

Tái hiện và so sánh các phương pháp học liên kết (federated learning) cho phát hiện xâm nhập mạng trên
CICIoT2023 với 10 client non-IID. Mã và notebook do Git quản lý; dữ liệu, checkpoint và file nặng (> 5 MB) do
**DVC** quản lý, lưu trên Google Drive (`NCKH/ciciot2023/dvc-storage`).

## Cấu trúc

Mỗi phương pháp một thư mục; mỗi thư mục có bài báo gốc (`.md`) và các bản build:

| bản build | nghĩa |
|---|---|
| `b1_*` | thử nghiệm đầu (bản riêng của phương pháp) |
| `b2_tai_hien_rieng` | tái hiện riêng từng phương pháp, mỗi kịch bản một thư mục (notebook + `output/`) |
| `b3_cau_hinh_thong_nhat` | cùng một cấu hình chung cho mọi phương pháp: 3 kịch bản 10 client GRU / Transformer / CNN-1D |

| thư mục | nội dung |
|---|---|
| `data/` | `data_description.md`; `raw_merged/`, `global_train_test/`, `10_clients_noniid/` (DVC, ~31,6 GB) |
| `fd_ids/` | FD-IDS (`sensors-25-04309.md`); `b1_dnn_fedprox_kd` (DNN + FedProx + KD), `b2_…`, `b3_…` |
| `perfed_skd/` | PerFed-SKD — personalized FL với self-knowledge distillation; `b2_…`, `b3_…` |
| `fedcaps_pfi/` | FedCAPS + permutation feature importance (`2510.05535v3.md`); `b2_…`, `b3_…` |
| `pfedes/` | pFedES (`00121-YiL.md`); `b2_…`, `b3_…` |
| `proxymodel/` | ProxyModel; `b1_proxy_training`, `b2_…`, `b3_…` |
| `shap_kd/b1_shap_kd/` | KD có chọn đặc trưng bằng SHAP; `cached/` (DVC, 4,9 GB) |
| `b3_dung_chung/` | dùng chung cho mọi `b3_*`: `ARCHITECTURE_AND_OUTPUT_SPEC.md` (hợp đồng kiến trúc + output), `scripts/` (runtime, generator) |
| `reports/b2_bao_cao_5pp/` | báo cáo so sánh 5 phương pháp ở bản build 2 |
| `reports/b3_tong_hop_10pp/` | báo cáo tổng hợp ở bản build 3, `all_methods_metrics.csv` |
| `.claude/`, `.agents/`, `CLAUDE.md` | skill / lệnh cho agent (sinh notebook Kaggle) |

Ghi chú: `b3_dung_chung/README.md` và vài tài liệu bên trong còn nhắc đường dẫn cũ (`task10/`,
`permutation_feature_importance/`) — tương ứng `b3_dung_chung/` và `fedcaps_pfi/b3_cau_hinh_thong_nhat/`.

## Khôi phục

```bash
git clone https://github.com/odixe06/FL_NIDS_10client.git && cd FL_NIDS_10client
# remote DVC trỏ tới Google Drive mount tại /mnt/g (WSL); máy khác: sửa url cho đúng chỗ mount Drive
dvc remote modify --local gdrive url "<đường dẫn tới>/My Drive/NCKH/ciciot2023/dvc-storage"
dvc pull data/10_clients_noniid            # chỉ kéo phần cần (toàn bộ ~36,9 GB)
dvc pull fd_ids/b2_tai_hien_rieng/gru_10_clients/output/fd_ids_ciciot2023_10c_gru/checkpoints
```

Mọi đầu ra DVC: `find . -name '*.dvc' -not -path './.dvc/*'`. Bundle git dự phòng:
`NCKH/ciciot2023/FL_NIDS_10client.bundle` trên Drive.
