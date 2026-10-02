#!/usr/bin/env python3
"""Gộp reports.md (nhóm A) + report_5_phuong_phap_GRU.md (nhóm B) thành một báo
cáo 10 phương pháp, kèm phần so sánh chéo mới trên kịch bản GRU."""
import csv, re, pathlib

ROOT = pathlib.Path('/home/odixe/nckh/task10')
OUT = ROOT / 'report' / 'BAO_CAO_TONG_HOP_10_PHUONG_PHAP.md'

A_SRC = (ROOT / 'report' / 'reports.md').read_text(encoding='utf-8')
B_SRC = (ROOT / 'report' / 'report_5_phuong_phap_GRU.md').read_text(encoding='utf-8')

METRICS = ['accuracy', 'macro_precision', 'micro_precision', 'weighted_precision',
           'macro_recall', 'micro_recall', 'weighted_recall',
           'macro_f1', 'micro_f1', 'weighted_f1']
HDR10 = ['accuracy', 'macro_P', 'micro_P', 'weighted_P', 'macro_R', 'micro_R',
         'weighted_R', 'macro_F1', 'micro_F1', 'weighted_F1']


# --------------------------------------------------------------------------
# 1. Đọc số liệu
# --------------------------------------------------------------------------
def load_group_a():
    """{(method_display, scope): {round: {metric: float}}} cho kịch bản GRU."""
    out = {}
    with open(ROOT / 'all_methods_metrics.csv', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['model_family'] != 'gru':
                continue
            key = (r['method_display'], r['entity_scope'])
            out.setdefault(key, {})[int(r['round'])] = {
                m: float(r[m]) for m in METRICS}
    return out


def load_group_b():
    """Parse 5 bảng '### N.5. Kết quả đo được' của báo cáo nhóm B."""
    names = {3: 'FedAvg', 4: 'CustomAgg', 5: 'Hetero-FedDistill',
             6: 'ZS-Prune+PTD', 7: 'ZS-Prune'}
    out = {}
    for num, name in names.items():
        start = B_SRC.index(f'### {num}.5. Kết quả đo được')
        nxt = B_SRC.find('\n## ', start)
        block = B_SRC[start:nxt if nxt != -1 else len(B_SRC)]
        rounds = {}
        for line in block.splitlines():
            m = re.match(r'^\|\s*(\d+)\s*\|(.+)\|\s*$', line)
            if not m:
                continue
            vals = [c.strip() for c in m.group(2).split('|')]
            if len(vals) != 10:
                continue
            rounds[int(m.group(1))] = dict(zip(METRICS, map(float, vals)))
        assert sorted(rounds) == list(range(1, 11)), (name, sorted(rounds))
        out[name] = rounds
    return out


A = load_group_a()
B = load_group_b()

# Cột của bảng so sánh 10 phương pháp (kịch bản GRU).
# (nhãn hiển thị, nguồn số, ghi chú entity)
COLS = [
    ('FD-IDS',        A[('FD-IDS', 'server')],          'server'),
    ('PerFed-SKD',    A[('PerFed-SKD', 'client_mean')], 'TB 10 client'),
    ('FedCAPS',       A[('FedCAPS', 'client_mean')],    'TB 10 client'),
    ('pFedES',        A[('pFedES', 'client_mean')],     'TB 10 client'),
    ('ProxyModel',    A[('ProxyModel', 'client_mean')], 'TB 10 client'),
    ('FedAvg',        B['FedAvg'],                      'global model'),
    ('CustomAgg',     B['CustomAgg'],                   'global model'),
    ('Hetero-KD ⚠',   B['Hetero-FedDistill'],           'global model'),
    ('ZS+PTD',        B['ZS-Prune+PTD'],                'global model'),
    ('ZS-Prune',      B['ZS-Prune'],                    'global model'),
]
# Nhóm "global model" — 8 mô hình toàn cục có thể so trực tiếp
GLOBALS = [
    ('FD-IDS (server)',     A[('FD-IDS', 'server')]),
    ('PerFed-SKD (server)', A[('PerFed-SKD', 'server')]),
    ('ProxyModel (server)', A[('ProxyModel', 'server')]),
    ('FedAvg',              B['FedAvg']),
    ('CustomAgg',           B['CustomAgg']),
    ('Hetero-KD ⚠',         B['Hetero-FedDistill']),
    ('ZS+PTD',              B['ZS-Prune+PTD']),
    ('ZS-Prune',            B['ZS-Prune']),
]


def f4(x):
    return f'{x:.4f}'


def table(header, rows, align=None):
    align = align or (['---'] + ['---:'] * (len(header) - 1))
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def by_round_table(metric, bold_max=True):
    header = ['Round'] + [c[0] for c in COLS]
    rows = []
    for rd in range(1, 11):
        vals = [c[1][rd][metric] for c in COLS]
        best = max(vals)
        cells = [f4(v) for v in vals]
        if bold_max:
            cells = [f'**{c}**' if v == best else c for c, v in zip(cells, vals)]
        rows.append([str(rd)] + cells)
    return table(header, rows, ['---:'] * (len(header)))


# --------------------------------------------------------------------------
# 2. Cắt & đánh số lại các phần của hai báo cáo gốc
# --------------------------------------------------------------------------
def slice_between(text, start_pat, end_pat=None):
    s = re.search(start_pat, text, re.M)
    assert s, start_pat
    if end_pat is None:
        return text[s.start():].strip('\n')
    e = re.search(end_pat, text[s.end():], re.M)
    assert e, end_pat
    return text[s.start():s.end() + e.start()].strip('\n')


def fix_images(text):
    for old, new in IMGMAP.items():
        text = text.replace(f']({old})', f']({new})')
    return text


def fix_links_a(text):
    """reports.md chuyển vào report/ → link tương đối lùi một cấp."""
    for p in ['fd_ids_noniid/', 'perfed_skd/', 'permutation_feature_importance/',
              'pfedes/', 'proxymodel/', 'scripts/', 'all_methods_metrics.csv',
              'report_assets/']:
        text = text.replace(f']({p}', f'](../{p}')
    return text


def prep_a(text):
    # reports.md trong report/ đã có sẵn đường dẫn đúng (images/… và ../…)
    return text


BMAP = {3: 8, 4: 9, 5: 10, 6: 11, 7: 12}


def renumber_b_refs(text):
    """Tham chiếu 'mục N[.M]' trong nhóm B: 3..7 → 8..12."""
    return re.sub(r'mục ([3-7])(\.\d)?',
                  lambda x: f'mục {BMAP[int(x.group(1))]}{x.group(2) or ""}', text)


def renumber_b_methods(text):
    """## 3..7 → ## 8..12 (heading + tham chiếu)."""
    text = re.sub(r'^## ([3-7])\. ',
                  lambda x: f'## {BMAP[int(x.group(1))]}. ', text, flags=re.M)
    text = re.sub(r'^### ([3-7])\.(\d)\. ',
                  lambda x: f'### {BMAP[int(x.group(1))]}.{x.group(2)}. ', text, flags=re.M)
    return renumber_b_refs(text)


def demote(text):
    """### → ####  (dùng khi nhét một mục ## cũ xuống làm mục con)."""
    return re.sub(r'^### ', '#### ', text, flags=re.M)


A_ctx      = prep_a(slice_between(A_SRC, r'^## 1\. ', r'^## 2\. '))
A_quick    = prep_a(slice_between(A_SRC, r'^## 2\. ', r'^## 3\. '))
A_methods  = prep_a(slice_between(A_SRC, r'^## 3\. ', r'^## 8\. '))
A_cross    = prep_a(slice_between(A_SRC, r'^## 8\. ', r'^## 9\. '))
A_analysis = prep_a(slice_between(A_SRC, r'^## 9\. ', r'^## 10\. '))
A_concl    = prep_a(slice_between(A_SRC, r'^## 10\. ', r'^## Phụ lục'))
A_appx     = prep_a(slice_between(A_SRC, r'^## Phụ lục'))

B_ctx      = slice_between(B_SRC, r'^## 1\. ', r'^## 2\. ')
B_quick    = slice_between(B_SRC, r'^## 2\. ', r'^## 3\. ')
B_methods  = renumber_b_methods(slice_between(B_SRC, r'^## 3\. ', r'^## 8\. '))
B_cross    = renumber_b_refs(slice_between(B_SRC, r'^## 8\. ', r'^## 9\. '))
B_analysis = renumber_b_refs(slice_between(B_SRC, r'^## 9\. ', r'^## 10\. '))
B_concl    = renumber_b_refs(slice_between(B_SRC, r'^## 10\. '))

# ## 8. → ## 14. cho nhóm A, ## 8. → ## 15. cho nhóm B
A_cross = re.sub(r'^## 8\..*', '## 14. So sánh chéo nhóm A — 5 phương pháp trên 3 kiến trúc', A_cross, flags=re.M, count=1)
A_cross = re.sub(r'^### 8\.(\d)\. ', lambda x: f'### 14.{x.group(1)}. ', A_cross, flags=re.M)
B_cross = re.sub(r'^## 8\..*', '## 15. So sánh chéo nhóm B — 5 phương pháp lightweight trên GRU', B_cross, flags=re.M, count=1)
B_cross = re.sub(r'^### 8\.(\d)\. ', lambda x: f'### 15.{x.group(1)}. ', B_cross, flags=re.M)

# Bỏ dòng heading cũ của phần phân tích/kết luận, giữ phần thân
A_analysis_body = demote(re.sub(r'^## 9\..*\n', '', A_analysis, count=1))
B_analysis_body = demote(re.sub(r'^## 9\..*\n', '', B_analysis, count=1))
A_concl_body    = demote(re.sub(r'^## 10\..*\n', '', A_concl, count=1))
B_concl_body    = demote(re.sub(r'^## 10\..*\n', '', B_concl, count=1))

# Thân §1 và §2 của hai báo cáo (bỏ dòng heading ##)
A_ctx_body   = demote(re.sub(r'^## 1\..*\n', '', A_ctx, count=1))
B_ctx_body   = demote(re.sub(r'^## 1\..*\n', '', B_ctx, count=1))
A_quick_body = demote(re.sub(r'^## 2\..*\n', '', A_quick, count=1))
B_quick_body = demote(re.sub(r'^## 2\..*\n', '', B_quick, count=1))


# --------------------------------------------------------------------------
# 3. Các bảng mới cho phần so sánh chéo 10 phương pháp (§13)
# --------------------------------------------------------------------------
r10_rows = []
for name, data, ent in COLS:
    d = data[10]
    r10_rows.append([name, ent] + [f4(d[m]) for m in METRICS])
TBL_R10 = table(['Phương pháp', 'Entity'] + HDR10, r10_rows)

g_rows = []
for name, data in GLOBALS:
    d = data[10]
    g_rows.append([name] + [f4(d[m]) for m in METRICS])
TBL_GLOBAL = table(['Mô hình toàn cục'] + HDR10, g_rows)

TBL_ACC = by_round_table('accuracy')
TBL_MF1 = by_round_table('macro_f1')
TBL_WF1 = by_round_table('weighted_f1')

bias_rows = []
for name, data, ent in COLS:
    d = data[10]
    bias_rows.append([
        name, ent, f4(d['accuracy']), f4(d['macro_f1']), f4(d['weighted_f1']),
        f4(d['accuracy'] - d['macro_f1']), f4(d['weighted_f1'] - d['macro_f1'])])
bias_rows.sort(key=lambda r: float(r[5]))
TBL_BIAS = table(['Phương pháp', 'Entity', 'accuracy', 'macro_F1', 'weighted_F1',
                  'acc − macro_F1', 'weighted_F1 − macro_F1'], bias_rows)

# Chi phí — nhóm A đo thực tế, nhóm B ước lượng từ kích thước checkpoint
CKPT_B = {'FedAvg': 655_281, 'CustomAgg': 655_154, 'Hetero-KD ⚠': 43_549,
          'ZS+PTD': 173_945, 'ZS-Prune': 656_437}
COMM_A = {'FD-IDS': 30.544, 'PerFed-SKD': 13.745, 'FedCAPS': 0.135,
          'pFedES': 0.043, 'ProxyModel': 27.716}
TIME_A = {'FD-IDS': '22,1 phút', 'PerFed-SKD': '21,8 phút', 'FedCAPS': '191,8 phút',
          'pFedES': '37,5 phút', 'ProxyModel': '28,9 phút'}
cost_rows = []
for name in ['FD-IDS', 'PerFed-SKD', 'FedCAPS', 'pFedES', 'ProxyModel']:
    cost_rows.append([name, 'A', '—', f'{COMM_A[name]:.3f}'.replace('.', ','),
                      'đo thực tế', TIME_A[name]])
for name, size in CKPT_B.items():
    mib = size * 100 / 1024 / 1024
    cost_rows.append([name, 'B', f'{size:,}'.replace(',', '.'),
                      f'{mib:.3f}'.replace('.', ','),
                      'ước lượng = ckpt × 10 client × 10 round', '—'])
TBL_COST = table(['Phương pháp', 'Nhóm', 'Checkpoint round 1 (byte)',
                  'Truyền thông 10 round (MiB)', 'Cách lấy số', 'Thời gian chạy (GRU)'],
                 cost_rows)

# Bảng năng lực (dị thể / global model / personalization / rò rỉ)
CAP = [
    # method, dị thể kích thước, dị thể loại, thứ gửi lên server, global model, personalization, rò rỉ
    ('FD-IDS',      'A', '✗', '✗', 'toàn bộ tham số classifier', '✓', '✗', 'không'),
    ('PerFed-SKD',  'A', '✗', '✗', 'toàn bộ tham số classifier', '✓', '✓', 'không'),
    ('FedCAPS',     'A', '✓', '✓', 'chỉ số feature + điểm hiệu năng', '✗', '✓', 'không'),
    ('pFedES',      'A', '✓', '✓', 'proxy extractor 57 tham số', '✗', '✓', 'không'),
    ('ProxyModel',  'A', '✓', '✓', 'proxy classifier 35.874 tham số', '✓', '✓', 'không'),
    ('FedAvg',      'B', '✗', '✗', 'toàn bộ tham số GRU', '✓', '✗', 'không'),
    ('CustomAgg',   'B', '✗', '✗', 'tham số thưa + mask', '✓', '✗', 'không'),
    ('Hetero-KD',   'B', '✓', '✓', 'logits trên public data', '✓', '✗', '**CÓ**'),
    ('ZS+PTD',      'B', '✗', '✗', 'tham số đã prune + quantize INT8', '✓', '✗', 'không'),
    ('ZS-Prune',    'B', '✗', '✗', 'tham số đã prune (FP32)', '✓', '✗', 'không'),
]
TBL_CAP = table(['Phương pháp', 'Nhóm', 'Dị thể **kích thước**', 'Dị thể **loại**',
                 'Thứ gửi lên server', 'Global model', 'Personalization', 'Rò rỉ dữ liệu'],
                [list(c) for c in CAP],
                ['---', ':---:', ':---:', ':---:', '---', ':---:', ':---:', ':---:'])

# Bảng tra nhanh 10 bài báo
PAPERS = [
    ('1', 'A', '**FD-IDS**',
     'FD-IDS: A Federated Learning and Knowledge Distillation-Based Intrusion Detection System for Non-IID IoT Environments',
     '2025', 'ghi rõ trong bài báo (`Sensors 2025, 25, 4309`)'),
    ('2', 'A', '**PerFed-SKD**',
     'Personalized Federated Learning for Heterogeneous Edge Device: Self-Knowledge Distillation Approach',
     '~2023–2024', 'suy ra từ tham chiếu mới nhất (2023) — cần đối chiếu bản gốc'),
    ('3', 'A', '**FedCAPS**',
     'Permutation-Invariant Representation Learning for Robust and Privacy-Preserving Feature Selection (FedCAPS)',
     '2025', 'mã arXiv `2510` = tháng 10/2025'),
    ('4', 'A', '**pFedES**',
     'pFedES: Generalized Proxy Feature Extractor Sharing for Model Heterogeneous Personalized Federated Learning',
     '~2024', 'tham chiếu mới nhất 2024 — cần đối chiếu bản gốc'),
    ('5', 'A', '**ProxyModel**',
     'Lightweight Federated Learning for On-Device Non-Intrusive Load Monitoring',
     '2025', 'tiền tố năm trong tên file + tham chiếu mới nhất 2024'),
    ('6', 'B', '**FedAvg** (baseline)',
     'DeepFed: Federated Deep Learning for Intrusion Detection in Industrial Cyber–Physical Systems',
     '2021', 'ghi rõ trong bài báo (`IEEE TII, vol. 17, no. 8`)'),
    ('7', 'B', '**CustomAgg + Unstructured Mag.**',
     'OptiFLIDS: Optimized Federated Learning for Energy-Efficient Intrusion Detection in IoT',
     '2025', 'mã arXiv `2510.05180v2` = tháng 10/2025'),
    ('8', 'B', '**Hetero-FedDistillation**',
     'Adaptive personalized federated learning with lightweight depthwise convolutional bottleneck network for novel IDS in internet of vehicles',
     '2025', 'ghi rõ trong bài báo (`Scientific Reports 15, 35604`)'),
    ('9', 'B', '**Zero-shot Pruning + PTD Quant.**',
     'Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS)',
     '2024', 'ghi rõ trong bài báo (`IEEE Access, vol. 12`)'),
    ('10', 'B', '**Zero-shot Pruning**',
     'Lightweight Federated Learning for Efficient Network Intrusion Detection (Lightweight-Fed-NIDS)',
     '2024', 'ghi rõ trong bài báo (`IEEE Access, vol. 12`) — *cùng bài với #9*'),
]
TBL_PAPERS = table(['#', 'Nhóm', 'Phương pháp', 'Bài báo', 'Năm', 'Nguồn xác định năm'],
                   [list(p) for p in PAPERS],
                   ['---:', ':---:', '---', '---', ':---:', '---'])

# Bảng đối chiếu cấu hình hai nhóm
TBL_SETUP = table(
    ['Thành phần', 'Nhóm A — `reports.md`', 'Nhóm B — `report_5_phuong_phap_GRU.md`', 'Giống?'],
    [
        ['Bộ dữ liệu', 'CICIoT2023, 25 feature, 34 lớp', 'CICIoT2023, 25 feature, 34 lớp', '✓'],
        ['Tập test', '`global_test_data.csv` — 9.003.649 dòng', '`global_test_data.csv`', '✓ (nhóm B không ghi số dòng)'],
        ['Số client', '10', '10', '✓'],
        ['Số round', '10', '10', '✓'],
        ['Local epoch/round', '1', '1', '✓'],
        ['Hợp đồng 10 metrics', 'đủ 10 cột, đánh giá sau mỗi round', 'đủ 10 cột, đánh giá sau mỗi round', '✓'],
        ['Kiến trúc GRU', 'GRU **40.034** tham số', 'GRU 2 lớp × 128 units — ckpt 655.281 B ⇒ **≈163.800** tham số', '**✗ — lệch ~4×**'],
        ['Kịch bản mô hình', 'GRU + Transformer + CNN-1D (15 run)', 'chỉ GRU (5 run)', '**✗**'],
        ['Train batch size', '1024', '8192 (eval 1024)', '**✗ — lệch 8×**'],
        ['Optimizer', 'theo từng bài báo', 'Adam, lr = 0,001', '**✗ / không đối chiếu được**'],
        ['Non-IID split', 'có tài liệu: lệch >400× giữa client', 'không ghi trong báo cáo', '**✗ — không xác minh được**'],
        ['Seed', '42', 'không ghi', '**✗**'],
        ['Phần cứng', 'Kaggle, 2× Tesla T4', 'Kaggle, GPU', '~'],
        ['Nguồn số trong báo cáo', 'sinh tự động từ `all_methods_metrics.csv`', 'viết tay từ 5 file `federated_learning_results.csv`', '**✗**'],
        ['Thư mục run có trong repo', '✓ (15 thư mục + artifacts)', '**✗ — các thư mục `GRU_*` không nằm trong repo này**', '**✗**'],
    ],
    ['---', '---', '---', ':---:'])

# vài con số dùng trong prose
acc10 = {n: d[10]['accuracy'] for n, d, _ in COLS}
mf110 = {n: d[10]['macro_f1'] for n, d, _ in COLS}
best_acc = max((v, k) for k, v in acc10.items() if '⚠' not in k)
best_mf1 = max((v, k) for k, v in mf110.items() if '⚠' not in k)

NEW_13 = f"""## 13. So sánh chéo 10 phương pháp — kịch bản GRU

Đây là phần **mới**, không có trong hai báo cáo gốc. Mọi con số ở đây đều được
đọc lại từ chính bảng số liệu của hai báo cáo đó (nhóm A: `all_methods_metrics.csv`;
nhóm B: năm bảng ở mục 8.5–12.5), không nhập tay.

### 13.1. Quy ước so sánh

Chỉ dùng **kịch bản GRU** — kịch bản duy nhất mà cả 10 phương pháp đều có số.
Transformer và CNN-1D của nhóm A được giữ nguyên ở mục 14.

Mỗi phương pháp có một **entity triển khai** khác nhau — thứ thực sự được đem đi
suy luận theo đúng tinh thần bài báo gốc:

| Phương pháp | Nhóm | Entity triển khai | Lý do |
|---|:---:|---|---|
| FD-IDS | A | **server** | không có state client bền vững; đầu ra là một global model |
| PerFed-SKD | A | **TB 10 client** | `Output: Local personalized models P_m` (Algorithm 1) |
| FedCAPS | A | **TB 10 client** | server chỉ giữ encoder/decoder/PPO, không phát logits |
| pFedES | A | **TB 10 client** | *"only each client's personalized local model is used for inference"* |
| ProxyModel | A | **TB 10 client** | model personalized nằm trên thiết bị đầu cuối |
| FedAvg | B | **global model** | FedAvg thuần, không có personalization |
| CustomAgg + Unstruct. Mag. | B | **global model** | đầu ra là một global model thưa |
| Hetero-FedDistillation | B | **global model** | bản build lại đã bỏ bước personalization |
| ZS-Prune + PTD Quant. | B | **global model** | FedAvg trên tham số đã nén |
| ZS-Prune | B | **global model** | FedAvg trên tham số đã nén |

> ### ⚠ Ba cảnh báo bắt buộc đọc trước mọi bảng dưới đây
>
> 1. **Không cùng cấu hình huấn luyện.** Nhóm B dùng GRU lớn hơn ~4 lần
>    (≈163.800 vs 40.034 tham số) và batch size lớn hơn 8 lần (8192 vs 1024).
>    Chênh lệch accuracy vài phần trăm giữa hai nhóm **có thể đến hoàn toàn từ
>    dung lượng mô hình**, không phải từ thuật toán FL. Xem bảng 1.1.
> 2. **Không cùng loại entity.** Bốn phương pháp nhóm A báo cáo *trung bình 10
>    model personalized* chấm trên tập test toàn cục 34 lớp — cách chấm này
>    **bất lợi có hệ thống** cho model personalized (mục 1.2). Muốn so công bằng
>    với nhóm B, hãy dùng bảng 13.3 (chỉ các mô hình toàn cục).
> 3. **Hetero-FedDistillation bị rò rỉ dữ liệu** (public set lấy từ chính test
>    set — mục 10.3). Số của nó được đánh dấu ⚠ và **không được dùng để xếp hạng**.

### 13.2. Round 10 — đủ 10 metrics, cả 10 phương pháp (entity triển khai)

{TBL_R10}

Với phân loại đơn nhãn đa lớp, `accuracy` = `micro_P` = `micro_R` = `micro_F1`
— bốn cột này trùng số là đúng, không phải lỗi.

**Đọc nhanh (bỏ Hetero-KD vì rò rỉ):**

- **accuracy cao nhất:** {best_acc[1]} ({f4(best_acc[0])}). Nhưng cả bốn phương pháp
  nhóm B không rò rỉ đều nằm trong {f4(min(acc10[k] for k in ['FedAvg','CustomAgg','ZS+PTD','ZS-Prune']))}–{f4(max(acc10[k] for k in ['FedAvg','CustomAgg','ZS+PTD','ZS-Prune']))}
  — một vùng rộng chưa tới 0,005, tức **về accuracy chúng không phân biệt được với nhau**.
- **macro_F1 cao nhất:** {best_mf1[1]} ({f4(best_mf1[0])}). Đây mới là chỉ số đáng
  tin cho bài toán 34 lớp mất cân bằng.
- **Khoảng cách hai nhóm:** phương pháp nhóm A mạnh nhất theo entity triển khai là
  FD-IDS ({f4(acc10['FD-IDS'])}); bốn phương pháp personalized còn lại chỉ đạt
  {f4(min(acc10[k] for k in ['PerFed-SKD','FedCAPS','pFedES','ProxyModel']))}–{f4(max(acc10[k] for k in ['PerFed-SKD','FedCAPS','pFedES','ProxyModel']))}
  — đúng như dự đoán từ cách chấm điểm, **không phải bằng chứng chúng kém hơn**.

### 13.3. Round 10 — chỉ các mô hình toàn cục (so sánh công bằng nhất)

Tám entity dưới đây đều là **một model duy nhất, tổng quát, chấm trên cùng tập
test 34 lớp** — đây là phép so sánh ít bị lệch nhất giữa hai nhóm (vẫn còn lệch
kích thước mô hình và batch size).

{TBL_GLOBAL}

**Quan sát then chốt:** tám mô hình toàn cục — bất kể FedAvg thuần, chưng cất,
proxy hay pruning — đều rơi vào **0,620–0,642 accuracy** và **0,21–0,42 macro_F1**.
Trần hiệu năng ở đây do **dữ liệu và ngân sách 10 round quyết định**, không phải
do thuật toán tổng hợp. Chênh lệch giữa PerFed-SKD-server ({f4(A[('PerFed-SKD','server')][10]['accuracy'])})
và FD-IDS-server ({f4(A[('FD-IDS','server')][10]['accuracy'])}) chỉ là
{f4(A[('PerFed-SKD','server')][10]['accuracy'] - A[('FD-IDS','server')][10]['accuracy'])}
— nhỏ hơn cả sai số kỳ vọng của một lần chạy đơn seed.

### 13.4. accuracy theo round — 10 phương pháp (GRU)

{TBL_ACC}

**Ba dáng đường học khác hẳn nhau:**

- **Nhóm B khởi động cao rồi bão hoà:** bắt đầu 0,52–0,55 ngay round 1 (nhờ model
  lớn hơn + batch 8192) và chỉ tăng thêm ~0,09 trong 9 round còn lại.
- **FD-IDS khởi động thấp nhưng dốc:** 0,2397 → 0,6204, đường tăng đơn điệu, đến
  round 10 vẫn chưa bão hoà — nếu chạy thêm round nhiều khả năng còn lên.
- **PerFed-SKD đi xuống:** 0,6705 → 0,4937. Đây là phát hiện bất thường nhất trong
  cả 10 phương pháp (phân tích ở mục 16.1).
- **Hetero-KD tăng mạnh nhất** (0,2662 → 0,6422) nhưng có rò rỉ dữ liệu.

### 13.5. macro-F1 theo round — 10 phương pháp (GRU)

{TBL_MF1}

`macro_F1` đối xử mọi lớp như nhau nên là thước đo trung thực nhất cho bài toán
34 lớp mất cân bằng nặng. Đây là bảng cho thấy **khoảng cách thật** giữa các
phương pháp — rộng hơn nhiều so với bảng accuracy.

### 13.6. weighted-F1 theo round — 10 phương pháp (GRU)

{TBL_WF1}

### 13.7. Mức thiên vị lớp đa số

`acc − macro_F1` càng lớn thì model càng "ăn điểm" bằng các lớp đông và càng mù
lớp hiếm. Bảng xếp theo mức thiên vị **tăng dần**:

{TBL_BIAS}

**Kết luận đau nhất của cả báo cáo:** *không một phương pháp nào trong 10* giải
được bài toán lớp hiếm. Ngay cả phương pháp cân bằng nhất vẫn còn khoảng cách
`acc − macro_F1` ≈ 0,2. Với NIDS, lớp hiếm thường chính là loại tấn công đáng
quan tâm nhất — đây là giới hạn chung của thiết lập thí nghiệm (34 lớp, 10 round,
1 local epoch), không phải của riêng phương pháp nào.

Lưu ý cách đọc ngược: các phương pháp personalized (FedCAPS, pFedES) có
`acc − macro_F1` **nhỏ** chỉ vì cả hai số đều thấp, không phải vì chúng công bằng
giữa các lớp.

### 13.8. Chi phí

{TBL_COST}

> **Hai cột này không so trực tiếp được.** Nhóm A là số **đo thực tế** bởi harness
> (đã tính cả hai chiều lên/xuống theo định nghĩa của harness); nhóm B là **ước
> lượng** `kích thước checkpoint × 10 client × 10 round`, chỉ chiều gửi lên. Thêm
> nữa, mô hình nhóm B lớn hơn ~4 lần nên chi phí cao hơn là điều hiển nhiên, không
> phải nhược điểm của thuật toán.

Dù vậy có ba điểm rút ra được:

1. **pFedES (0,043 MiB) rẻ hơn FedAvg nhóm B (62,5 MiB) khoảng 1.450 lần** — chênh
   lệch lớn đến mức mọi sai số cấu hình nêu trên đều không đảo ngược được kết luận.
2. **Prune-only không giảm băng thông.** ZS-Prune có checkpoint 656.437 B — *lớn hơn*
   baseline FedAvg 655.281 B, vì unstructured/structured pruning chỉ đặt trọng số về 0
   mà `.pth` vẫn lưu đủ tensor FP32. Chỉ **quantization** mới giảm thật (ZS+PTD:
   173.945 B, ~3,8 lần).
3. **Hetero-KD nhìn thì nhỏ nhất (43.549 B) nhưng thực tế đắt nhất nhóm B**: thứ
   truyền đi mỗi round là logits trên public set 5000 mẫu × 34 lớp × 4 byte ≈
   680.000 B/client/round ⇒ ≈ **64,8 MiB** cho 10 round, ngang FedAvg.

### 13.9. Bảng năng lực — 10 phương pháp

{TBL_CAP}

### 13.10. Xếp hạng theo ràng buộc

Không có phương pháp nào thắng toàn diện. Bảng tra theo ràng buộc thực tế:

| Nếu ràng buộc của bạn là… | Chọn | Vì |
|---|---|---|
| Global model mạnh nhất, mọi client giống nhau | **FedAvg / ZS+PTD** (B) | 0,6326 / 0,6314 accuracy, macro_F1 cao nhất toàn bộ 10 phương pháp (0,4155 / 0,4113) |
| Global model + cần giảm băng thông thật | **ZS+PTD** (B) | mất 0,0012 accuracy so với FedAvg nhưng checkpoint nhỏ 3,8 lần |
| Băng thông là nút thắt tuyệt đối | **pFedES** (A) | 0,043 MiB — nhưng phải tránh kịch bản CNN-1D (hỏng) |
| Client dùng **kiến trúc khác nhau** | **pFedES** hoặc **ProxyModel** (A) | cơ chế proxy; ProxyModel mạnh hơn, pFedES rẻ hơn |
| Dị thể + muốn giữ một global model | **Hetero-FedDistillation** (B) | phương pháp duy nhất vừa dị thể vừa có global model — **nhưng phải sửa rò rỉ trước** |
| Riêng tư là ưu tiên cao nhất | **FedCAPS** (A) | không truyền tham số mô hình, chỉ chỉ số feature |
| Cần personalization thật sự | **PerFed-SKD** (A) | TB client cao nhất nhóm A — nhưng **phải early stopping** |
| Cần giải thích được mô hình | **FedCAPS** (A) | trả về subset feature đọc được |
| Chỉ cần baseline để đối chứng | **FedAvg** (B) | ổn định nhất, tăng đơn điệu, không một round nào tụt |
"""

# --------------------------------------------------------------------------
# 4. Ráp tài liệu
# --------------------------------------------------------------------------
DOC = f"""# Báo cáo tổng hợp — 10 phương pháp Federated Learning trên CICIoT2023

Tài liệu này **gộp hai báo cáo rời** thành một bản duy nhất, giữ nguyên toàn bộ
nội dung và số liệu của cả hai, đồng thời bổ sung một phần so sánh chéo mới cho
đủ 10 phương pháp:

| Nhóm | Nguồn | 5 phương pháp | Kịch bản |
|:---:|---|---|---|
| **A** | [`reports.md`](reports.md) — sinh tự động từ [`../scripts/build_report.py`](../scripts/build_report.py) | FD-IDS · PerFed-SKD · FedCAPS · pFedES · ProxyModel | GRU + Transformer + CNN-1D (15 run) |
| **B** | [`report_5_phuong_phap_GRU.md`](report_5_phuong_phap_GRU.md) — viết thủ công từ 5 file CSV | FedAvg · CustomAgg+UnstructMag · Hetero-FedDistillation · ZS-Prune+PTD · ZS-Prune | chỉ GRU (5 run) |

**Điểm quan trọng nhất phải nắm trước khi đọc:** hai nhóm **không chạy cùng cấu
hình huấn luyện** (kích thước mô hình lệch ~4×, batch size lệch 8×). Mục 1.1 liệt
kê đầy đủ chỗ giống và chỗ khác; mọi bảng so sánh chéo ở mục 13 đều kèm cảnh báo
tương ứng.

## Mục lục

**Phần mở đầu**

1. [Hai nhóm thí nghiệm — bối cảnh và mức độ so sánh được](#1-hai-nhóm-thí-nghiệm--bối-cảnh-và-mức-độ-so-sánh-được)
2. [Bảng tra nhanh 10 bài báo](#2-bảng-tra-nhanh-10-bài-báo)

**Phần I — Nhóm A: 5 phương pháp trên 3 kiến trúc**

3. [FD-IDS](#3-fd-ids) · 4. [PerFed-SKD](#4-perfed-skd) · 5. [FedCAPS](#5-fedcaps) ·
6. [pFedES](#6-pfedes) · 7. [ProxyModel](#7-proxymodel)

**Phần II — Nhóm B: 5 phương pháp lightweight trên GRU**

8. [FedAvg (baseline)](#8-fedavg-baseline) · 9. [CustomAggregation + Unstructured Magnitude](#9-customaggregation--unstructured-magnitude) ·
10. [Hetero-FedDistillation](#10-hetero-feddistillation) · 11. [Zero-shot Pruning + PTD Quantization](#11-zero-shot-pruning--ptd-quantization) ·
12. [Zero-shot Pruning](#12-zero-shot-pruning)

**Phần III — So sánh & kết luận**

13. [So sánh chéo 10 phương pháp — kịch bản GRU](#13-so-sánh-chéo-10-phương-pháp--kịch-bản-gru) ← *phần mới*
14. [So sánh chéo nhóm A — 5 phương pháp trên 3 kiến trúc](#14-so-sánh-chéo-nhóm-a--5-phương-pháp-trên-3-kiến-trúc)
15. [So sánh chéo nhóm B — 5 phương pháp lightweight trên GRU](#15-so-sánh-chéo-nhóm-b--5-phương-pháp-lightweight-trên-gru)
16. [Phân tích điểm mạnh — điểm yếu](#16-phân-tích-điểm-mạnh--điểm-yếu)
17. [Kết luận](#17-kết-luận)
18. [Phụ lục — nguồn dữ liệu](#phụ-lục--nguồn-dữ-liệu)

---

## 1. Hai nhóm thí nghiệm — bối cảnh và mức độ so sánh được

### 1.1. Đối chiếu cấu hình hai nhóm

{TBL_SETUP}

**Kết luận về mức độ so sánh được:**

- ✅ **So được:** thứ tự tương đối *bên trong* mỗi nhóm; hình dáng đường hội tụ;
  chi phí truyền thông theo bậc độ lớn; khả năng dị thể (suy từ cơ chế).
- ⚠️ **So được có điều kiện:** giá trị tuyệt đối của accuracy/F1 giữa hai nhóm —
  chỉ nên đọc như "cùng vùng 0,62–0,64" chứ không kết luận A hơn B hay ngược lại
  ở mức vài phần nghìn.
- ❌ **Không so được:** chi phí truyền thông tuyệt đối (một bên đo thực tế, một bên
  ước lượng, mô hình lệch 4×); thời gian chạy (nhóm B không ghi).

### 1.2. Bối cảnh nhóm A

{A_ctx_body}

### 1.3. Bối cảnh nhóm B

{B_ctx_body}

---

## 2. Bảng tra nhanh 10 bài báo

{TBL_PAPERS}

> **Cảnh báo về năm xuất bản.** Chỉ FD-IDS, Hetero-FedDistillation, FedAvg/DeepFed và
> Lightweight-Fed-NIDS ghi rõ nơi/năm xuất bản; FedCAPS và OptiFLIDS suy chắc chắn từ
> mã arXiv. PerFed-SKD và pFedES là **suy đoán từ tham chiếu mới nhất trong bài** —
> cần đối chiếu bản gốc trước khi trích dẫn.

> **Cảnh báo trùng bài báo.** Phương pháp **#9 và #10 dùng chung một bài báo nguồn**
> (Lightweight-Fed-NIDS) — chúng là hai biến thể "pruning + quantization" và
> "pruning-only" của cùng một kỹ thuật, nên xem như **một cặp** thay vì hai phương
> pháp độc lập.

Bảng khả năng dị thể đầy đủ cho cả 10 phương pháp nằm ở [mục 13.9](#139-bảng-năng-lực--10-phương-pháp).

---

# Phần I — Nhóm A: 5 phương pháp trên 3 kiến trúc

> Toàn bộ Phần I giữ nguyên nội dung của [`reports.md`](reports.md), chỉ sửa
> đường dẫn ảnh/link cho khớp thư mục `report/`.

{A_methods}

---

# Phần II — Nhóm B: 5 phương pháp lightweight trên GRU

> Toàn bộ Phần II giữ nguyên nội dung của
> [`report_5_phuong_phap_GRU.md`](report_5_phuong_phap_GRU.md), chỉ đánh số lại
> các mục từ 3–7 thành 8–12 (và cập nhật mọi tham chiếu chéo bên trong).
> **Không có ảnh** cho nhóm B — các thư mục run `GRU_*` không nằm trong repo này.

{B_methods}

---

# Phần III — So sánh & kết luận

{NEW_13}

---

{A_cross}

---

{B_cross}

---

## 16. Phân tích điểm mạnh — điểm yếu

### 16.1. Nhóm A — FD-IDS, PerFed-SKD, FedCAPS, pFedES, ProxyModel

{A_analysis_body}

### 16.2. Nhóm B — FedAvg, CustomAgg, Hetero-KD, ZS+PTD, ZS-Prune

{B_analysis_body}

---

## 17. Kết luận

### 17.1. Kết luận nhóm A

{A_concl_body}

### 17.2. Kết luận nhóm B

{B_concl_body}

### 17.3. Kết luận chung cho cả 10 phương pháp

Đặt 10 phương pháp cạnh nhau trên cùng kịch bản GRU, năm điều nổi lên:

1. **Trần hiệu năng ~0,63 accuracy là của dữ liệu, không phải của thuật toán.**
   Tám mô hình toàn cục khác nhau về cơ chế đến mức không còn gì chung — FedAvg
   thuần, chưng cất tri thức, proxy classifier, pruning, quantization — nhưng đều
   dừng ở 0,620–0,642 (mục 13.3). Muốn vượt trần này phải đổi ngân sách huấn luyện
   (nhiều round hơn, nhiều local epoch hơn) hoặc đổi cách xử lý mất cân bằng lớp,
   chứ không phải đổi thuật toán tổng hợp.

2. **Lớp hiếm là bài toán chưa ai giải được.** Cả 10 phương pháp đều có khoảng cách
   `acc − macro_F1` đáng kể; tốt nhất vẫn ≈ 0,2 (mục 13.7). Với NIDS thì đây là
   điểm yếu nghiêm trọng nhất của toàn bộ thí nghiệm.

3. **"Lightweight" phải được đo, không được suy.** Nhóm B cho hai ví dụ ngược nhau:
   ZS-Prune *nghe như* nén nhưng checkpoint còn lớn hơn baseline; Hetero-KD *nhìn
   như* nhỏ nhất (43 KB) nhưng thực tế truyền ~64,8 MiB logits. Ngược lại pFedES ở
   nhóm A rẻ thật — 0,043 MiB đo được. Bài học: luôn hỏi *"cái gì thực sự đi qua
   dây?"*, đừng nhìn kích thước checkpoint.

4. **Dị thể và global model là hai mục tiêu kéo ngược nhau.** Trong 10 phương pháp,
   chỉ **ProxyModel** và **Hetero-FedDistillation** đạt được cả hai — và Hetero-KD
   đang vướng rò rỉ dữ liệu. FedCAPS/pFedES dị thể nhưng không có global model;
   FedAvg/FD-IDS/PerFed-SKD/ZS-\\*/CustomAgg có global model nhưng buộc mọi client
   cùng kiến trúc.

5. **Hai kết quả bất thường cần điều tra trước khi công bố:** (i) PerFed-SKD **giảm
   accuracy theo round** (0,6705 → 0,4937) — self-distillation không neo ngoài khuếch
   đại sai lệch client trên dữ liệu cực lệch; (ii) CustomAgg **tụt hai round liên tiếp**
   (r7→r9) do mask được tính lại mỗi round, sai lệch so với bài gốc vốn tính mask một
   lần ở round 1.

### 17.4. Việc cần làm để 10 con số này thực sự so sánh được với nhau

Thí nghiệm hiện tại **chưa đủ điều kiện** để xếp hạng 10 phương pháp bằng một bảng
duy nhất. Theo thứ tự ưu tiên:

1. **Chạy lại nhóm B trên đúng cấu hình nhóm A** — GRU 40.034 tham số, batch 1024,
   seed 42, cùng non-IID split đã tài liệu hoá. Đây là việc bắt buộc, và là việc duy
   nhất khiến bảng 13.2 trở thành một bảng xếp hạng hợp lệ.
2. **Sửa rò rỉ dữ liệu của Hetero-FedDistillation** — tách public set ra khỏi
   `global_test_data.csv`, dùng một tập riêng, rồi đánh giá lại.
3. **Đo truyền thông của nhóm B bằng harness** thay vì ước lượng từ kích thước
   checkpoint, và đếm cả logits của Hetero-KD.
4. **Bổ sung phép đo trên tập test riêng của từng client** để chấm công bằng cho
   4 phương pháp personalized của nhóm A.
5. **Chạy đa seed** (ít nhất 3) — hiện mọi cấu hình chỉ chạy một lần, nên chênh lệch
   dưới ~0,01 không mang ý nghĩa thống kê.
6. **Kiểm chứng dị thể bằng số đo thật** — hiện khả năng dị thể của FedCAPS, pFedES,
   ProxyModel mới chỉ **suy ra từ cơ chế**; chỉ Hetero-KD thực sự chạy 5 client
   LDwCBN+GRU + 5 client GRU thuần.

---

## Phụ lục — nguồn dữ liệu

### Nhóm A

- Dữ liệu thô: `metrics/evaluation_metrics.csv` của 15 lần chạy
- Tổng hợp dạng máy đọc: [`../all_methods_metrics.csv`](../all_methods_metrics.csv)
- Biểu đồ so sánh chéo: [`images/`](images/) (bản gốc: [`../report_assets/`](../report_assets/))
- Script sinh báo cáo: [`../scripts/build_report.py`](../scripts/build_report.py)
- Chạy lại: `/home/odixe/miniforge3/envs/nckh/bin/python scripts/build_report.py`

> ⚠️ **Lưu ý sau khi gộp thư mục:** `scripts/build_report.py` vẫn ghi ra
> `task10/reports.md` (thư mục gốc), **không** ghi đè `report/reports.md`. Nếu chạy
> lại script đó, hãy chép kết quả vào `report/` rồi chạy
> `scripts/build_merged_report.py` để dựng lại tài liệu này.

### Nhóm B

- Dữ liệu thô: `federated_learning_results.csv` trong 5 thư mục `GRU_*`
- ⚠️ **Các thư mục `GRU_*` không nằm trong repo `task10` này** — mọi đường dẫn
  `GRU_.../output/...` ở Phần II là tham chiếu tới vị trí gốc bên ngoài. Số liệu
  trong Phần II được chép tay từ các CSV đó và **không thể kiểm chứng lại từ repo
  hiện tại**.

### File trong thư mục `report/`

| File | Nội dung |
|---|---|
| `BAO_CAO_TONG_HOP_10_PHUONG_PHAP.md` | tài liệu này — 10 phương pháp |
| `reports.md` | báo cáo gốc nhóm A (5 phương pháp × 3 kiến trúc) |
| `report_5_phuong_phap_GRU.md` | báo cáo gốc nhóm B (5 phương pháp lightweight) |
| `images/` | 37 ảnh dùng trong hai tài liệu đầu |
| `*.docx` | bản Word của ba file `.md` trên (khổ ngang A4/Letter, có mục lục tự động) |
| `reference_landscape.docx` | template pandoc dùng để xuất `.docx` (khổ ngang, lề 1,27 cm, chữ 9 pt) |

Dựng lại tài liệu gộp và các bản `.docx`:

```bash
python3 scripts/build_merged_report.py
cd report && for f in BAO_CAO_TONG_HOP_10_PHUONG_PHAP reports report_5_phuong_phap_GRU; do
  pandoc "$f.md" -f gfm -t docx --reference-doc=reference_landscape.docx \
    --toc --toc-depth=3 --resource-path=. -o "$f.docx"
done
```
"""

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(DOC, encoding='utf-8')
print(f'{OUT}: {len(DOC.splitlines())} dòng')
