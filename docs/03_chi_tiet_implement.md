# Báo cáo chi tiết: Tái hiện "Heterogeneous Graph Auto-Encoder for Credit Card Fraud Detection"

> Tài liệu kỹ thuật đầy đủ phục vụ tổng hợp báo cáo nhóm.
> Paper: M. T. Singh, R. K. Prasad, et al., *Heterogeneous Graph Auto-Encoder for Credit
> Card Fraud Detection*, arXiv:2410.08121v1 (2024).
> Mã nguồn: thư mục `src/`, `main.py`, `app.py`. Ngôn ngữ: Python 3.14, PyTorch 2.12 (CPU),
> PyTorch Geometric 2.8.

---

## 1. Bài toán & ý tưởng

Phát hiện gian lận thẻ tín dụng trên dữ liệu **mất cân bằng nghiêm trọng** (gian lận
< 0.6%). Thay vì học có giám sát (dễ thiên về lớp đa số), paper đề xuất:

1. Biểu diễn dữ liệu giao dịch dạng **đồ thị không đồng nhất** (heterogeneous graph) gồm
   nhiều loại thực thể (khách hàng, merchant, giao dịch).
2. Mã hoá đồ thị bằng **GNN có cơ chế attention** (Heterogeneous Graph Transformer).
3. Gắn **autoencoder biến phân (VAE)**: train **chỉ trên giao dịch hợp lệ**, học tái tạo
   chúng. Giao dịch gian lận lệch khỏi phân phối hợp lệ ⇒ **lỗi tái tạo cao** ⇒ gắn cờ.
   Cách này xử lý mất cân bằng **không cần** over/under-sampling.

---

## 2. Dữ liệu

- Nguồn: Kaggle `kartik2112/fraud-detection` (bộ giả lập Sparkov), đặt tại `archive/`.
- `fraudTrain.csv`: **1,296,675** giao dịch, **7,506** gian lận (**0.579%**).
- `fraudTest.csv`: **555,719** giao dịch, **2,145** gian lận (**0.386%**).
- 23 cột gốc: `trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender,
  street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time,
  merch_lat, merch_long, is_fraud`.
- **Lấy mẫu**: vì train trên CPU, mặc định dùng **120,000** giao dịch train (giữ **toàn bộ**
  fraud + lấy ngẫu nhiên genuine), đánh giá trên **toàn bộ** 555k test. (`src/config.py`:
  `train_subsample=120_000`, `test_subsample=None`, `keep_all_fraud=True`.)

### Phân tích đặc trưng phân biệt (EDA — quyết định thiết kế quan trọng)

| Đặc trưng | Genuine | Fraud | Độ tách (|Δμ|/σ) |
|---|---|---|---|
| **log_amt** | median $46.7 | median **$396.5** | **1.61σ** |
| **hour** (sin/cos) | ban ngày | ban đêm | **~1.04σ** |
| category (shopping_net/grocery_pos/misc_net) | — | fraud-rate 20–24% | 0.4–0.6σ |
| age | — | — | 0.16σ |
| distance, city_pop | — | — | ~0.01σ (vô dụng) |

→ Tín hiệu gian lận tập trung ở **số tiền và thời gian**. Đây là cơ sở để chọn mục tiêu tái tạo (mục 5).

---

## 3. Tiền xử lý & feature engineering

File: [src/data_prep.py](../src/data_prep.py). Từ ~10 cột thô → **23 đặc trưng** cho mỗi giao dịch:

| Nhóm | Đặc trưng (số chiều) | Cách tính | Chuẩn hoá |
|---|---|---|---|
| Liên tục (4) | `log_amt`, `age`, `log_city_pop`, `log_distance` | `log1p(amt)`; tuổi = năm giao dịch − năm sinh; `log1p(city_pop)`; khoảng cách **haversine** giữa nhà KH và merchant | StandardScaler (fit trên genuine) |
| Chu kỳ (4) | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` | mã hoá sin/cos của giờ (24) và thứ trong tuần (7) | không (đã ∈ [−1,1]) |
| Category (14) | `cat=...` one-hot | one-hot theo từ điển category học từ train | không |
| Giới tính (1) | `gender_M` | M=1, F=0 | không |

Thuộc tính quan trọng: **mọi đặc trưng tính được từ MỘT bản ghi đơn lẻ** → cùng một bộ
featurizer được dùng lại y hệt lúc inference (demo). Hàm `Featurizer.transform_one(record)`.

---

## 4. Dựng đồ thị không đồng nhất

File: [src/data_prep.py](../src/data_prep.py) (`build_hetero_data`). Dùng `torch_geometric.data.HeteroData`.

- **Node**:
  - `transaction`: vector 23 đặc trưng nói trên.
  - `customer` (gom theo `cc_num`), `merchant` (gom theo `merchant`): đặc trưng = **trung bình
    đặc trưng các giao dịch** thuộc thực thể đó (cùng không gian 23 chiều).
- **Cạnh** (hai chiều để message passing chảy về node transaction):
  - `(customer) —makes→ (transaction)` và cạnh ngược `rev_makes`
  - `(merchant) —sells→ (transaction)` và cạnh ngược `rev_sells`

> Vì sao node customer/merchant = trung bình giao dịch? (1) cùng số chiều, **không cần scaler
> riêng**; (2) **inductive** (suy diễn được cho đồ thị mới); (3) **cold-start** tự nhiên: thực
> thể mới chỉ có 1 giao dịch ⇒ hồ sơ = chính giao dịch đó. Paper không nêu chi tiết này.

---

## 5. Kiến trúc model

File: [src/model.py](../src/model.py), lớp `HeteroGraphAutoEncoder`.

### 5.1 Encoder (Eq. 1–5 của paper)
```
x_dict (mỗi loại node) ──Linear theo loại──▶ ẩn 64 chiều
       └─▶ 2 × HGTConv(hidden=64, heads=8)  ─ attention đa đầu theo loại cạnh ─▶ ReLU + Dropout(0.1)
       └─▶ lấy embedding node 'transaction'  ─▶ fc_mu (→12),  fc_logvar (→12)
```
- `HGTConv` của PyG **chính là** Heterogeneous Graph Transformer (Hu et al. 2020 = reference
  [41] paper dựa vào), hiện thực đúng attention theo từng loại cạnh (Eq. 3–5).

### 5.2 Tầng biến phân (VAE — Eq. cuối mục 4.1)
```
z = μ + ε · exp(½ · logσ²),   ε ~ N(0, I)     (lúc train)
z = μ                                          (lúc eval → tái tạo ổn định)
```

### 5.3 Decoder (Eq. 6) & mục tiêu tái tạo
```
z(12) ─▶ Linear 64 ─ReLU─Dropout─▶ Linear 64 ─ReLU─Dropout─▶ Linear 3
```
- **CHỈ tái tạo 3 đặc trưng**: `log_amt`, `hour_sin`, `hour_cos` (`RECON_COLS`). 20 đặc trưng còn
  lại chỉ là **ngữ cảnh đầu vào** (xem mục 9 — lý do).

### 5.4 Hàm mất mát
```
L = MSE(recon, target_amt_hour) + β · KL(μ, logσ²),   β = 5e-4
```

### 5.5 Điểm bất thường (anomaly score)
```
score(giao dịch) = trung bình theo 3 đặc trưng của (recon − target)²
```
Lỗi ≥ ngưỡng ⇒ **FRAUD**.

---

## 6. Huấn luyện

File: [src/train.py](../src/train.py). Quy trình **train một lần** rồi lưu artifacts.

1. Tách train: `genuine` chia thành train (85%) + validation (15%); **toàn bộ fraud của tập
   train** dồn vào validation (để chọn ngưỡng & theo dõi AUC-PR).
2. Fit `Featurizer` **chỉ trên genuine** (tránh rò rỉ).
3. Dựng 3 đồ thị: train (chỉ genuine), val, test.
4. Train full-batch bằng Adam (`lr=2e-3`, `weight_decay=1e-5`), tối đa 150 epoch,
   **early stopping** theo val AUC-PR (patience 25). Chọn checkpoint tốt nhất.
5. Tìm **ngưỡng** tối đa F1 trên validation (đường Precision–Recall).
6. Đánh giá trên test (mục 8) và lưu mọi thứ vào `outputs/artifacts.pt`.

### Siêu tham số (`src/config.py`)
| Tham số | Giá trị | Ghi chú so với paper |
|---|---|---|
| hidden_dim | 64 | = paper (Table 4) |
| heads (attention) | 8 | paper ghi 16; giảm để nhanh/ổn trên CPU |
| encoder_layers | 2 | paper ghi "124" (bất hợp lý) |
| latent_dim (bottleneck) | 12 | paper không nêu; **bắt buộc < 23** |
| decoder_hidden | 64 | paper ghi "decoder = 64" |
| dropout | 0.1 | paper ghi 0.4 (gây underfit) |
| weight_decay | 1e-5 | paper ghi 0.01 (over-regularize) |
| β (KL) | 5e-4 | paper không nêu |
| epochs / patience | 150 / 25 | — |

---

## 7. Inference & Demo (tách rời train)

File: [src/scorer.py](../src/scorer.py) (`FraudScorer`), [app.py](../app.py).

- `FraudScorer.load()` nạp lại model + featurizer + ngưỡng + **kho hồ sơ** customer/merchant.
- `score_transaction(record)`: dựng **đồ thị 3 node** (giao dịch + customer + merchant của nó),
  chạy encoder→decoder, tính lỗi tái tạo → verdict. Thực thể đã biết dùng hồ sơ lịch sử; thực
  thể mới → cold-start. **Chạy tức thì trên CPU, không train lại.**
- Trả về: `reconstruction_error`, `threshold`, `ratio = error/threshold`, `verdict`,
  cờ known/new, và **top features** (xếp theo lỗi tái tạo từng đặc trưng) để giải thích.

---

## 8. Đánh giá & kết quả

File: [src/evaluate.py](../src/evaluate.py). Chỉ số: Precision, Recall, F1, ROC-AUC, AUC-PR
(mục 5.1 paper). Báo cáo theo **hai phân phối**:

- **Full test** (giữ tỉ lệ thật 0.39% fraud) — trung thực với thực tế; ROC-AUC, AUC-PR là
  threshold-free.
- **Balanced test** (toàn bộ fraud + số genuine bằng nhau) — F1/Precision/Recall dễ diễn giải
  và **so sánh được với paper**.

| Chỉ số | Model nhóm | Paper (Table 5) |
|---|---|---|
| ROC-AUC (full test) | **0.835** | 0.85 |
| AUC-PR (full test, 0.39%) | 0.113 | — |
| **F1 (balanced)** | **0.796** | 0.81 |
| **AUC-PR (balanced)** | **0.858** | 0.89 |
| Precision / Recall (balanced) | 0.86 / 0.74 | 0.50 / 0.99 |

Biểu đồ sinh ra ở `outputs/plots/` (giống Figure 5 của paper): đường loss train/val, phân
phối lỗi tái tạo theo lớp, F1 theo ngưỡng, ROC, Precision–Recall.

**Baseline tham chiếu** (chỉ dùng đặc trưng thô, để định vị model):
chỉ `amt` → ROC-AUC 0.833; `amt + chỉ báo ban đêm` → ROC-AUC 0.882. Model AE (0.835) ngang
mức tín hiệu `amt` — tức đã khai thác đúng tín hiệu chính.

---

## 9. Các quyết định thiết kế then chốt (và lý do)

1. **Chỉ tái tạo `amt` + `hour`**: ban đầu tái tạo cả 23 đặc trưng → ROC-AUC ~0.74, AUC-PR ~0.02
   (gần ngẫu nhiên) vì 14 one-hot category **không tái tạo được** (AE chỉ học phân phối chung →
   lỗi gần như hằng số cho cả 2 lớp) tạo "sàn nhiễu" nuốt tín hiệu amt. Thu hẹp mục tiêu tái tạo
   về amt+giờ → ROC-AUC 0.835, AUC-PR 0.86 (balanced).
2. **Bottleneck (latent 12 < 23)**: latent ≥ số đặc trưng ⇒ AE học hàm đồng nhất, lỗi không tách
   được lớp (đã quan sát: latent 64 → AUC-PR 0.012).
3. **Không chuẩn hoá lỗi theo phương sai genuine**: thử chia lỗi từng đặc trưng cho phương sai
   genuine → **giảm** AUC-PR (vì amt vốn dao động mạnh ở genuine nên bị giảm trọng số). ⇒ giữ lỗi
   tái tạo **thuần** (đúng như paper).
4. **Giảm dropout/weight_decay**: giá trị gốc của paper gây underfit (lỗi genuine kẹt ~0.33).

---

## 10. Cấu trúc mã nguồn

```
SeminarProject/
├─ archive/                fraudTrain.csv, fraudTest.csv (dữ liệu Kaggle thật)
├─ src/
│  ├─ config.py            siêu tham số tập trung
│  ├─ data_prep.py         feature engineering + dựng HeteroData + kho hồ sơ demo
│  ├─ model.py             HGT encoder + VAE + MLP decoder + hàm tính lỗi
│  ├─ train.py             pipeline train-một-lần, lưu artifacts, đánh giá
│  ├─ evaluate.py          tìm ngưỡng, chỉ số, biểu đồ (Figure 5)
│  ├─ scorer.py            FraudScorer: chấm điểm 1 giao dịch (lõi demo)
│  └─ utils.py             seed, chọn device
├─ make_sample_data.py     sinh dữ liệu giả lập (chạy thử không cần Kaggle)
├─ main.py                 `python main.py` → train
├─ app.py                  `streamlit run app.py` → demo
└─ outputs/                artifacts.pt, metrics.json, plots/
```

---

## 11. Tái lập (reproducibility)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python main.py            # train ~3 phút trên CPU (early-stop ~epoch 30), eval full 555k test
streamlit run app.py      # mở demo
```
- Seed cố định (42). Thiết bị tự chọn CUDA nếu có, mặc định CPU.
- Kết quả số học ở `outputs/metrics.json`; biểu đồ ở `outputs/plots/`.

---

## 12. Hạn chế & hướng phát triển

- Lỗi tái tạo amt+giờ ≈ baseline amt thuần (ROC-AUC ~0.83); chưa tận dụng triệt để tín hiệu
  "ban đêm" (trần ~0.88) do đặc trưng chu kỳ khó tái tạo bằng MSE.
- Chưa mô hình hoá **quan hệ thời gian** giữa các giao dịch của một thẻ (paper cũng nêu là
  hướng tương lai).
- Đề xuất: thêm đặc trưng hành vi theo thời gian; thử attention/HGT sâu hơn với dữ liệu đầy đủ
  trên GPU; học trọng số đặc trưng cho điểm bất thường.
