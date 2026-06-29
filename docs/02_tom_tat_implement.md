# Tóm tắt việc implement lại model

> Paper gốc: M. T. Singh et al., *Heterogeneous Graph Auto-Encoder for Credit Card Fraud
> Detection*, arXiv:2410.08121 (2024). Dataset: Kaggle `kartik2112/fraud-detection` (Sparkov).

## 1. Ý tưởng cốt lõi tái sử dụng từ paper

| Ý tưởng / kỹ thuật | Mô tả | Vị trí trong code |
|---|---|---|
| **Đồ thị không đồng nhất** (heterogeneous graph) | 3 loại node: `customer` (cc_num), `merchant`, `transaction` (trans_num); cạnh customer↔transaction↔merchant | [data_prep.py](../src/data_prep.py) |
| **GNN + attention** | Encoder dùng **Heterogeneous Graph Transformer (HGT)** — chính là reference [41] mà paper dựa vào; attention đa đầu theo từng loại cạnh (Eq. 2–5) | [model.py](../src/model.py) |
| **Autoencoder phát hiện bất thường** | Train **chỉ trên giao dịch hợp lệ**, lỗi tái tạo lớn ⇒ fraud → giải quyết mất cân bằng lớp **không cần** over/under-sampling | [model.py](../src/model.py), [train.py](../src/train.py) |
| **Biến phân (VAE)** | Tham số hoá lại (reparameterization) `z = μ + ε·exp(½·logσ²)` (Eq. cuối mục 4.1, reference [43]) | [model.py](../src/model.py) |
| **Tìm ngưỡng tối ưu** | Quét ngưỡng để tối đa F1 (Algorithm 1 + Fig 5c của paper) | [evaluate.py](../src/evaluate.py) |
| **Bộ chỉ số & biểu đồ** | Precision, Recall, F1, ROC-AUC, AUC-PR + các biểu đồ giống Figure 5 | [evaluate.py](../src/evaluate.py) |

## 2. Những điểm bổ sung / cải tiến của nhóm

| Bổ sung | Lý do |
|---|---|
| **Chỉ tái tạo `amt` + `hour`** (thay vì toàn bộ đặc trưng) | EDA cho thấy chỉ số tiền (tách 1.6σ) và giờ (1.0σ) mang tín hiệu gian lận; tái tạo cả category/giới tính/vị trí chỉ tạo "sàn nhiễu" làm **loãng** tín hiệu. Đây là cải tiến quan trọng nhất giúp model hoạt động. |
| **Bottleneck thật** (latent = 12 < 23 đặc trưng) | Nếu latent ≥ số đặc trưng, AE học được hàm đồng nhất (copy input) ⇒ lỗi không tách được fraud. Bottleneck là bắt buộc. |
| **Giảm regularization** (dropout 0.4→0.1, weight_decay 0.01→1e-5) | Tham số gốc của paper gây **underfit** nặng (không tái tạo nổi cả giao dịch hợp lệ). |
| **Đặc trưng node customer/merchant = trung bình đặc trưng giao dịch của họ** | Giúp **inductive** + xử lý **cold-start** cho thực thể mới một cách tự nhiên. |
| **Tách train ↔ inference**: train 1 lần → lưu artifacts → chấm điểm tức thì 1 giao dịch | Phục vụ **demo ứng dụng**: tạo giao dịch / khách / merchant mới và bắt gian lận ngay, không train lại. |
| **Demo Streamlit + giải thích đặc trưng** | Trực quan cho buổi seminar; chỉ rõ feature nào gây bất thường. |
| **Đánh giá kép: full-imbalanced + balanced** | Báo cáo trung thực ở phân phối thật **và** con số dễ diễn giải/so sánh với paper. |
| **Bộ sinh dữ liệu giả lập** (`make_sample_data.py`) | Chạy thử toàn bộ pipeline khi chưa có dữ liệu Kaggle. |

## 3. Những chỗ paper KHÔNG nói rõ → nhóm phải tự tái dựng

Paper mô tả công thức ở mức ý tưởng nhưng **thiếu nhiều chi tiết kỹ thuật**:

1. **Feature engineering**: paper không liệt kê đặc trưng đầu vào. Nhóm tự thiết kế **23 đặc trưng**
   từ ~10 cột thô (log số tiền, giờ/thứ dạng sin-cos, tuổi, khoảng cách haversine, one-hot category, giới tính…).
2. **Đặc trưng của node customer/merchant**: paper không nói. Nhóm dùng trung bình đặc trưng giao dịch.
3. **Cách dựng cạnh** (loại cạnh, chiều cạnh, cạnh ngược cho message passing): tự thiết kế.
4. **Số liệu Table 4 bất hợp lý**: "Encoder layers = 124" (124 lớp GNN sẽ over-smoothing) và
   "Decoder = 64" → nhóm hiểu là **2 lớp HGT** và **decoder rộng 64**.
5. **Kích thước bottleneck**: paper chỉ ghi hidden=64, không nói bottleneck → nhóm xác định latent=12.
6. **Kiến trúc decoder**: paper chỉ ghi "deep neural network" → nhóm dùng MLP 3 lớp.
7. **Hàm mất mát**: trọng số KL (β) không nêu → nhóm dùng β=5e-4 (gần như AE thuần).
8. **Thuật toán tìm ngưỡng** ("straightforward search") → nhóm quét theo đường PR tối đa F1.
9. **Cách chia train/val để chọn ngưỡng**: tự thiết kế (val = genuine giữ lại + toàn bộ fraud train).
10. **Phân phối khi đánh giá**: con số AUC-PR 0.89 / F1 0.81 của paper ở mức 0.39% fraud gần như
    chỉ đạt được nếu đánh giá trên **tập cân bằng** → nhóm tái hiện và báo cáo cả hai cách.

## 4. Kết quả

| Chỉ số | Model của nhóm | Paper |
|---|---|---|
| ROC-AUC (full test, 0.39% fraud) | **0.835** | 0.85 |
| F1 (balanced test 50/50) | **0.796** | 0.81 |
| AUC-PR (balanced test) | **0.858** | 0.89 |
| Precision / Recall (balanced) | 0.86 / 0.74 | 0.50 / 0.99 |

→ Tái hiện **sát paper** về ROC-AUC và F1. Việc paper đạt số rất cao trên tập 0.39% củng cố
giả thuyết họ đánh giá trên tập cân bằng.

## 5. Hạn chế & hướng phát triển

- Lỗi tái tạo dựa trên amt+giờ ≈ baseline "amt thuần" (ROC-AUC ~0.83); chưa khai thác hết
  tín hiệu "giờ ban đêm" (trần ~0.88) vì đặc trưng chu kỳ khó tái tạo bằng MSE.
- Chưa mô hình hoá **quan hệ thời gian** giữa các giao dịch (paper cũng nêu đây là hạn chế).
- Hướng phát triển: thêm đặc trưng hành vi theo thời gian, thử trọng số đặc trưng học được,
  hoặc dùng GPU để train trên toàn bộ dữ liệu.
