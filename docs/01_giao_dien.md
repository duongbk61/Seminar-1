# Giới thiệu giao diện Demo (Streamlit)

> File: [app.py](../app.py) · Chạy: `.venv/Scripts/streamlit run app.py` rồi mở `http://localhost:8501`

Demo cho phép **nhập một giao dịch và xem model phán đoán Fraud / Non-Fraud ngay lập tức** —
không cần train lại. Model đã được huấn luyện sẵn và lưu ở `outputs/artifacts.pt`; ứng dụng
chỉ nạp lại rồi chấm điểm.

---

## Bố cục tổng quan

```
┌───────────────┬─────────────────────────────────────────────┐
│   SIDEBAR     │   1 · Pick a transaction                    │
│ (thông tin    │   [🎲 genuine][🚨 fraud][🆕 cust][🆕 merch] │
│  model)       │                                             │
│               │   ┌─Customer─┐ ┌─Merchant─┐ ┌─Transaction─┐ │
│ ROC-AUC       │   │ chọn/ mới │ │ chọn/ mới │ │ amount/giờ  │ │
│ AUC-PR        │   └──────────┘ └──────────┘ └─────────────┘ │
│ F1, P/R       │                                             │
│ threshold     │   2 · Score it   [🔎 Check transaction]     │
│               │   → Verdict + điểm số + giải thích           │
└───────────────┴─────────────────────────────────────────────┘
```

---

## 1. Sidebar — "Model performance"

Hiển thị chất lượng model (đọc từ `outputs/artifacts.pt`):

- **Full test (imbalanced ~0.4% fraud)**: `ROC-AUC`, `AUC-PR` — đánh giá trên toàn bộ
  555k giao dịch test với tỉ lệ gian lận thật (rất mất cân bằng).
- **Balanced test (50/50)**: `F1`, `Precision / Recall` — đánh giá trên tập cân bằng
  (toàn bộ fraud + số genuine bằng nhau) để con số dễ diễn giải và so sánh với paper.
- **Decision threshold**: ngưỡng quyết định (lỗi tái tạo ≥ ngưỡng ⇒ Fraud).
- Số lượng customer / merchant đã biết (có hồ sơ lịch sử).
- Cảnh báo nếu model đang chạy trên **dữ liệu giả lập** (chưa có file Kaggle thật).

---

## 2. Khu "1 · Pick a transaction" — Nút nạp nhanh

Bốn nút để tạo nhanh một kịch bản demo:

| Nút | Tác dụng |
|---|---|
| 🎲 **Random genuine** | Nạp một giao dịch hợp lệ thật từ tập test (kỳ vọng ra NON-FRAUD) |
| 🚨 **Random fraud** | Nạp một giao dịch gian lận thật từ tập test (kỳ vọng ra FRAUD) |
| 🆕 **New customer** | Sinh một chủ thẻ hoàn toàn mới, chưa có lịch sử (cold-start) |
| 🆕 **New merchant** | Sinh một merchant hoàn toàn mới, chưa có lịch sử (cold-start) |

→ Đây là cách trình diễn nhanh nhất: bấm *Random fraud* → Check → thấy FRAUD; bấm
*Random genuine* → Check → thấy NON-FRAUD.

---

## 3. Ba panel nhập giao dịch

### 👤 Customer (chủ thẻ)
- Chọn trong danh sách **khách đã biết** (có hồ sơ chi tiêu lịch sử) hoặc **"➕ New customer"**.
- Khách đã biết → model dùng **hồ sơ lịch sử** (chi tiêu trung bình…) làm ngữ cảnh.
- Khách mới → **cold-start**: khởi tạo từ chính giao dịch này (không lịch sử ⇒ vốn đáng ngờ hơn).
- Hiển thị: toạ độ nhà, giới tính, dân số thành phố.

### 🏪 Merchant
- Tương tự: chọn merchant đã biết hoặc tạo mới.
- Hiển thị: category, vị trí merchant.

### 💳 Transaction
- **Amount ($)**: số tiền — yếu tố quan trọng nhất.
- **Category**: ngành hàng.
- **Hour of day**: giờ trong ngày (gian lận hay xảy ra ban đêm).
- **Date**: ngày giao dịch (suy ra thứ trong tuần).

---

## 4. Khu "2 · Score it" — Kết quả

Bấm **🔎 Check transaction**, kết quả gồm:

1. **Verdict lớn**: 🚨 **FRAUD** (nền đỏ) hoặc ✅ **NON-FRAUD** (nền xanh).
2. **Ba chỉ số**:
   - *Reconstruction error*: lỗi tái tạo của giao dịch (điểm bất thường).
   - *Threshold*: ngưỡng quyết định.
   - *Error / Threshold*: tỉ số vượt ngưỡng (>1 = gian lận, càng lớn càng bất thường).
3. **Thanh Anomaly level**: trực quan mức độ bất thường (mốc 1.0 = đúng ngưỡng).
4. **Nhãn ngữ cảnh**: customer/merchant là "đã biết" hay "mới (cold-start)".
5. **Top contributing features**: bảng + biểu đồ cột các đặc trưng gây bất thường nhất
   (thường `log_amt` đứng đầu khi số tiền cao bất thường) — giải thích **vì sao** bị gắn cờ.
6. **Raw record**: JSON bản ghi gốc gửi vào model (để minh bạch).

---

## Kịch bản trình diễn gợi ý (3 phút)

1. 🎲 *Random genuine* → Check → **NON-FRAUD**, error thấp dưới ngưỡng.
2. 🚨 *Random fraud* → Check → **FRAUD**, error cao, `log_amt` đứng đầu bảng giải thích.
3. Lấy lại genuine, **tăng Amount** lên ~$1500 và đổi **Hour** sang 3h sáng → Check →
   chuyển thành **FRAUD** (minh hoạ model phản ứng theo số tiền + giờ).
4. 🆕 *New customer* + *New merchant* với số tiền lớn → Check → **FRAUD** (minh hoạ
   xử lý thực thể mới / cold-start).
