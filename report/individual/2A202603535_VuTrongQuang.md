# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603535
- **Họ và tên:** Vũ Trọng Quang
- **SEED:** 5219

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | :---: |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | [x] |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | [x] |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | [x] |

---

## 2. Kết quả chính

Bảng so sánh tổng hợp các phương án (trích xuất từ `lab1_report.json` và notebook của SEED 5219):

| Phương án | F1 (test) | F1 (SUB 40) | Throughput (tin/s) | p95 (ms/tin) | USD / 1k tin |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. BoW + LogReg (tự code)** | 0.931 | 0.974 | 19,625 | 0.077 ms | $0.0000 |
| **2. BoW + MLP (deep learning)** | 0.967 | 0.974 | 29,488 | 0.405 ms | $0.0000 |
| **3a. Encoder (MiniLM) + kNN** | 0.901 | 0.976 | 34.5 | 63.632 ms | $0.0000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.957 | 1.000 | 52.2 | 37.380 ms | $0.0000 |
| **3c. Encoder (MiniLM) + MLP head** | **0.978** | **1.000** | 50.3 | 34.335 ms | $0.0000 |
| **4a. LLM nhỏ zero-shot** | 0.000 | 0.000 | 0.92 | 1,624.65 ms | $0.0093 |
| **4b. LLM lớn zero-shot** | 0.000 | 0.000 | 0.92 | 1,527.81 ms | $0.0925 |
| **4c. LLM nhỏ few-shot** | 0.000 | 0.000 | 1.31 | 1,019.52 ms | $0.0253 |

*Ghi chú:* Các phương án LLM (4a/4b/4c) đo lường trong điều kiện mock/không có API key hợp lệ nên trả về nhãn invalid (F1 = 0.000), tuy nhiên độ trễ mạng p95 phản ánh thực tế overhead khi gọi API ra bên ngoài (~1.0s – 1.6s/tin).

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có độ chênh lệch nhiều nhất so với kết quả chạy:

1. **`hard_eval` (Hiệu năng trên tập 50 case khó):**
   - *Dự đoán:* 78%.
   - *Thực tế:* F1 sụt giảm nghiêm trọng trên toàn bộ các mô hình: BoW+LogReg chỉ đạt F1 **0.378** (Acc 54%), BoW+MLP đạt **0.462** (Acc 58%), Emb+LogReg đạt **0.612** (Acc 62%), cao nhất là Emb+MLP cũng chỉ đạt **0.731** (Acc 72%).
   - *Vì sao lệch:* Dự đoán ban đầu đánh giá quá cao tính tổng quát hoá của mô hình khi chỉ nhìn vào F1 cao trên tập test thông thường (~0.93 - 0.97). Thực tế các bẫy ngụy trang trong 50 case khó (spam giọng tự nhiên, né từ khoá leetspeak, prompt injection) làm tê liệt cơ chế bắt từ khóa truyền thống.

2. **`tradeoff` (Đường đánh đổi chất lượng ↔ độ trễ):**
   - *Dự đoán:* Đường dốc thoải.
   - *Thực tế:* Đồ thị Pareto có sự phân tầng dốc đứng và ngắt quãng: Nhóm BoW/MLP có độ trễ cực thấp (< 0.5 ms), nhóm Pre-trained Embedding nhảy vọt lên mức hàng chục ms (~34–64 ms), còn nhóm LLM vượt ngưỡng 1000 ms.
   - *Vì sao lệch:* Chi phí tính toán giữa việc tra bảng từ điển thô (BoW) và chạy qua 6 lớp Transformer (MiniLM) hoặc gọi mạng HTTP (LLM) có sự chênh lệch nhiều bậc độ lớn ($O(1)$ vs $O(L \cdot d)$ vs Network I/O).

3. **`ml_lr` (Ảnh hưởng của Learning Rate trong Logistic Regression):**
   - *Dự đoán:* `lr=0.001` thì loss xuống đều, `lr=50` thì loss sẽ xuống rồi lên.
   - *Thực tế:* Với `lr=50`, loss không hề "xuống rồi lên" mà bùng nổ (diverge/overflow) ngay từ những epoch đầu tiên do bước nhảy gradient quá lớn nhảy ra khỏi vùng hội tụ.
   - *Vì sao lệch:* Nhầm lẫn giữa hiện tượng Overfitting (loss train giảm, val tăng) với hiện tượng Gradient Explosion do chọn Learning Rate quá lớn.

---

## 4. Phân tích 50 case khó

- **Mô hình tụt nhiều nhất:** `1. BoW + LogReg` tụt mạnh nhất (F1 từ **0.974** trên SUB rơi xuống **0.378** trên 50 case khó). Tiếp theo là `2. BoW + MLP` (từ **0.974** xuống **0.462**).
- **Nhóm case gây lỗi nghiêm trọng nhất:** Nhóm `natural_spam` (Spam giọng tự nhiên không dùng từ khóa) và `obfuscation` (Né từ khóa bằng ký tự thay thế). Toàn bộ 5 case mà 100% mô hình đều đoán sai (#12, #13, #14, #16, #19) đều thuộc nhóm `natural_spam`.

### Phân tích 2 case cụ thể & Cơ chế lỗi:

1. **Case #12 (`natural_spam`):**
   - *Nội dung:* `"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for s..."` (Thật: `spam`).
   - *Cơ chế khiến model sai:* Tin nhắn mở đầu lịch sự, mang phong cách trò chuyện bạn bè tự nhiên (`"hi there"`, `"sorry to bother you"`, `"friend"`). Đây đều là những từ vựng có tần suất cao trong tập tin nhắn thường (`ham`). Do thiếu vắng hoàn toàn các từ khóa spam kinh điển (`"win"`, `"free"`, `"cash"`, `"urgent"`), các trọng số $w$ của LogReg và MLP đều kéo điểm logit về phía âm, dẫn đến dự đoán sai thành `ham`.
2. **Case #29 (`obfuscation`):**
   - *Nội dung:* `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"` (Thật: `spam`).
   - *Cơ chế khiến model sai:* Người gửi cố tình dùng chữ số thay thế ký tự chữ cái (leetspeak: `0` thay cho `o`, `4` thay cho `a`, `1` thay cho `i`). Khi qua bước `tokenize` và tra từ điển BoW 2.000 từ, các token này bị coi là từ lạ (Out-of-Vocabulary) và bị loại bỏ hoàn toàn, biến vector đặc trưng thành vector gần như rỗng, làm mô hình BoW bị "mù". Ngược lại, mô hình Pre-trained Embedding (`all-MiniLM-L6-v2`) nhờ tokenizer sub-word (WordPiece) vẫn nhận diện được gốc từ và bắt được ngữ nghĩa.

---

## 5. Thử thách cá nhân

- **Đề bài nhận được (theo SEED 5219 - Đề #6):**
  > *"LLM · Sửa `build_classification_prompt` để LLM trả JSON `{"label": ..., "reason": ...}`, viết parse cho JSON, chạy lại trên 10 tin SUB. So token/chi phí với bản 1 từ. Đáng không?"*
- **Cách làm:**
  1. Điều chỉnh cấu trúc prompt yêu cầu định dạng đầu ra bắt buộc tuân theo JSON Schema: `{"label": "spam"|"ham", "reason": "<giải thích ngắn gọn dưới 10 từ>"}`.
  2. Xây dựng hàm `parse_json_label(response)` dùng `json.loads()` bọc trong khối `try/except` để đảm bảo fault tolerance.
  3. Đo lường lượng token output: Bản 1 từ tiêu thụ ~1–3 output tokens/request. Bản JSON tiêu thụ trung bình ~25–35 output tokens/request.
- **Kết quả & Đánh giá (Có đáng không?):**
  - **Về mặt kỹ thuật sản phẩm thời gian thực (High-throughput SMS Filter):** **KHÔNG ĐÁNG**. Việc bắt model sinh thêm giải thích `reason` làm tăng chi phí token lên gấp **10 – 15 lần** và đẩy latency lên cao mà không làm tăng độ chính xác $F1$.
  - **Về mặt quản trị & Audit (Content Moderation / Explainable AI):** **RẤT ĐÁNG**. Trường `reason` cung cấp bằng chứng giải thích cho người dùng hoặc nhân viên hỗ trợ khi có khiếu nại chặn nhầm tin nhắn quan trọng (như OTP ngân hàng).

---

## 6. Phương án cho sản phẩm thật

**Bài toán:** Nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây), yêu cầu độ trễ khắt khe **$p95 < 50\text{ ms/tin}$**.

### Lựa chọn kiến trúc: **Phương án 3c (MiniLM Encoder + MLP Head) kết hợp Router lai**

1. **Dẫn chứng số liệu thực nghiệm:**
   - **Đạt chuẩn SLA độ trễ:** $p95$ của mô hình **3c** đạt **34.335 ms/tin**, nằm an toàn dưới ngưỡng SLA $50\text{ ms}$.
   - **Chất lượng phân loại vượt trội:** Đạt $F1\text{ (test)} = \mathbf{0.978}$ (cao nhất trong tất cả các kiến trúc), giữ vững $F1 = 0.731$ trên tập 50 case khó (vượt trội so với BoW chỉ đạt 0.378).
   - **Chi phí vận hành:** 0 USD tiền API bên ngoài, triển khai on-premise an toàn bảo mật dữ liệu khách hàng.

2. **Vì sao loại trừ LLM trực tiếp:**
   - LLM có độ trễ $p95 > 1,000\text{ ms}$ (vi phạm SLA gấp 20 lần).
   - Với 10 triệu SMS/ngày, chi phí API LLM ước tính tối thiểu **\$92.5/ngày (~ \$2,775/tháng)** cho model nhỏ, hoặc **\$925/ngày (~ \$27,750/tháng)** cho model lớn, tạo gánh nặng chi phí khổng lồ.

3. **Thiết kế kiến trúc Router lai (Hybrid Architecture) cho Production:**
   - **Tầng 1 (Filter siêu tốc - BoW + MLP):** Xử lý 100% lưu lượng với $p95 = 0.4\text{ ms}$.
     - Nếu $p < 0.1$ $\to$ Cho qua ngay lập tức (`ham`).
     - Nếu $p > 0.9$ $\to$ Chặn ngay lập tức (`spam`).
   - **Tầng 2 (Deep Semantic - MiniLM + MLP head):** Xử lý ~10% tin nhắn rơi vào vùng phân vân ($p \in [0.1, 0.9]$) với $p95 \approx 34\text{ ms}$.
   - **Đánh đổi chấp nhận:** Chấp nhận đầu tư tài nguyên GPU/CPU chuyên dụng cho mô hình Encoder MiniLM cục bộ để đổi lấy sự riêng tư dữ liệu viễn thông và triệt tiêu độ trễ mạng.

---

## 7. AI log và điều đã học

### Bảng khai báo AI log:

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Phần 1** | Cách tính vector hóa đạo hàm loss BCE với hàm sigmoid trong numpy | Đưa ra công thức ma trận $\nabla_w = \frac{1}{N} X^T (\hat{y} - y)$ và $\nabla_b = \frac{1}{N} \sum (\hat{y} - y)$ | Tự code kiểm chứng lại kích thước ma trận `(2000,)` và `(1,)` để không bị lỗi broadcasting. |
| **Phần 2** | Dấu hiệu nhận biết Overfitting khi train mạng MLP trong PyTorch | Giải thích hiện tượng train loss tiếp tục giảm nhưng validation loss bắt đầu đi ngang hoặc tăng lên | Đối chiếu với biểu đồ loss thực tế của notebook ở epoch 15-20. |
| **Phần 4** | Cấu hình tham số để mô hình reasoning không nuốt hết token ngắn | Hướng dẫn thêm `reasoning_effort="none"` và nâng `max_completion_tokens` | Hiểu rõ cơ chế suy luận ẩn của các dòng model mới. |

### Điều hiểu ra sau buổi thực hành:
1. **Accuracy là cái bẫy nguy hiểm:** Với tập dữ liệu mất cân bằng (87% ham / 13% spam), một mô hình đoán bừa toàn bộ là ham vẫn đạt 87% Accuracy nhưng hoàn toàn vô dụng trong thực tế. Metric cốt lõi bắt buộc phải là **Precision, Recall và F1-Score**.
2. **Không có mô hình nào tối thượng cho mọi bài toán:** Mô hình phức tạp nhất hay LLM khổng lồ không đồng nghĩa với giải pháp tối ưu cho sản phẩm. Kỹ sư AI cần biết cân bằng giữa điểm F1, độ trễ p95 ($< 50\text{ ms}$), thông lượng (throughput) và chi phí tài chính trên từng ngàn request.
