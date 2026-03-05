# Giới thiệu quá trình thực hiện multimodal 
## Mô tả về dataset: 
### Tên bệnh: Ung thu vú
### Nguồn: https://www.kaggle.com/datasets/orvile/mias-dataset
#### Tham khảo từ: J Suckling et al (1994): The Mammographic Image Analysis Society Digital Mammogram Database Exerpta Medica. International Congress Series 1069 pp375-378.

**File images:**
Bộ dữ liệu MIAS phân tích ảnh chụp vú (mammography). Bộ dữ liệu này thường bao gồm:

1.  Hình ảnh chụp X-quang vú (mammogram) với độ phân giải tiêu chuẩn
2.  Thông tin về các trường hợp bình thường (normal), lành tính (benign) và ác tính (malignant)
3.  Chú thích về vị trí và kích thước của các bất thường nếu có
4.  Thông tin về loại bất thường (như khối u, vôi hóa, biến dạng, v.v.) 
5.  Thông tin về mật độ mô vú (Density)


**File csv:**
**Mô tả Cột ngắn gọn:**
1. Số MIAS: Số tham chiếu cơ sở dữ liệu MIAS.
2. BG (Mô nền): Loại mô nền:
    -   F: Mỡ
    -   G: Tuyến mỡ
    -   D: Tuyến đặc
3. LỚP: Loại bất thường hiện tại:
    -   CALC: Vôi hóa
    -   CIRC: Khối được xác định rõ/giới hạn
    -   SPIC: Khối có gai
    -   MISC: Khối khác, không xác định rõ
    -   ARCH: Biến dạng kiến trúc
    -   ASYM: Không đối xứng
    -   NORM: Bình thường
4. SEVERITY: Mức độ nghiêm trọng của bất thường:
    -   B: Lành tính
    -   M: Ác tính
5. (5-6) tọa độ x, y: Tọa độ của tâm bất thường.
6. Bán kính (pixel): Bán kính gần đúng của hình tròn bao quanh bất thường.
**Mô tả cho các Cột bổ sung :**
    -   MẬT ĐỘ: Phân loại mật độ mô, được chỉ định trong hình ảnh chụp nhũ ảnh là A (mật độ thấp), B, C hoặc D (mật độ cao).
    -   BI-RADS: Phân loại BI-RADS được sử dụng để đánh giá các bất thường chụp nhũ ảnh, ví dụ: BI-RADS 1 cho bình thường, BI-RADS 5 cho nghi ngờ ác tính cao.
    -   Nhóm: Thể loại phân loại chung (ví dụ: Bình thường, Khối u, Vôi hóa).

**Thông tin từ dữ liệu CSV**
Bao gồm các trường dữ liệu như Refnum, BG, class, X, Y, Radius, Density, Bi-rads, class_full, class_group 
   ![alt text](image.png)

**Hai biểu đồ cột từ dữ liệu trong file csv:**
Phân phối của các lớp chẩn đoán (NORM, CIRC, SPIC, vv.)
Phân phối mức độ nghiêm trọng (Normal, Benign, Malignant)

![alt text](image-1.png)

**Biểu đồ trực quan hóa:**
Phân phối mật độ vú (A, B, C/D)
Mối quan hệ giữa lớp chẩn đoán và mức độ nghiêm trọng

![alt text](image-2.png)

**Biểu đồ thể hiện mối quan hệ giữa classclass và mức độ nghiêm trọng**

![alt text](image-3.png)

**Một số hình ảnh của dataset trước khi xử lý:**

![alt text](image-4.png)

**Traning multimodal được trình bày trong link này:** https://drive.google.com/file/d/1MSQgC3hM3xF7qQRSXncufMzQRjdcETuO/view?usp=drive_link
