# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy: 11/09/2026**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics: Python**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** 

- Thêm 2 cell: 
  ```notebook-python
  !zip -r day1_lab_outputs.zip day1_lab_outputs

  from google.colab import files
  files.download('day1_lab_outputs.zip')
  ```

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id = 468`, `class_name = "cab"`, `rank = 1`, `score = 0.510915`, `taxonomy_name = "ImageNet-1K"`
- **Record này mô tả toàn ảnh như thế nào?**  
Model chọn `cab` **(taxi)** là lớp có xác suất/score cao nhất để **đại diện cho toàn bộ ảnh** `traffic`, với score **0.510915**. Đây không phải là nhãn của từng xe trong ảnh.
- **Ai định nghĩa class list mà checkpoint có thể dự đoán?**  
**Checkpoint** `yolo11n-cls.pt` **và taxonomy ImageNet-1K** định nghĩa danh sách các lớp mà model có thể dự đoán. Trong code, tên lớp được lấy từ `result.names`.
- **Vì sao cần giữ cả ID, tên lớp và tên taxonomy?**
  - `class_id = 468`: định danh lớp trong checkpoint.
  - `class_name = cab`: tên lớp dễ đọc.
  - `taxonomy_name = ImageNet-1K`: xác định lớp này thuộc hệ phân loại nào.  
  → Giúp **định danh và đối chiếu prediction chính xác**.
- **Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**  
Cần quy định **cách chọn một lớp đại diện cho toàn ảnh**, ví dụ ưu tiên chủ thể chính/nổi bật nhất, và áp dụng thống nhất cho các ảnh.
- **Vì sao model score không phải ground truth?**  
Vì `0.510915` là **score do model dự đoán**, thể hiện mức độ model tin vào lớp `cab`; nó **không phải nhãn được con người xác nhận**. Model có thể có score cao nhưng prediction vẫn sai.



## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
  - class_name: person
  - score: 0.91
  - "bbox_xyxy": [       385.33,       69.24,       498.92,       348.92     ],    
  - "bbox_width": 113.58,    
  - "bbox_height": 279.68

- **Diễn giải vị trí box:** Box `person` nằm **ở phía bên phải ảnh**, bao quanh người đang đứng trong khu vực bếp.
- **So sánh số prediction ở hai threshold:**
  - `threshold = 0.20`: **17 vật thể**
  - `threshold = 0.35`: **17 vật thể**
  - `threshold = 0.60`: **6 vật thể**
- **Độ bao phủ và khối lượng reviewer:** Threshold thấp giữ được **nhiều detection hơn**, tăng độ bao phủ nhưng reviewer phải kiểm tra nhiều hơn và có thể gặp false positive. Threshold cao giảm số prediction, giảm tải reviewer nhưng có nguy cơ **bỏ sót vật thể**.
- **Quy tắc box chặt:** Box phải **ôm sát toàn bộ phần nhìn thấy của object**, không lấy quá nhiều nền và không cắt vào phần object nhìn thấy.
- **Object bị che khuất/cắt mép:** Guideline cần quy định rõ **có annotate phần nhìn thấy hay toàn bộ object ước lượng**, và khi không đủ chắc chắn thì **đưa vào escalation/review** thay vì tự quyết.



## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- **Một record:** Ví dụ `kitchen-001`: `class_name = person`, `score = 0.899318`, có **348 điểm polygon**. `polygon_xy` chứa các tọa độ pixel của đường biên mask, ví dụ bắt đầu bằng các điểm `[446,70]`, `[446,71]`, `[445,70]`…
- **Polygon bổ sung gì so với box?**  
Box chỉ xác định **hình chữ nhật bao quanh object**; polygon mô tả **đường biên/hình dạng thực tế của object**, nên chính xác hơn khi object có hình dạng không vuông hoặc bị che khuất.
- `instance_id` **dùng để làm gì và không phải loại ID nào?**  
`instance_id` dùng để **định danh từng object cụ thể trong một ảnh**, ví dụ `kitchen-001`. Nó **không phải** `class_id` (ID của loại object), cũng không phải `coco_image_id` (ID của ảnh).
- **Quy tắc biên mask:**  
Mask phải **bám sát phần pixel nhìn thấy của object**, không ăn sang nền; không cố đoán phần bị che khuất nếu không nhìn thấy rõ.
- **Vùng mờ/tiếp xúc/che khuất:**  
Guideline cần quy định **ranh giới giữa các object và phần bị che khuất**. Nếu không xác định rõ biên mask thì **escalate/review** thay vì annotator tự đoán.



## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`


| Tác vụ                | Đơn vị/định dạng ground truth                       | Lỗi hoặc điểm mơ hồ quan sát được                                                                                           | Annotator làm gì?                                                                                 | Reviewer xem gì?                                                                     |
| --------------------- | --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Phân loại ảnh         | class_id + class_name + taxonomy                    | Ảnh có nhiều chủ thể nên không rõ chọn lớp nào để đại diện                                                                  | Chọn lớp đại diện cho toàn ảnh theo guideline                                                     | Kiểm tra class đúng, đúng taxonomy và quy tắc chọn lớp được áp dụng nhất quán        |
| Phát hiện vật thể     | Mỗi object = 1 box xyxy theo pixel + class          | Box quá rộng/hẹp; object bị che khuất hoặc cắt mép; threshold thấp có thể nhiều false positive, threshold cao có thể bỏ sót | Gán đúng class và vẽ **box chặt quanh phần object nhìn thấy**; trường hợp không rõ → escalation   | Kiểm tra class, số object, vị trí/kích thước box và các trường hợp che khuất/cắt mép |
| Instance segmentation | Mỗi instance = 1 polygon/mask + class + instance_id | Biên object mờ, tiếp xúc hoặc bị che khuất → khó xác định ranh giới mask                                                    | Vẽ polygon bám sát phần nhìn thấy của object, không ăn sang nền; trường hợp không rõ → escalation | Kiểm tra class, từng instance, instance_id và độ chính xác của biên mask/polygon     |




## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng các dữ liệu đúng phạm vi được giao, không tự ý sao chép, chia sẻ hoặc đưa dữ liệu ra ngoài hệ thống làm việc
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Người phụ trách dự án hoặc đầu mối quản lý dữ liệu



## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.