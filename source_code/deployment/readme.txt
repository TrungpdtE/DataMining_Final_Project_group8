Deployment - Ứng dụng web cho người dùng sử dụng (được xem là 1 hướng phát triển riêng)
================================================





Mục đích
--------
Thư mục này thì nó đang chứa phiên bản triển khai ứng dụng web cho đồ án dự đoán giá nhà
Nhật Bản. Phần này tách riêng với các task modeling chính và được xây dựng lại
để phục vụ demo tương tác.





Nội dung thư mục gồm nhiều folder và file khác nhau:
----------------

- api/
  Backend FastAPI. File chính: api/main.py

- frontend/
  Frontend tĩnh sử dụng React và Leaflet.

- src/
  Code tiền xử lý, feature engineering, train model, predict và visualization.

- dataset_jp_house/
  Dữ liệu nhà Nhật Bản, dữ liệu macro, dữ liệu tọa độ và dữ liệu ga tàu.

- data/processed/
  File train/test đã xử lý cho deployment pipeline.

- saved_models/
  Model đã lưu và metadata.

- reports/
  Kết quả model, feature importance và các hình đánh giá.





Cách cài thư viện (nếu chưa có)
-----------------

Chạy lệnh sau trong thư mục deployment:

pip install -r requirements.txt





Cách train model deployment
---------------------------


Train mặc định:

python -m src.models.train_xgb


Train toàn bộ dataset:

PowerShell:

$env:JAPAN_HOUSE_MAX_ROWS="0"
python -m src.models.train_xgb


Chọn model để train:

$env:JAPAN_HOUSE_MODELS="xgboost,lightgbm,extra_trees"
python -m src.models.train_xgb




Cách chạy ứng dụng web
----------------------

Chạy trong thư mục deployment:

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

Sau đó mở trình duyệt:

http://127.0.0.1:8000/
(lỡ mà trùng port 8000 thì nhớ đổi/ kill pid của port đó đi rồi chạy lại)






Chức năng chính
---------------
- Gồm có bản đồ (free api src) tương tác thị trường nhà Nhật Bản.
- Chế độ xem giá lịch sử và giá dự đoán.
- Có chức năng hiển thị dạng cluster và heatmap.
- Click trên bản đồ để dự đoán giá.
- Tìm ga tàu gần nhất và tính thời gian đi bộ ước lượng.
- Hệ thống sẽ tự động đi tạo feature địa lý từ tọa độ click của người dùng.
- Nhập kịch bản macro .
- Hiển thị phiếu kết quả (là tab detail panel).
- Mô phỏng giá theo timeline.
- Endpoint train lại model.



Tương lai: có thể sẽ phát triển thêm dashboard thống kê data 


Lưu ý quan trọng
----------------
Kết quả model trong thư mục này là kết quả của riêng cho pipeline deployment. 
Nên trình bày riêng trong chương Deployment, không trộn với chương Modeling/Evaluation
chính của các thành viên khác.