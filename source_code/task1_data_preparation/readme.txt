Task 1 - Chuẩn bị dữ liệu và EDA,...
================================
Mục đích
--------------------------------

Thư mục này chứa notebook gốc cho phần chuẩn bị data và xử lí dữ liệu,.. 

Nội dungm chính bao gồm:
- Đầu tiên là đi đọc bộ dữ liệu giá nhà Nhật Bản nè,
- Đi tìm hiểu ý nghĩa các cột quan trọng cần cho mô hình,
- Đi kiểm thử xem có tồn tại missing values không,
- Đi kiểm tra và xóa dòng trùng lặp trong dataset,
- Xử lý dữ liệu thiếu (nếu có),
- Phân tích biến mục tiêu price,
- Vẽ các biểu đồ EDA như phân phối giá, price vs area,
price vs house age, price by prefecture,
- Tạo feature mới như house_age và density,
- Đi xử lý outlier,
- Chọn tập feature nhỏ cho phiên bản giữa kì,
- Xuất file train/test raw và processed.

File chính (duy nhất 1 file code):
----------------------------------
- task1_data_preparation.ipynb
  Notebook gốc cho phần làm sạch dữ liệu, EDA và tiền xử lý.

Cách chạy
---------
1. Mở file task1_data_preparation.ipynb bằng Google Colab/vscode.

2. Đảm bảo có file dữ liệu gốc. Trong notebook ban đầu, file CSV chính được
   đọc với tên:

   All_prefectures_buildings_with_migration.csv

   Nếu file CSV không nằm trong thư mục này, có 2 cách:
   - copy file từ:
     ../deployment/dataset_jp_house/
   - copy file từ:
     source_code/dataset_jp_house/All_prefectures_buildings_with_migration.csv

3. Chạy notebook từ trên xuống dưới.

Kết quả mong đợi
----------------
Notebook có thể xuất các file:
- train_raw.csv
- test_raw.csv
- train_processed.csv
- test_processed.csv

Các file này có thể được dùng làm đầu vào cho phần modeling ở Task 2 và Task 3.