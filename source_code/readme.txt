Nhom8 Final Project
================================
Cấu trúc thư mục source_code
---------------------------------
Gồm có 4 thư mục, tương ứng với các vai trò khác nhau, các thành viên làm khác nhau:

1. task1_data_preparation
   - Chứa file code (xài notebook chạy gg colab) gốc cho phần tìm hiểu dữ liệu dataset, EDA data,
   làm sạch dữ liệu, tiền xử lí data, feature engineering và xuất file train/test đồng thời thử nghiệm
   nhẹ cho phần chạy mô hình cơ bản.
   - File chinh: task1_data_preparation.ipynb
   - Đây là phần xử lí dữ liệu là chính, lưu ý, số feature chọn là ít và có giới hạn,
   nó không phản ánh số feature được sử dụng cho các hướng làm khác của các thành viên sau. 

2. task2_modeling_direction_1
   - Hướng làm cơ bản mô hình, thử nghiệm 1

3. task3_modeling_direction_2
   - Hướng làm nâng cao (advanced model),thử nghiệm 2

4. deployment
   - Chứa phiên bản triển khai ứng dụng web cho người dùng sử dụng.
   - Có làm lại phần data, và lựa lại feature khác đi để có thể thõa mãn điều kiện sử dụng thực tế (tọa độ,..)
   - Có bao gồm: FastAPI backend, React/Leaflet frontend, saved model,
     dữ liệu đã xử lí, reports/figures.

5. Dataset
   - Chứa các file .csv la dataset đước sử dụng cho các thưc mục: 1. 2. 3. 4.
   - Link dataset:


Quy trình sử dụng như sau:
---------------------
1. Đọc task1_data_preparation để hiểu cách nhóm xử lí cơ bản data gốc.
2. Sau đó là đến task2_modeling_direction_1 va task3_modeling_direction_2.
3. Cuối cùng là deployment, chạy nó để demo ung dung web dự đoán giá nhà thực tế như nào.


Lưu ý quan trọng:
----------------
Do 2 hướng được làm khác nhau, và chưa có sự thống nhất, nên chúng tôi quyết định, tự phát triển
riêng lại từ đầu cho phần "deployment" này về data,code (lựa feature phù hợp,mô hình), nó không trộn với chương modeling/evaluation
Modeling/Evaluation chính của các thành viên khác nên sẽ không thắc mắc sao nó khác.

