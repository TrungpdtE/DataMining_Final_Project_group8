Task 2 - Xây dung và so sánh mô hình dự đoán (Modeling)
=======================================================

Mục đích
---------
Huấn luyện, đánh giá và so sánh nhiều mô hình máy học khác nhau (từ Regression truyền thống đến Deep Learning) để tìm ra giải pháp tối ưu cho bài toán dự đoán giá nhà tại Nhật Bản.

Nội dung chính bao gồm:
- Thiết lập cấu hình chung cho notebook (số fold, tolerance cho độ chính xác, số epoch cho Deep Learning,...).
- Nạp dữ liệu đã qua tiền xử lý (train_processed.csv và test_processed.csv).
- Kiểm tra cấu trúc dữ liệu và phân tích phân phối của biến mục tiêu price.
- Phát hiện các cột dữ liệu bị mất cân bằng (imbalanced features) và phân tích các khoảng giá hiếm.
- Xây dựng Pipeline huấn luyện cho danh sách các mô hình:
    + Baseline: Linear Regression, Ridge, Lasso.
    + Tree-based: RandomForest, ExtraTrees, GradientBoosting, XGBoost.
    + Deep Learning: PyTorch MLP (Multi-Layer Perceptron).
- So sánh hiệu suất giữa các mô hình dựa trên các chỉ số: RMSE, MAE, R2 và độ chính xác tùy chỉnh (accuracy_tol_03).
- Thử nghiệm và so sánh các chiến lược Cross-validation: KFold, RepeatedKFold, ShuffleSplit, StratifiedKFold.
- Lựa chọn mô hình tốt nhất (thường là XGBoost) dựa trên kết quả validation.
- Đánh giá cuối cùng trên tập test và hiển thị bảng dự đoán mẫu.

File chính (duy nhất 1 file code):
----------------------------------
- DataMining_PredictModel.ipynb
  Notebook thực hiện toàn bộ quy trình so sánh mô hình, Cross-validation và huấn luyện Deep Learning.

Cách chạy
---------
1. Mở file DataMining_PredictModel.ipynb bằng Google Colab hoặc VS Code.

2. Đảm bảo các file dữ liệu đầu vào có sẵn trong cùng thư mục:
   - train_processed.csv
   - test_processed.csv

3. Cấu hình các tham số tại phần "1. Cấu hình chung" (ví dụ: điều chỉnh N_SPLITS hoặc ACCURACY_TOLERANCE) nếu cần thiết.

4. Chạy notebook từ trên xuống dưới.

Kết quả mong đợi
----------------
- Bảng tổng hợp so sánh các mô hình học máy (Model Comparison Table).
- Đánh giá hiệu quả của các chiến lược Cross-validation khác nhau.
- Kết quả dự đoán cuối cùng trên tập test với các chỉ số đánh giá chi tiết.
- Danh sách top các đặc trưng quan trọng (Feature Importance) nếu sử dung các mô hình cây.

