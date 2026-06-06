==================================================
  DATA MINING - DỰ ĐOÁN GIÁ BẤT ĐỘNG SẢN NHẬT BẢN
==================================================

MÔ TẢ
------
Notebook này thực hiện bài toán dự đoán tổng giá trị giao dịch bất động sản
(TotalTransactionValue) dựa trên các đặc trưng như diện tích, vị trí, năm xây
dựng, các chỉ số kinh tế vĩ mô, v.v.

Dữ liệu gồm hai file: train.csv và test.csv.


CÁC BƯỚC THỰC HIỆN
-------------------
1. Chuẩn bị dữ liệu
   - Đọc dữ liệu từ train.csv và test.csv
   - Tạo feature mới: house_age = Year - ConstructionYear
   - Xóa cột Type (chỉ có 1 giá trị duy nhất)
   - Cyclic encoding cho cột Quarter (sin/cos)
   - One-Hot Encoding cho cột Prefecture
   - Log transform biến mục tiêu: log1p(TotalTransactionValue)
   - Chuẩn hóa các cột số bằng RobustScaler

2. Phân tích tương quan
   - Vẽ heatmap ma trận tương quan giữa các feature
   - Loại bỏ các cột tương quan cao (> 0.85):
     RegionResidentialArea, Prefecture_Hokkaido, GDP_Deflator_Change_pct

3. Phát hiện và xử lý Outlier (Ensemble 3 phương pháp)
   - Isolation Forest
   - Autoencoder (PyTorch): kiến trúc input → 64 → latent(12) → 64 → input,
     đánh dấu outlier dựa trên reconstruction loss (ngưỡng percentile 95)
   - Mahalanobis Distance trên latent space của Autoencoder
   - Vote 2/3: mẫu bị ít nhất 2/3 phương pháp đánh dấu → giảm trọng số (weight)

4. Thêm Latent Features
   - Trích xuất 12 chiều latent space từ Autoencoder đã huấn luyện
   - Ghép vào tập train và test để hỗ trợ các mô hình ML phía sau

5. Huấn luyện mô hình
   Các mô hình được thử nghiệm:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - XGBoost (GPU)
   - Random Forest
   - Extra Trees
   - Gradient Boosting
   - MLP (mạng nơ-ron, PyTorch): input → 128 → 64 → 1, BatchNorm + Dropout

6. Đánh giá
   Các chỉ số: RMSE, MAE, R², SMAPE, Accuracy ±5/10/20/30/40%
   (đo trên cả log scale và original scale)

7. Lưu mô hình
   Mô hình XGBoost tốt nhất được lưu ra file: xgboost_house_price.pkl


YÊU CẦU THƯ VIỆN
-----------------
pandas, numpy, scikit-learn, xgboost, torch, matplotlib, seaborn, scipy, joblib, tqdm


FILE ĐẦU RA
-----------
- xgboost_house_price.pkl : Mô hình XGBoost đã huấn luyện

==================================================
