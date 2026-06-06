# 1. Tổng số feature model đang dùng

Mô hình deployment hiện DÙNG 61 feature.

Danh sách này có thể xem trong:

```text
folder: saved_models/model_metadata.json
```

Trong API khi predict thì backend sẽ load đúng cái danh sách _feature_columns_ này rooif đưa vào model. Vì vậy mà web sẽ không gửi đủ 61 feature trực tiếp, mà web chỉ gửi một phần input. Backend sẽ tạo lại các feature còn lại để khớp với 61 feature lúc train.

Luồng predict chính:

```text
Frontend input/click map
        |
        v
POST /predict
        |
        v
Backend lấy context gần nhất trong dataset
        |
        v
Backend ghi đè các giá trị user nhập
        |
        v
prepare_model_frame()
        |
        v
clusterer + target_encoder
        |
        v
model.predict(feature_columns)
```

# 2. 61 feature của deployment model

```text
1. area
2. floor_area
3. frontage
4. house_age
5. coverage
6. FloorAreaRatio
7. Year
8. Quarter
9. Migration
10. AverageTimeToStation
11. MunicipalityCategory
12. density
13. latitude
14. longitude
15. Distance_to_designated_city
16. dist_to_tokyo_km
17. dist_to_osaka_km
18. dist_to_nagoya_km
19. dist_to_fukuoka_km
20. dist_to_sapporo_km
21. dist_to_nearest_major_center_km
22. log_dist_to_tokyo
23. log_dist_to_nearest_major_center
24. quarter_sin
25. quarter_cos
26. location_cluster
27. Prefecture_target_mean
28. Location_target_mean
29. Close_to_Tokyo
30. Close_to_greater_Tokyo_area
31. Close_to_designated_city_flag
32. RegionCommercialArea
33. RegionIndustrialArea
34. RegionPotentialResidentialArea
35. RegionResidentialArea
36. Region_Chubu
37. Region_Chugoku
38. Region_Hokkaido
39. Region_Kansai
40. Region_Kanto
41. Region_Kyushu
42. Region_Shikoku
43. Region_Tohoku
44. gdp_growth
45. interest_rate
46. inflation_rate
47. population_growth
48. housing_price_index
49. last_year_prefecture_avg_price
50. last_year_prefecture_tx_count
51. prefecture_price_growth_1y
52. prefecture_price_growth_3y
53. last_year_location_avg_price
54. location_price_growth_1y
55. last_year_global_avg_price
56. global_price_growth_1y
57. global_price_growth_3y
58. Type
59. Prefecture
60. nearest_major_center
61. Nearest_designated_city
```

# 3. Feature người dùng nhập trực tiếp trên web

Các feature - giá trị sau là do người dùng nhập hoặc chọn trên giao diện:

| Input trên web | Feature model tương ứng | Ghi chú |
|---|---|---|
| Tỉnh / đô thị | `Prefecture` | Có thể chọn tay hoặc tự fill khi click map |
| Thành phố / quận | `Location` | Có thể chọn tay hoặc tự fill khi click map |
| Diện tích đất | `area` | User nhập |
| Diện tích sàn | `floor_area` | User nhập |
| Mặt tiền | `frontage` | User nhập, hoặc app gợi ý từ context gần nhất |
| Mật độ xây dựng % | `coverage` | User nhập, hoặc app gợi ý từ context gần nhất |
| Hệ số sử dụng đất | `FloorAreaRatio` | User nhập, hoặc app gợi ý từ context gần nhất |
| Loại bất động sản | `Type` | User chọn, hoặc app gợi ý từ context gần nhất |
| Năm xây dựng | dùng để tính `house_age` | User nhập |
| Phút tới ga | `AverageTimeToStation` | User nhập hoặc app tự tính từ ga gần nhất |
| Tăng trưởng GDP | `gdp_growth` | Web load default từ `/macro/latest`, user có thể chỉnh |
| Lãi suất | `interest_rate` | Web load default từ `/macro/latest`, user có thể chỉnh |
| Lạm phát | `inflation_rate` | Web load default từ `/macro/latest`, user có thể chỉnh |
| Chỉ số giá nhà | `housing_price_index` | Web load default từ `/macro/latest`, user có thể chỉnh |

Ghi chú:

- `population_growth` cũng là feature model dùng. Web load giá trị này từ `/macro/latest` và gửi trong payload nhưng hiện giao diện chưa có ô nhập riêng cho nó, tương lai sẽ phát triển thêm.
- `Year` lấy từ slider///năm đang chọn trên web.
- `Quarter` hiện frontend đang gửi cố định là `2` khi gọi `/predict`, thử nghiệm nên để tạm vậy

# 4. Feature app tự điền khi user click map

Khi user click một điểm trên bản đồ, frontend gọi:

```text
GET /location/context?latitude=...&longitude=...
```

Backend sẽ tìm context và trả về một số giá trị để frontend tự điền vào form:

| Giá trị app tự điền | Cách lấy |
|---|---|
| `latitude` | Tọa độ điểm user click |
| `longitude` | Tọa độ điểm user click |
| `Prefecture` | Lấy từ giao dịch lịch sử gần nhất trong dataset |
| `Location` | Lấy từ giao dịch lịch sử gần nhất trong dataset |
| `AverageTimeToStation` | Tính từ ga gần nhất rồi đổi sang walking minutes |
| `frontage` | Gợi ý từ giao dịch lịch sử gần nhất, nếu có |
| `coverage` | Gợi ý từ giao dịch lịch sử gần nhất, nếu có |
| `FloorAreaRatio` | Gợi ý từ giao dịch lịch sử gần nhất, nếu có |
| `Type` | Gợi ý từ giao dịch lịch sử gần nhất, nếu có |

Điểm quan trọng cần lưu ý là:

- Khi chúng ta click map thì không thể tự biết tất cả thông tin thật của căn nhà.
- Vì vậy app dùng điểm click để lấy tọa độ  sau đó lấy một phần context từ giao dịch lịch sử gần nhất.
- User vẫn có thể chỉnh lại area, floor_area, frontage, coverage, năm xây dựng, phút tới ga, macro scenario trước khi predict.

# 5. Feature backend tự tính từ input

## 5.1. `house_age`

Tính từ năm dự đoán và năm xây dựng:

```text
house_age = Year - construction_year
```

Giá trị được giới hạn trong khoảng hợp lý:

```text
0 <= house_age <= 100
```

## 5.2. `density`

Tính từ diện tích sàn và diện tích đất:

```text
density = floor_area / area
```

Ghi chú: phần này đã được chỉnh trong API để khi user đổi `area` hoặc `floor_area`, `density` cũng được tính lại trước khi predict.

## 5.3. `quarter_sin` và `quarter_cos`

Tính từ `Quarter` để biểu diễn tính chu kỳ theo quý:

```text
quarter_sin = sin(2*pi*Quarter/4)
quarter_cos = cos(2*pi*Quarter/4)
```

Hiện frontend gửi `Quarter = 2` cho predict đơn lẻ.

## 5.4. Các khoảng cách đến trung tâm lớn

Backend dùng tọa độ `latitude`, `longitude` để tính khoảng cách Haversine đến 5 trung tâm lớn:

```text
Tokyo   = (35.681236, 139.767125)
Osaka   = (34.702485, 135.495951)
Nagoya  = (35.170915, 136.881537)
Fukuoka = (33.590355, 130.401716)
Sapporo = (43.068661, 141.350755)
```

Các feature tạo ra:

```text
dist_to_tokyo_km
dist_to_osaka_km
dist_to_nagoya_km
dist_to_fukuoka_km
dist_to_sapporo_km
```

Công thức Haversine:

```text
distance = 2 * R * arcsin(sqrt(a))
R = 6371.0088 km
```

## 5.5. Trung tâm lớn gần nhất

Sau khi tính khoảng cách đến 5 trung tâm lớn, backend lấy khoảng cách nhỏ nhất:

```text
dist_to_nearest_major_center_km = min(dist_to_tokyo_km,
                                      dist_to_osaka_km,
                                      dist_to_nagoya_km,
                                      dist_to_fukuoka_km,
                                      dist_to_sapporo_km)
```

Tên trung tâm gần nhất được lưu vào:

```text
nearest_major_center
```

## 5.6. Log distance

Backend tạo thêm log distance để giảm độ lệch của biến khoảng cách:

```text
log_dist_to_tokyo = log1p(dist_to_tokyo_km)
log_dist_to_nearest_major_center = log1p(dist_to_nearest_major_center_km)
```

## 5.7. `location_cluster`

Backend dùng KMeans model đã fit lúc train để phân cụm tọa độ:

```text
location_cluster = kmeans.predict(latitude, longitude)
```

Số cluster hiện dùng:

```text
12
```

## 5.8. Target encoding

Backend dùng encoder đã fit lúc train để tạo:

```text
Prefecture_target_mean
Location_target_mean
```

Cách tính khi train:

```text
encoded = global_mean * (1 - weight) + group_mean * weight
weight = count / (count + smoothing)
```

Khi predict, nếu gặp prefecture/location lạ, encoder dùng global mean.

# 6. Feature lấy từ context lịch sử gần nhất

Một số feature không thể biết chỉ từ map click hoặc form input, nên backend lấy từ dòng dữ liệu lịch sử gần nhất hoặc matching context trong dataset:

```text
Migration
MunicipalityCategory
Distance_to_designated_city
Close_to_Tokyo
Close_to_greater_Tokyo_area
Close_to_designated_city_flag
RegionCommercialArea
RegionIndustrialArea
RegionPotentialResidentialArea
RegionResidentialArea
Region_Chubu
Region_Chugoku
Region_Hokkaido
Region_Kansai
Region_Kanto
Region_Kyushu
Region_Shikoku
Region_Tohoku
Nearest_designated_city
```


- User click một điểm mới nhưng app cần biết vùng đó thuộc kiểu đô thị nào, vùng địa lý nào, có gần Tokyo hay không, migration ra sao.
- Những thông tin này được suy ra bằng cách chọn một giao dịch gần nhất trong dataset làm context.

# 7. Feature lag-history

Các feature sau là thông tin giá lịch sử:

```text
last_year_prefecture_avg_price
last_year_prefecture_tx_count
prefecture_price_growth_1y
prefecture_price_growth_3y
last_year_location_avg_price
location_price_growth_1y
last_year_global_avg_price
global_price_growth_1y
global_price_growth_3y
```

Với dữ liệu lịch sử có sẵn thì các feature này đã được tính trong preprocessing.

Với forecast năm tương lai như năm 2026 (năm nay), backend gọi `_apply_forecast_lags()`:

- Sẽ lấy năm quan sát gần nhất trước năm dự đoán,
- Sẽ tính giá trung bình theo prefecture/location/global,
- Sẽ tính tăng trưởng 1 năm và 3 năm,
- Sẽ dùng các giá trị đó làm context lịch sử cho năm tương lai.

# 8. Feature macro

Các feature macro:

```text
gdp_growth
interest_rate
inflation_rate
population_growth
housing_price_index
```

Cách mà nó lấy:

- Khi load web thì frontend gọi `/macro/latest?year=2026`.
- Backend lấy giá trị macro gần nhất theo năm.
- Nguoiwfb dùng có thể chỉnh một số ô trên giao diện như GDP growth, interest rate, inflation rate, housing price index.
- `population_growth` hiện được load và gửi theo payload, nhưng chưa có ô input riêng trên UI.

## 9. Feature ga tàu

Thông tin ga gần nhất KHÔNG đi trực tiếp vào model dưới dạng tên ga.

Backend dùng ga gần nhất để tính:

```text
distance_km
walking_minutes
```

Sau đó:

```text
AverageTimeToStation = walking_minutes
```

Cách tính:

```text
walking_minutes = distance_km / 4.8 * 60
```

Trong đó `4.8 km/h` là tốc độ đi bộ giả định.
