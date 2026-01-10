import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from my_transformers import OutlierHandler, CorrelationDropper

# 1. Đọc dữ liệu
df = pd.read_excel('D:/Pumkin/Pumpkin_Seeds_Dataset/Pumpkin_Seeds_Dataset.xlsx')
X = df.drop(columns=["Class"])
y = df["Class"]

# 2. Chia tập dữ liệu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Danh sách cột cần xử lý outlier
cols_outliers = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Convex_Area',
                'Equiv_Diameter']

# 3. Tạo Pipeline
full_pipeline = Pipeline([
    ("outlier_remover", OutlierHandler(columns=cols_outliers)),
    ("corr_dropper", CorrelationDropper(threshold=0.95)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(penalty="l2", class_weight="balanced", max_iter=1000))
])

# 4. Huấn luyện (Pipeline sẽ fit các transformer chỉ trên X_train)
full_pipeline.fit(X_train, y_train)

# 5. Đánh giá
print("Train Score:", full_pipeline.score(X_train, y_train))
print("Val Score  :", full_pipeline.score(X_val, y_val))
print("Test Score :", full_pipeline.score(X_test, y_test))

# 6. Lưu Pipeline lại để sử dụng cho thực tế
joblib.dump(full_pipeline, 'model_pipeline.pkl')