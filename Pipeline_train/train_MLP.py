import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_excel('Pumpkin_Seeds_Dataset.xlsx')

# Mã hoá đầu ra object thành  0 và 1
df['Class'] = df['Class'].map({
    'Çerçevelik': 0,
    'Ürgüp Sivrisi': 1
})
df['Class'].isna().sum()

# 2. Tách Feature (X) và Target (y) ngay từ đầu (chưa xử lý gì cả)
X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Xữ lý outlier bằng IQR
cols_outliers = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 'Convex_Area',
                 'Equiv_Diameter']

def handle_outliers_with_train_bounds_tensorflow(train_df, test_df, columns):
    # CHỈNH SỬA TẠI ĐÂY: Chuyển kiểu dữ liệu sang float để tránh cảnh báo dtype
    train_copy = train_df.astype(float)
    test_copy = test_df.astype(float)

    for col in columns:
        # Tính toán Q1, Q3 và IQR CHỈ dựa trên tập Train
        Q1 = train_copy[col].quantile(0.25)
        Q3 = train_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Giá trị trung vị để thay thế cũng tính từ tập Train
        median_val = train_copy[col].median()

        # Thay thế outlier ở cả 3 tập bằng giá trị Mean của tập Train
        for d in [train_copy, test_copy]:
            # Dùng .loc để gán giá trị mà không gây lỗi SettingWithCopy
            mask = (d[col] < lower_bound) | (d[col] > upper_bound)
            d.loc[mask, col] = median_val

    return train_copy, test_copy

X_train, X_test = handle_outliers_with_train_bounds_tensorflow(X_train.copy(), X_test.copy(), cols_outliers)


#xữ lý loại bỏ bớt cột features có |corr| >0.95
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]

# Loại bỏ các cột
X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)

#khởi tạo scaler
scaler = StandardScaler()
# Fit trên X_train và transform cả 2 tập
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dropout(0.3),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.3, # Chia 30% từ X_train để làm validation
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],

)


# --- 1. Lưu mô hình TensorFlow/Keras ---
model.save('pumpkin_model.keras')

# --- 2. Lưu bộ tiền xử lý và thông tin đặc trưng ---
# Chúng ta cần lưu Scaler và danh sách các cột bị drop
# Ngoài ra, hàm xử lý outlier cần thông số Q1, Q3, Mean của tập Train
outlier_params = {}
for col in cols_outliers:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_params[col] = {
        'lower': Q1 - 1.5 * IQR,
        'upper': Q3 + 1.5 * IQR,
        'median': X[col].median()
    }

preprocessing_data = {
    'scaler': scaler,
    'to_drop': to_drop,
    'cols_outliers': cols_outliers,
    'outlier_params': outlier_params
}

joblib.dump(preprocessing_data, 'preprocessing_bundle.joblib')

print("Đã lưu mô hình và bộ tiền xử lý thành công!")
