import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load dữ liệu
df = pd.read_excel('D:\Pumkin\Pumpkin_Seeds_Dataset\Pumpkin_Seeds_Dataset.xlsx')

# Mã hoá nhãn
df['Class'] = df['Class'].map({'Çerçevelik': 0, 'Ürgüp Sivrisi': 1})

# Lấy tất cả 12 đặc trưng (không bỏ cột nào)
X = df.drop(columns=["Class"])
y = df["Class"]

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Chuẩn hóa dữ liệu (Toàn bộ 12 đặc trưng)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lưu scaler độc lập
joblib.dump(scaler, 'scaler.joblib')

# 3. Xây dựng mô hình
model = Sequential([
    Input(shape=(X_train.shape[1],)), # 12 features
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# 5. Huấn luyện
print("dang huan luyen mo hinh...")
model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 6. Đánh giá và lưu
loss, acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nAccuracy Test: {acc*100:.2f}%")
model.save('pumpkin_model.keras')
print("Done!")