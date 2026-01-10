from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import List
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from my_transformers import OutlierHandler, CorrelationDropper


app = FastAPI()

# Mô hình phân loại Logistic Regression
#model_pipeline = joblib.load('model_pipeline.pkl')

#Mô hình phân loại SVM
model_pipeline = joblib.load('model_svm_pipeline.pkl')

class SeedData(BaseModel): # Schema dữ liệu đầu vào

    Area: float = Field(..., gt=0, description="Diện tích hạt phải lớn hơn 0")
    Perimeter: float = Field(..., gt=0, description="Chu vi hạt phải lớn hơn 0")
    Major_Axis_Length: float = Field(..., gt=0, description="Chiều dài trục chính phải lớn hơn 0")
    Minor_Axis_Length: float = Field(..., gt=0, description="Chiều dài trục phụ phải lớn hơn 0")
    Convex_Area: float = Field(..., gt=0, description="Diện tích lồi phải lớn hơn 0")
    Equiv_Diameter: float = Field(..., gt=0, description="Đường kính tương đương phải lớn hơn 0")
    Eccentricity: float = Field(..., gt=0, lt=1, description="Độ lệch tâm phải nằm trong khoảng (0, 1)")
    Solidity: float = Field(..., gt=0, lt=1, description="Độ đặc phải nằm trong khoảng (0, 1)")
    Extent: float = Field(..., gt=0, lt=1, description="Phạm vi phải nằm trong khoảng (0, 1)")
    Roundness: float = Field(..., gt=0, lt=1, description="Độ tròn phải nằm trong khoảng (0, 1)")
    Aspect_Ration: float = Field(..., gt=0, lt=4, description="Tỷ lệ khía cạnh phải nằm trong khoảng (0, 4)")
    Compactness: float = Field(..., gt=0, lt=1, description="Độ gọn phải nằm trong khoảng (0, 1)")

@app.post("/predict_batch")
def predict_batch(data: List[SeedData]):
    # 1. Chuyển List Pydantic sang DataFrame
    df = pd.DataFrame([item.model_dump() for item in data])
    
    # 2. Dự đoán (Pipeline tự xử lý Outlier và Dropper bên trong)
    predictions = model_pipeline.predict(df)
    probabilities = model_pipeline.predict_proba(df).max(axis=1)
    
    # 3. Mapping nhãn
    class_mapping = {1: "Ürgüp Sivrisi", 0: "Çerçevelik"}
    results = [class_mapping.get(p, p) for p in predictions]
    
    return {
        "predictions": results,
        "probabilities": [f"{round(p * 100, 2)}%" for p in probabilities]
    }