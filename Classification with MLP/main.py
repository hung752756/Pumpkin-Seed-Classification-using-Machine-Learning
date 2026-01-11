from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from typing import List
import io

app = FastAPI(title="Pumpkin Seed Classification API")

# --- LOAD MODEL VÀ PREPROCESSING ASSETS ---
try:
    model = tf.keras.models.load_model('pumpkin_model.keras')
    pre_data = joblib.load('preprocessing_bundle.joblib')
    
    scaler = pre_data['scaler']
    to_drop = pre_data['to_drop']
    cols_outliers = pre_data['cols_outliers']
    outlier_params = pre_data['outlier_params']
    
    input_features = cols_outliers 
except Exception as e:
    print(f"Error loading assets: {e}")

class SeedData(BaseModel):
    Area: float = Field(..., gt=0)
    Perimeter: float = Field(..., gt=0)
    Major_Axis_Length: float = Field(..., gt=0)
    Minor_Axis_Length: float = Field(..., gt=0)
    Convex_Area: float = Field(..., gt=0)
    Equiv_Diameter: float = Field(..., gt=0)
    Eccentricity: float = Field(..., gt=0, lt=1)
    Solidity: float = Field(..., gt=0, lt=1)
    Extent: float = Field(..., gt=0, lt=1)
    Roundness: float = Field(..., gt=0, lt=1)
    Aspect_Ration: float = Field(..., gt=0, lt=4)
    Compactness: float = Field(..., gt=0, lt=1)

def preprocess_input(df: pd.DataFrame):
    df_processed = df.copy().astype(float)
    for col in cols_outliers:
        if col in df_processed.columns:
            lower = outlier_params[col]['lower']
            upper = outlier_params[col]['upper']
            median = outlier_params[col]['median']
            mask = (df_processed[col] < lower) | (df_processed[col] > upper)
            df_processed.loc[mask, col] = median
            
    df_processed = df_processed.drop(columns=to_drop)
    scaled_data = scaler.transform(df_processed)
    return scaled_data

@app.post("/predict")
async def predict(data: SeedData):
    try:
        input_df = pd.DataFrame([data.model_dump()])
        processed_data = preprocess_input(input_df)
        
        prediction = model.predict(processed_data)
        prob_class_1 = float(prediction[0][0])
        
        # Tính toán class và phần trăm tin cậy cao nhất
        if prob_class_1 > 0.5:
            class_name = "Ürgüp Sivrisi"
            confidence = prob_class_1
            class_idx = 1
        else:
            class_name = "Çerçevelik"
            confidence = 1 - prob_class_1
            class_idx = 0
        
        return {
            "prediction": class_name,
            "confidence": f"{confidence * 100:.2f}%", # Hiển thị %
            "probability_raw": prob_class_1,
            "class_id": class_idx
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    try:
        contents = await file.read()
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Định dạng file không hỗ trợ.")
        
        processed_data = preprocess_input(df)
        predictions = model.predict(processed_data)
        
        results = []
        confidences = []
        
        for p in predictions:
            prob_val = float(p[0])
            if prob_val > 0.5:
                results.append("Ürgüp Sivrisi")
                confidences.append(f"{prob_val * 100:.2f}%")
            else:
                results.append("Çerçevelik")
                confidences.append(f"{(1 - prob_val) * 100:.2f}%")
        
        # Thêm cột kết quả vào dataframe
        df['Prediction'] = results
        df['Confidence'] = confidences
        
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)