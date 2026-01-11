from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import io
import os

app = FastAPI(title="Pumpkin Seed Classification API")

# --- LOAD ASSETS ---
# Khai báo biến toàn cục trước
model = None
scaler = None

# Kiểm tra sự tồn tại của file trước khi load
if os.path.exists('pumpkin_model.keras') and os.path.exists('scaler.joblib'):
    try:
        model = tf.keras.models.load_model('pumpkin_model.keras')
        scaler = joblib.load('scaler.joblib')
        print("Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading assets: {e}")
else:
    print("CRITICAL ERROR: File 'pumpkin_model.keras' or 'scaler.joblib' not found!")

class SeedData(BaseModel):
    Area: float
    Perimeter: float
    Major_Axis_Length: float
    Minor_Axis_Length: float
    Convex_Area: float
    Equiv_Diameter: float
    Eccentricity: float
    Solidity: float
    Extent: float
    Roundness: float
    Aspect_Ration: float
    Compactness: float

def preprocess_input(df: pd.DataFrame):
    # Kiểm tra xem scaler đã được load chưa
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler is not initialized. Check server logs.")
    
    # Thực hiện scale dữ liệu
    scaled_data = scaler.transform(df)
    return scaled_data

@app.get("/")
async def root():
    return {"status": "Server is running", "message": "Pumpkin Seed API is alive!"}
@app.post("/predict")
async def predict(data: SeedData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        processed_data = preprocess_input(input_df)
        prediction = model.predict(processed_data)
        
        prob_class_1 = float(prediction[0][0])
        class_name = "Urgup Sivrisi" if prob_class_1 > 0.5 else "Cercevelik"
        confidence = prob_class_1 if prob_class_1 > 0.5 else 1 - prob_class_1
        
        return {
            "prediction": class_name,
            "confidence": f"{confidence * 100:.2f}%",
            "class_id": 1 if prob_class_1 > 0.5 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "Server is running", "message": "Pumpkin Seed API is alive!"}
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
    try:
        contents = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        # Danh sách 12 cột đặc trưng chuẩn
        feature_cols = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length', 
                        'Convex_Area', 'Equiv_Diameter', 'Eccentricity', 'Solidity', 
                        'Extent', 'Roundness', 'Aspect_Ration', 'Compactness']
        
        # Kiểm tra xem file upload có đủ cột không
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")

        input_data = df[feature_cols]
        processed_data = preprocess_input(input_data)
        predictions = model.predict(processed_data)
        
        results = []
        confidences = []
        for p in predictions:
            prob_val = float(p[0])
            if prob_val > 0.5:
                results.append("Urgup Sivrisi")
                confidences.append(f"{prob_val * 100:.2f}%")
            else:
                results.append("Cercevelik")
                confidences.append(f"{(1 - prob_val) * 100:.2f}%")
        
        df['Prediction'] = results
        df['Confidence'] = confidences
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
