import streamlit as st
import requests
import pandas as pd
import io
import os

st.set_page_config(page_title="Pumpkin Seed Classifier", page_icon="🎃", layout="wide")

st.title("🎃 Hệ thống Dự đoán Hạt Bí ngô hàng loạt")
st.info("Hệ thống sử dụng toàn bộ 12 đặc trưng hình thái để dự đoán.")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_URL_FILE = f"{BACKEND_URL}/predict_file"

uploaded_file = st.file_uploader("Tải lên file dữ liệu (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            preview_df = pd.read_csv(uploaded_file)
        else:
            preview_df = pd.read_excel(uploaded_file)

        st.write("🔍 **Xem trước dữ liệu:**")
        st.dataframe(preview_df.head(), use_container_width=True)
        
        if st.button("🚀 Dự đoán hàng loạt"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            
            with st.spinner('Đang xử lý...'):
                try:
                    response = requests.post(API_URL_FILE, files=files)
                    if response.status_code == 200:
                        results_df = pd.DataFrame(response.json())
                        st.success("✅ Thành công!")
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Tải về CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")
                    else:
                        st.error(f"Lỗi: {response.text}")
                except Exception as e:
                    st.error(f"Không kết nối được server: {e}")
    except Exception as e:

        st.error(f"Lỗi file: {e}")

