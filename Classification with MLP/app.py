import streamlit as st
import requests
import pandas as pd
import io

# Cấu hình trang
st.set_page_config(
    page_title="Pumpkin Seed Batch Classifier", 
    page_icon="🎃", 
    layout="wide"
)

# Tùy chỉnh CSS để giao diện chuyên nghiệp hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎃 Hệ thống Dự đoán Hạt Bí ngô hàng loạt")
st.info("Hệ thống cho phép tải lên file dữ liệu (.csv, .xlsx) để dự đoán loại hạt (Ürgüp Sivrisi hoặc Çerçevelik) kèm độ tin cậy.")

# URL của Backend FastAPI
API_URL_FILE = "http://localhost:8000/predict_file"

# Khu vực Upload file
st.subheader("1. Tải lên tệp dữ liệu")
uploaded_file = st.file_uploader(
    "Chọn file CSV hoặc Excel (Yêu cầu đầy đủ 12 cột đặc trưng: Area, Perimeter, Major_Axis_Length, ...)", 
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    # Hiển thị xem trước dữ liệu đã upload
    try:
        if uploaded_file.name.endswith('.csv'):
            preview_df = pd.read_csv(uploaded_file)
        else:
            preview_df = pd.read_excel(uploaded_file)
        
        st.write("🔍 **Xem trước dữ liệu tải lên (5 dòng đầu):**")
        st.dataframe(preview_df.head(), use_container_width=True)
        
        # Nút nhấn dự đoán
        if st.button("🚀 Bắt đầu Dự đoán hàng loạt"):
            # Chuẩn bị file để gửi qua API
            # Đưa con trỏ file về đầu để đọc lại từ đầu sau khi preview
            uploaded_file.seek(0)
            
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if uploaded_file.name.endswith('.xlsx') else "text/csv"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime_type)}
            
            with st.spinner('Đang kết nối Server và xử lý dữ liệu...'):
                try:
                    response = requests.post(API_URL_FILE, files=files)
                    
                    if response.status_code == 200:
                        results_json = response.json()
                        results_df = pd.DataFrame(results_json)
                        
                        st.success("✅ Dự đoán thành công!")
                        
                        # Hiển thị kết quả
                        st.subheader("2. Kết quả dự đoán")
                        
                        # Highlight kết quả cho dễ nhìn
                        def highlight_class(val):
                            color = '#e1f5fe' if val == 'Ürgüp Sivrisi' else '#fff3e0'
                            return f'background-color: {color}'

                        # Hiển thị bảng kết quả với định dạng
                        st.dataframe(
                            results_df.style.applymap(highlight_class, subset=['Prediction']), 
                            use_container_width=True
                        )
                        
                        # Thống kê nhanh
                        col1, col2 = st.columns(2)
                        counts = results_df['Prediction'].value_counts()
                        col1.metric("Tổng số lượng hạt", len(results_df))
                        col2.write(counts)

                        # Nút tải về kết quả
                        st.subheader("3. Xuất dữ liệu")
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Tải xuống kết quả (.csv)",
                            data=csv_data,
                            file_name="pumpkin_seed_predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error(f"Lỗi từ Server: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Không thể kết nối tới Backend (FastAPI). Hãy đảm bảo server đang chạy tại port 8000.")

    except Exception as e:
        st.error(f"Lỗi định dạng file: {e}")

else:
    st.warning("Vui lòng tải lên một file để bắt đầu.")

# Hướng dẫn định dạng
with st.expander("📌 Yêu cầu về định dạng dữ liệu (Headers)"):
    st.write("""
    File của bạn cần chứa chính xác các cột sau (phân biệt hoa thường tùy vào model):
    - `Area`, `Perimeter`, `Major_Axis_Length`, `Minor_Axis_Length`, `Convex_Area`, `Equiv_Diameter`, 
    - `Eccentricity`, `Solidity`, `Extent`, `Roundness`, `Aspect_Ration`, `Compactness`.
    """)