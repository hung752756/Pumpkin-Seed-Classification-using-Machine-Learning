import streamlit as st
import requests
import pandas as pd
import io
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Pumpkin Seed Classifier", page_icon="🎃", layout="wide")

st.title("🎃 Hệ thống Phân loại Hạt Bí ngô")
st.markdown("---")

# --- CẤU HÌNH API ---
# Lấy URL từ biến môi trường hoặc dùng mặc định localhost
raw_url = os.getenv("BACKEND_URL", "http://localhost:8000").strip().rstrip('/')
BACKEND_URL = raw_url
API_URL_PREDICT = f"{BACKEND_URL}/predict"
API_URL_FILE = f"{BACKEND_URL}/predict_file"

# --- SIDEBAR ---
st.sidebar.title("⚙️ Cấu hình hệ thống")
st.sidebar.info(f"Đang kết nối Backend tại:\n`{BACKEND_URL}`")

if st.sidebar.button("Kiểm tra kết nối Server"):
    try:
        response = requests.get(BACKEND_URL, timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ Kết nối tới Backend thành công!")
        else:
            st.sidebar.error(f"❌ Backend trả về lỗi: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"❌ Không thể kết nối: {e}")

# --- GIAO DIỆN CHÍNH (TABS) ---
tab1, tab2 = st.tabs(["🧩 Dự đoán Đơn lẻ (Nhập tay)", "📂 Dự đoán Hàng loạt (Upload File)"])

# ==========================================
# TAB 1: DỰ ĐOÁN ĐƠN LẺ (Gửi tới /predict)
# ==========================================
with tab1:
    st.header("Nhập thông số kỹ thuật của hạt")
    st.write("Vui lòng nhập 12 đặc trưng hình thái để phân loại.")

    # Tạo Form để gom nhóm input
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        # Nhóm 1: Các chỉ số kích thước lớn
        with col1:
            st.subheader("Kích thước cơ bản")
            # Các giá trị min/max/value dựa trên Pydantic constraints trong Backend
            area = st.number_input("Area (Diện tích)", min_value=40001.0, max_value=144999.0, value=80000.0, step=100.0)
            perimeter = st.number_input("Perimeter (Chu vi)", min_value=801.0, max_value=1599.0, value=1200.0)
            major_axis = st.number_input("Major_Axis_Length (Trục lớn)", min_value=301.0, max_value=699.0, value=500.0)
            minor_axis = st.number_input("Minor_Axis_Length (Trục nhỏ)", min_value=141.0, max_value=349.0, value=250.0)

        # Nhóm 2: Diện tích và Đường kính
        with col2:
            st.subheader("Diện tích & Đường kính")
            convex_area = st.number_input("Convex_Area (Diện tích bao lồi)", min_value=40001.0, max_value=144999.0, value=81000.0, step=100.0)
            equiv_diameter = st.number_input("Equiv_Diameter (ĐK tương đương)", min_value=0.1, max_value=429.0, value=300.0)
            eccentricity = st.number_input("Eccentricity (Độ tâm sai)", min_value=0.01, max_value=0.99, value=0.8, format="%.4f")
            solidity = st.number_input("Solidity (Độ đặc)", min_value=0.01, max_value=0.99, value=0.9, format="%.4f")

        # Nhóm 3: Các hệ số hình dạng (0-1 hoặc nhỏ)
        with col3:
            st.subheader("Hệ số hình dạng")
            extent = st.number_input("Extent (Độ mở rộng)", min_value=0.01, max_value=0.99, value=0.7, format="%.4f")
            roundness = st.number_input("Roundness (Độ tròn)", min_value=0.01, max_value=0.99, value=0.8, format="%.4f")
            # Lưu ý: Backend bạn ghi là Aspect_Ration (thiếu chữ 'o' ở cuối nhưng khớp model pydantic)
            aspect_ratio = st.number_input("Aspect_Ration (Tỷ lệ khung hình)", min_value=0.01, max_value=3.49, value=2.0, format="%.4f")
            compactness = st.number_input("Compactness (Độ nén)", min_value=0.01, max_value=0.99, value=0.7, format="%.4f")

        submitted = st.form_submit_button("🚀 Phân loại ngay")

    if submitted:
        # Chuẩn bị payload JSON gửi đi (Key phải khớp 100% với Pydantic BaseModel trong FastAPI)
        payload = {
            "Area": area,
            "Perimeter": perimeter,
            "Major_Axis_Length": major_axis,
            "Minor_Axis_Length": minor_axis,
            "Convex_Area": convex_area,
            "Equiv_Diameter": equiv_diameter,
            "Eccentricity": eccentricity,
            "Solidity": solidity,
            "Extent": extent,
            "Roundness": roundness,
            "Aspect_Ration": aspect_ratio, # Khớp với Backend
            "Compactness": compactness
        }

        with st.spinner("Đang gửi dữ liệu tới AI..."):
            try:
                response = requests.post(API_URL_PREDICT, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    # Hiển thị kết quả đẹp
                    st.success("✅ Dự đoán thành công!")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric(label="Loại hạt dự đoán", value=result.get("prediction", "Unknown"))
                    with col_res2:
                        st.metric(label="Độ tin cậy", value=result.get("confidence", "0%"))
                        
                else:
                    st.error(f"❌ Lỗi từ Server ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ Không kết nối được server: {e}")

# ==========================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (Gửi tới /predict_file) - Code cũ của bạn
# ==========================================
with tab2:
    st.header("Tải lên file dữ liệu")
    uploaded_file = st.file_uploader("Chọn file (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Đọc file để preview
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)

            st.write("🔍 **Xem trước dữ liệu (5 dòng đầu):**")
            st.dataframe(preview_df.head(), use_container_width=True)
            
            if st.button("🚀 Xử lý toàn bộ file"):
                # Reset con trỏ file về đầu để gửi request
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                
                with st.spinner('Đang xử lý hàng loạt...'):
                    try:
                        response = requests.post(API_URL_FILE, files=files, timeout=30)
                        if response.status_code == 200:
                            results_df = pd.DataFrame(response.json())
                            st.success("✅ Xử lý hoàn tất!")
                            
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Chuyển đổi để download
                            csv_data = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Tải về kết quả (CSV)", 
                                data=csv_data, 
                                file_name="prediction_results.csv", 
                                mime="text/csv"
                            )
                        else:
                            st.error(f"Lỗi từ server: {response.text}")
                    except Exception as e:
                        st.error(f"Không kết nối được server: {e}")
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")
