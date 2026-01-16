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
# TAB 1: DỰ ĐOÁN ĐƠN LẺ (REAL-TIME VALIDATION)
# ==========================================
with tab1:
    st.header("Nhập thông số kỹ thuật của hạt")
    st.info("💡 Nút 'Phân loại' sẽ bị khóa cho đến khi tất cả thông số hợp lệ.")

    # --- 1. NHẬP LIỆU (Bỏ st.form để cập nhật tức thì) ---
    col1, col2, col3 = st.columns(3)

    # Để validate bằng code logic (disable button), ta nới rộng min/max của widget
    # để người dùng có thể nhập sai, sau đó ta bắt lỗi và khóa nút.
    
    with col1:
        st.subheader("Kích thước cơ bản")
        area = st.number_input("Area", value=80000.0, step=100.0, min_value=0.0)
        perimeter = st.number_input("Perimeter", value=1200.0, step=10.0, min_value=0.0)
        major_axis = st.number_input("Major_Axis_Length", value=500.0, step=10.0, min_value=0.0)
        minor_axis = st.number_input("Minor_Axis_Length", value=250.0, step=10.0, min_value=0.0)

    with col2:
        st.subheader("Diện tích & Đường kính")
        convex_area = st.number_input("Convex_Area", value=81000.0, step=100.0, min_value=0.0)
        equiv_diameter = st.number_input("Equiv_Diameter", value=300.0, step=10.0, min_value=0.0)
        
        # Các chỉ số bé
        eccentricity = st.number_input("Eccentricity", value=0.8500, step=0.0001, format="%.4f")
        solidity = st.number_input("Solidity (Độ đặc)", value=0.9850, step=0.0001, format="%.4f")

    with col3:
        st.subheader("Hệ số hình dạng")
        extent = st.number_input("Extent", value=0.7000, step=0.0001, format="%.4f")
        roundness = st.number_input("Roundness", value=0.8000, step=0.0001, format="%.4f")
        aspect_ratio = st.number_input("Aspect_Ration", value=2.0000, step=0.0001, format="%.4f")
        compactness = st.number_input("Compactness", value=0.7000, step=0.0001, format="%.4f")

    # --- 2. LOGIC KIỂM TRA (VALIDATION) ---
    # Kiểm tra ngay lập tức các giá trị vừa nhập
    errors = []

    # Nhóm 1
    if not (40000 < area < 145000): errors.append(f"Area: {area} (Phải từ 40,000 - 145,000)")
    if not (800 < perimeter < 1600): errors.append(f"Perimeter: {perimeter} (Phải từ 800 - 1,600)")
    if not (300 < major_axis < 700): errors.append(f"Major Axis: {major_axis} (Phải từ 300 - 700)")
    if not (140 < minor_axis < 350): errors.append(f"Minor Axis: {minor_axis} (Phải từ 140 - 350)")

    # Nhóm 2
    if not (40000 < convex_area < 145000): errors.append(f"Convex Area: {convex_area} (Phải từ 40,000 - 145,000)")
    if not (0 < equiv_diameter < 430): errors.append(f"Equiv Diameter: {equiv_diameter} (Phải từ 0 - 430)")

    # Nhóm 3 (0 < x < 1)
    if not (0 < eccentricity < 1): errors.append(f"Eccentricity: {eccentricity} (Phải < 1)")
    if not (0 < solidity < 1): errors.append(f"Solidity: {solidity} (Phải < 1)")
    if not (0 < extent < 1): errors.append(f"Extent: {extent} (Phải < 1)")
    if not (0 < roundness < 1): errors.append(f"Roundness: {roundness} (Phải < 1)")
    if not (0 < compactness < 1): errors.append(f"Compactness: {compactness} (Phải < 1)")
    
    # Aspect Ratio
    if not (0 < aspect_ratio < 3.5): errors.append(f"Aspect Ration: {aspect_ratio} (Phải < 3.5)")

    # --- 3. HIỂN THỊ LỖI VÀ NÚT BẤM ---
    
    # Biến cờ kiểm tra hợp lệ
    is_valid = len(errors) == 0

    if not is_valid:
        st.error("⛔ Phát hiện dữ liệu không hợp lệ:")
        for err in errors:
            st.warning(err)
    else:
        st.success("✅ Dữ liệu hợp lệ. Sẵn sàng phân loại!")

    # Nút bấm: disabled=True nếu dữ liệu không hợp lệ (not is_valid)
    btn_predict = st.button("🚀 Phân loại ngay", type="primary", disabled=not is_valid)

    if btn_predict:
        # Gom dữ liệu
        payload = {
            "Area": area, "Perimeter": perimeter, "Major_Axis_Length": major_axis,
            "Minor_Axis_Length": minor_axis, "Convex_Area": convex_area,
            "Equiv_Diameter": equiv_diameter, "Eccentricity": eccentricity,
            "Solidity": solidity, "Extent": extent, "Roundness": roundness,
            "Aspect_Ration": aspect_ratio, "Compactness": compactness
        }

        with st.spinner("Đang gửi tới AI..."):
            try:
                response = requests.post(API_URL_PREDICT, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Phân loại thành công!")
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Loại hạt", result.get("prediction", "Unknown"))
                    with col_res2:
                        st.metric("Độ tin cậy", result.get("confidence", "0%"))
                else:
                    st.error(f"❌ Server trả về lỗi ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"❌ Lỗi kết nối: {e}")

# ==========================================
# TAB 2: DỰ ĐOÁN HÀNG LOẠT (Giữ nguyên)
# ==========================================
with tab2:
    st.header("Tải lên file dữ liệu")
    uploaded_file = st.file_uploader("Chọn file (.csv, .xlsx)", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                preview_df = pd.read_csv(uploaded_file)
            else:
                preview_df = pd.read_excel(uploaded_file)

            st.write("🔍 **Xem trước dữ liệu (5 dòng đầu):**")
            st.dataframe(preview_df.head(), use_container_width=True)
            
            if st.button("🚀 Xử lý toàn bộ file"):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                
                with st.spinner('Đang xử lý hàng loạt...'):
                    try:
                        response = requests.post(API_URL_FILE, files=files, timeout=30)
                        if response.status_code == 200:
                            results_df = pd.DataFrame(response.json())
                            st.success("✅ Xử lý hoàn tất!")
                            st.dataframe(results_df, use_container_width=True)
                            
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
