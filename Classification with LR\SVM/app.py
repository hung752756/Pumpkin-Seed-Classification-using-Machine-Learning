import streamlit as st
import pandas as pd
import requests
import io

# Cấu hình trang
st.set_page_config(page_title="Hệ thống Phân loại Hạt Bí", layout="wide", page_icon="🎃")

st.title("Dự đoán Loại Hạt Bí Hàng Loạt")
st.markdown("""
Ứng dụng này cho phép bạn tải lên file dữ liệu chứa các thông số kỹ thuật của hạt bí để phân loại tự động thành 
**Ürgüp Sivrisi** hoặc **Çerçevelik**.
""")

# 1. Định nghĩa danh sách 12 cột bắt buộc mà Model/Pipeline yêu cầu
REQUIRED_COLUMNS = [
    "Area", "Perimeter", "Major_Axis_Length", "Minor_Axis_Length",
    "Convex_Area", "Equiv_Diameter", "Eccentricity", "Solidity",
    "Extent", "Roundness", "Aspect_Ration", "Compactness"
]

# 2. Thành phần Upload file (Hỗ trợ CSV và XLSX)
uploaded_file = st.file_uploader("Tải file dữ liệu của bạn (CSV hoặc Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Kiểm tra đuôi file để dùng hàm đọc phù hợp
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Đã tải file thành công: {uploaded_file.name}")
        
        # Hiển thị bản xem trước dữ liệu
        st.subheader("Bản xem trước dữ liệu (5 dòng đầu):")
        st.dataframe(df.head())

        # 3. Kiểm tra tính hợp lệ của các cột
        # Chuyển tên cột về dạng chuẩn (xóa khoảng trắng thừa) để tránh lỗi so khớp
        df.columns = [col.strip() for col in df.columns]
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

        if not missing_cols:
            st.info("✅ File có đầy đủ các cột cần thiết. Sẵn sàng dự đoán.")
            
            if st.button("🚀 Tiến hành Dự đoán Hàng loạt"):
                # Chuẩn bị dữ liệu để gửi lên API (chỉ lấy 12 cột yêu cầu)
                data_to_send = df[REQUIRED_COLUMNS].to_dict(orient="records")
                
                with st.spinner('Đang kết nối với Server để xử lý...'):
                    try:
                        # Gửi request POST đến FastAPI
                        # Lưu ý: Địa chỉ này phải khớp với địa chỉ uvicorn đang chạy
                        API_URL = "http://127.0.0.1:8000/predict_batch"
                        response = requests.post(API_URL, json=data_to_send, timeout=30)
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            # 4. Gộp kết quả dự đoán vào DataFrame hiện tại
                            df['Dự đoán'] = results['predictions']
                            df['Độ tin cậy'] = results['probabilities']
                            
                            st.divider()
                            st.subheader("📊 Kết quả dự đoán:")
                            
                            # Hiển thị bảng kết quả với màu sắc (tùy chọn)
                            st.dataframe(df)

                            # 5. Tạo file Excel để người dùng tải về
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False)
                            
                            st.download_button(
                                label="📥 Tải xuống kết quả Full (.xlsx)",
                                data=output.getvalue(),
                                file_name="ket_qua_du_doan_hat_bi.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.error(f"❌ Lỗi từ Server (Mã lỗi: {response.status_code})")
                            st.write(response.text)

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Không thể kết nối với Server API. Hãy đảm bảo file 'app.py' đang chạy (uvicorn).")
        else:
            st.error(f"❌ File thiếu các cột sau: {', '.join(missing_cols)}")
            st.warning("Vui lòng kiểm tra lại định dạng file. Tên cột phải khớp chính xác tuyệt đối.")
            
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi khi đọc file: {e}")

# Hướng dẫn nhỏ ở cuối trang
st.sidebar.header("Hướng dẫn")
st.sidebar.write("""
1. File của bạn cần có ít nhất 12 cột dữ liệu kỹ thuật.
2. Hệ thống sẽ tự động xử lý các giá trị ngoại lệ (Outliers) và chuẩn hóa dữ liệu.
3. Nhấn 'Dự đoán' và đợi trong giây lát.
""")