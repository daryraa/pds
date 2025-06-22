import streamlit as st
import pandas as pd
import joblib

# Load model dan encoder
model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Risiko Dropout Mahasiswa")
st.write("Isi data berikut sesuai dengan kondisi mahasiswa.")

# Mapping kategori (disesuaikan dengan data numerik training)
course_map = {
    'Biofuel Production Technologies': 33,
    'Animation and Multimedia Design': 171,
    'Social Service (evening attendance)': 8014,
    'Agronomy': 9003,
    'Communication Design': 9070,
    'Veterinary Nursing': 9085,
    'Informatics Engineering': 9119,
    'Equinculture': 9130,
    'Management': 9147,
    'Social Service': 9238,
    'Tourism': 9254,
    'Nursing': 9500,
    'Oral Hygiene': 9556,
    'Advertising and Marketing Management': 9670,
    'Journalism and Communication': 9773,
    'Basic Education': 9853,
    'Management (evening attendance)': 9991
}

app_mode_map = {
    "1st phase - general contingent": 1,
    "Ordinance No. 612/93": 2,
    "1st phase - special contingent (Azores Island)": 5,
    "Holders of other higher courses": 7,
    "Ordinance No. 854-B/99": 10,
    "International student (bachelor)": 15,
    "1st phase - special contingent (Madeira Island)": 16,
    "2nd phase - general contingent": 17,
    "3rd phase - general contingent": 18,
    "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
    "Ordinance No. 533-A/99, item b3 (Other Institution)": 27,
    "Over 23 years old": 39,
    "Transfer": 42,
    "Change of course": 43,
    "Technological specialization diploma holders": 44,
    "Change of institution/course": 51,
    "Short cycle diploma holders": 53,
    "Change of institution/course (International)": 57
}

# Form input
approved_2nd = st.number_input("Jumlah MK Lulus Semester 2", min_value=0, max_value=20, step=1)
approved_1st = st.number_input("Jumlah MK Lulus Semester 1", min_value=0, max_value=20, step=1)
tuition_paid = st.selectbox("Pembayaran Kuliah Lancar", ["Yes", "No"])
grade_2nd = st.number_input("Nilai Rata-rata Semester 2", min_value=0.0, max_value=20.0, step=0.1)
enrolled_2nd = st.number_input("Jumlah MK Diambil Semester 2", min_value=0, max_value=20, step=1)
course_text = st.selectbox("Program Studi", list(course_map.keys()))
enrolled_1st = st.number_input("Jumlah MK Diambil Semester 1", min_value=0, max_value=20, step=1)
debtor = st.selectbox("Memiliki Utang Pendidikan", ["Yes", "No"])
eval_1st = st.number_input("Jumlah Evaluasi Semester 1", min_value=0, max_value=20, step=1)
scholarship = st.selectbox("Menerima Beasiswa", ["Yes", "No"])
gdp = st.number_input("GDP Negara Asal", step=0.01)
credited_1st = st.number_input("Jumlah MK Disetarakan Semester 1", min_value=0, max_value=20, step=1)
grade_1st = st.number_input("Nilai Rata-rata Semester 1", min_value=0.0, max_value=20.0, step=0.1)
app_mode_text = st.selectbox("Mode Aplikasi", list(app_mode_map.keys()))
no_eval_1st = st.number_input("Jumlah MK tanpa Evaluasi Semester 1", min_value=0, max_value=12, step=1)

# Encoder untuk binary
def encode_binary(val):
    return 1 if val == "Yes" else 0

# Data input
input_data = pd.DataFrame({
    'Curricular_units_2nd_sem_approved': [approved_2nd],
    'Curricular_units_1st_sem_approved': [approved_1st],
    'Tuition_fees_up_to_date': [encode_binary(tuition_paid)],
    'Curricular_units_2nd_sem_grade': [grade_2nd],
    'Curricular_units_2nd_sem_enrolled': [enrolled_2nd],
    'Course': [course_map[course_text]],
    'Curricular_units_1st_sem_enrolled': [enrolled_1st],
    'Debtor': [encode_binary(debtor)],
    'Curricular_units_1st_sem_evaluations': [eval_1st],
    'Scholarship_holder': [encode_binary(scholarship)],
    'GDP': [gdp],
    'Curricular_units_1st_sem_credited': [credited_1st],
    'Curricular_units_1st_sem_grade': [grade_1st],
    'Application_mode': [app_mode_map[app_mode_text]],
    'Curricular_units_1st_sem_without_evaluations': [no_eval_1st]
})

# Prediksi
if st.button("Prediksi Status Mahasiswa"):
    pred = model.predict(input_data)
    label = label_encoder.inverse_transform(pred)[0]
    st.success(f"Prediksi status mahasiswa: **{label}**")
