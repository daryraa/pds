import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan encoder
model = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Prediksi Risiko Dropout Mahasiswa")
st.write("Silakan isi informasi di bawah ini untuk memprediksi apakah mahasiswa berisiko dropout, lulus, atau masih terdaftar.")

# Form input
approved_2nd = st.number_input("Jumlah mata kuliah lulus semester 2 (maks: 20)", min_value=0, max_value=20, step=1)
approved_1st = st.number_input("Jumlah mata kuliah lulus semester 1 (maks: 20)", min_value=0, max_value=20, step=1)
tuition_paid = st.selectbox("Apakah pembayaran kuliah lancar?", ["Yes", "No"])
grade_2nd = st.number_input("Rata-rata nilai semester 2 (0.0 - 20.0)", min_value=0.0, max_value=20.0, step=0.1)
scholarship = st.selectbox("Apakah menerima beasiswa?", ["Yes", "No"])
enrolled_1st = st.number_input("Jumlah mata kuliah diambil semester 1 (maks: 20)", min_value=0, max_value=20, step=1)
age = st.number_input("Umur saat masuk kuliah (kisaran umum 17 - 40)", min_value=15, max_value=70, step=1)
gdp = st.number_input("GDP negara asal", step=0.01)
app_order = st.number_input("Urutan pilihan jurusan (1 = pilihan pertama, maks 10)", min_value=1, max_value=10, step=1)
gender = st.selectbox("Jenis kelamin", ["Male", "Female"])
unemployment = st.number_input("Tingkat pengangguran saat itu (%)", step=0.1)
debtor = st.selectbox("Apakah memiliki utang pendidikan?", ["Yes", "No"])
no_eval_1st = st.number_input("Jumlah MK tanpa evaluasi semester 1 (maks: 12)", min_value=0, max_value=12, step=1)
no_eval_2nd = st.number_input("Jumlah MK tanpa evaluasi semester 2 (maks: 12)", min_value=0, max_value=12, step=1)
special_needs = st.selectbox("Apakah memiliki kebutuhan khusus?", ["Yes", "No"])

# Encoding categorical
def encode_binary(val):
    return 1 if val == "Yes" or val == "Male" else 0

data_input = pd.DataFrame({
    'Curricular_units_2nd_sem_approved': [approved_2nd],
    'Curricular_units_1st_sem_approved': [approved_1st],
    'Tuition_fees_up_to_date': [encode_binary(tuition_paid)],
    'Curricular_units_2nd_sem_grade': [grade_2nd],
    'Scholarship_holder': [encode_binary(scholarship)],
    'Curricular_units_1st_sem_enrolled': [enrolled_1st],
    'Age_at_enrollment': [age],
    'GDP': [gdp],
    'Application_order': [app_order],
    'Gender': [encode_binary(gender)],
    'Unemployment_rate': [unemployment],
    'Debtor': [encode_binary(debtor)],
    'Curricular_units_1st_sem_without_evaluations': [no_eval_1st],
    'Curricular_units_2nd_sem_without_evaluations': [no_eval_2nd],
    'Educational_special_needs': [encode_binary(special_needs)]
})

if st.button("Prediksi Status Mahasiswa"):
    pred = model.predict(data_input)
    label = label_encoder.inverse_transform(pred)[0]
    st.success(f"Prediksi status mahasiswa: **{label}**")
