
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide"
)

@st.cache_resource
def cargar_recursos():
    modelo_prob = tf.keras.models.load_model('modelo_probabilidad.keras')
    modelos_ensemble = []
    for i in range(1, 6):
        modelos_ensemble.append(tf.keras.models.load_model(f'modelo_ensemble_{i}.keras'))
    scaler = joblib.load('scaler_stroke.pkl')
    with open('columnas_stroke.pkl', 'rb') as f:
        columnas = pickle.load(f)
    return modelo_prob, modelos_ensemble, scaler, columnas

modelo_prob, modelos_ensemble, scaler, columnas = cargar_recursos()

st.title("🫀 Predicción de Riesgo Cardiovascular")
st.markdown("---")
st.write("""
Esta aplicación utiliza **redes neuronales artificiales** para estimar:
- La **probabilidad** de riesgo cardiovascular (regresión)
- La **categoría** de riesgo (Bajo / Medio / Alto)
""")
st.markdown("---")

st.header("Datos del Paciente")

with st.form("formulario_paciente"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Edad (años)", min_value=1, max_value=120, value=50, step=1)
        gender = st.selectbox("Sexo", options=["Male", "Female"])
        bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        hypertension = st.selectbox("¿Hipertensión?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
    
    with col2:
        heart_disease = st.selectbox("¿Enfermedad cardíaca?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        avg_glucose = st.number_input("Glucosa promedio (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.1)
        ever_married = st.selectbox("¿Casado/a?", options=["Yes", "No"])
        work_type = st.selectbox("Tipo de trabajo", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    
    with col3:
        residence = st.selectbox("Tipo de residencia", options=["Urban", "Rural"])
        smoking = st.selectbox("Estado de tabaquismo", options=["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    st.markdown("---")
    boton = st.form_submit_button("Predecir", use_container_width=True)

if boton:
    datos_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose,
        'bmi': bmi,
        'gender_Male': 1 if gender == "Male" else 0,
        'ever_married_Yes': 1 if ever_married == "Yes" else 0,
        'work_type_Never_worked': 1 if work_type == "Never_worked" else 0,
        'work_type_Private': 1 if work_type == "Private" else 0,
        'work_type_Self-employed': 1 if work_type == "Self-employed" else 0,
        'work_type_children': 1 if work_type == "children" else 0,
        'Residence_type_Urban': 1 if residence == "Urban" else 0,
        'smoking_status_formerly smoked': 1 if smoking == "formerly smoked" else 0,
        'smoking_status_never smoked': 1 if smoking == "never smoked" else 0,
        'smoking_status_smokes': 1 if smoking == "smokes" else 0,
    }
    
    df_input = pd.DataFrame([datos_dict])
    df_input = df_input.reindex(columns=columnas, fill_value=0)
    datos_escalados = scaler.transform(df_input)
    
    prob = modelo_prob.predict(datos_escalados, verbose=0)[0][0]
    porcentaje = prob * 100
    
    predicciones_ens = []
    for m in modelos_ensemble:
        predicciones_ens.append(m.predict(datos_escalados, verbose=0))
    promedio_ens = np.mean(predicciones_ens, axis=0)
    cat_idx = np.argmax(promedio_ens, axis=1)[0]
    categorias = ['Bajo', 'Medio', 'Alto']
    categoria = categorias[cat_idx]
    confianza = promedio_ens[0][cat_idx] * 100
    
    st.markdown("---")
    st.header("Resultado de la Predicción")
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.metric(label="Probabilidad de Riesgo", value=f"{porcentaje:.2f}%")
    
    with col_r2:
        if categoria == "Bajo":
            st.success(f"RIESGO {categoria.upper()} (confianza: {confianza:.1f}%)")
        elif categoria == "Medio":
            st.warning(f"RIESGO {categoria.upper()} (confianza: {confianza:.1f}%)")
        else:
            st.error(f"RIESGO {categoria.upper()} (confianza: {confianza:.1f}%)")
    
    st.progress(float(prob))
    
    st.markdown("### Interpretación")
    if categoria == "Bajo":
        st.info("El paciente tiene bajo riesgo cardiovascular. Se recomienda mantener hábitos saludables.")
    elif categoria == "Medio":
        st.warning("El paciente tiene riesgo moderado. Se recomienda consulta médica y seguimiento.")
    else:
        st.error("El paciente tiene alto riesgo cardiovascular. Se recomienda evaluación médica inmediata.")
    
    with st.expander("Ver datos ingresados"):
        st.dataframe(df_input.T.rename(columns={0: "Valor"}))

st.markdown("---")
st.caption("Proyecto académico - Universidad de Córdoba - Ingeniería de Sistemas - 2026-2")
