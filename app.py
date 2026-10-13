import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import base64

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular",
    page_icon=None,
    layout="wide"
)

# ==========================================
# FONDO CON IMAGEN + OVERLAY
# ==========================================
with open("fondo1.jpg", "rb") as f:
    img_data = f.read()
img_base64 = base64.b64encode(img_data).decode()

st.markdown(f"""
<style>
    .stApp {{
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
            url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    h1, h2, h3, p, label, .stMarkdown, .stCaption {{
        color: #ffffff !important;
    }}
    
    .stNumberInput label, .stSelectbox label {{
        color: #ffffff !important;
        font-weight: bold;
    }}
    
    .stNumberInput input, .stSelectbox select {{
        background-color: rgba(255, 255, 255, 0.92) !important;
        color: #000000 !important;
    }}
    
    .stMetric label, .stMetric div {{
        color: #ffffff !important;
    }}
    
    .stProgress > div > div > div > div {{
        background-color: #4CAF50 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGAR MODELOS Y RECURSOS
# ==========================================
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

# ==========================================
# TÍTULO
# ==========================================
st.title("Predicción de Riesgo Cardiovascular")
st.markdown("---")
st.write("""
Esta aplicación utiliza redes neuronales artificiales para estimar:
- La probabilidad de riesgo cardiovascular (regresión)
- La categoría de riesgo (Bajo / Medio / Alto)

Complete todos los campos del formulario y presione Predecir.
""")
st.markdown("---")

# ==========================================
# FORMULARIO
# ==========================================
st.header("Datos del Paciente")

with st.form("formulario_paciente"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Edad (años)", min_value=None, max_value=120, value=None, step=1, placeholder="Ej: 50")
        gender = st.selectbox("Sexo", options=[None, "Hombre", "Mujer"], format_func=lambda x: "Seleccione..." if x is None else x)
        bmi = st.number_input("IMC (kg/m²)", min_value=None, max_value=60.0, value=None, step=0.1, placeholder="Ej: 25.0")
        hypertension = st.selectbox("¿Tiene hipertensión?", options=[None, 0, 1], format_func=lambda x: "Seleccione..." if x is None else ("No" if x == 0 else "Sí"))
    
    with col2:
        heart_disease = st.selectbox("¿Tiene enfermedad cardíaca?", options=[None, 0, 1], format_func=lambda x: "Seleccione..." if x is None else ("No" if x == 0 else "Sí"))
        avg_glucose = st.number_input("Glucosa promedio (mg/dL)", min_value=None, max_value=300.0, value=None, step=0.1, placeholder="Ej: 100.0")
        ever_married = st.selectbox("¿Está casado/a?", options=[None, "Sí", "No"], format_func=lambda x: "Seleccione..." if x is None else x)
        work_type = st.selectbox("Tipo de trabajo", options=[None, "Privado", "Independiente", "Empleado público", "Niño/a", "Nunca ha trabajado"], format_func=lambda x: "Seleccione..." if x is None else x)
    
    with col3:
        residence = st.selectbox("Tipo de residencia", options=[None, "Urbana", "Rural"], format_func=lambda x: "Seleccione..." if x is None else x)
        smoking = st.selectbox("Estado de tabaquismo", options=[None, "Nunca ha fumado", "Fumó anteriormente", "Fuma actualmente", "Desconocido"], format_func=lambda x: "Seleccione..." if x is None else x)
    
    st.markdown("---")
    boton = st.form_submit_button("Predecir", use_container_width=True)

# ==========================================
# PREDICCIÓN
# ==========================================
if boton:
    # 1. Verificar que TODOS los campos estén llenos
    campos_faltantes = []
    if age is None: campos_faltantes.append("Edad")
    if gender is None: campos_faltantes.append("Sexo")
    if bmi is None: campos_faltantes.append("IMC")
    if hypertension is None: campos_faltantes.append("Hipertensión")
    if heart_disease is None: campos_faltantes.append("Enfermedad cardíaca")
    if avg_glucose is None: campos_faltantes.append("Glucosa promedio")
    if ever_married is None: campos_faltantes.append("Estado civil")
    if work_type is None: campos_faltantes.append("Tipo de trabajo")
    if residence is None: campos_faltantes.append("Residencia")
    if smoking is None: campos_faltantes.append("Tabaquismo")
    
    if campos_faltantes:
        st.error(f"Por favor complete todos los campos. Faltan: {', '.join(campos_faltantes)}")
    else:
        # 2. Mapear los valores en español a los que espera el modelo (inglés)
        gender_map = {"Hombre": "Male", "Mujer": "Female"}
        married_map = {"Sí": "Yes", "No": "No"}
        work_map = {
            "Privado": "Private",
            "Independiente": "Self-employed",
            "Empleado público": "Govt_job",
            "Niño/a": "children",
            "Nunca ha trabajado": "Never_worked"
        }
        residence_map = {"Urbana": "Urban", "Rural": "Rural"}
        smoking_map = {
            "Nunca ha fumado": "never smoked",
            "Fumó anteriormente": "formerly smoked",
            "Fuma actualmente": "smokes",
            "Desconocido": "Unknown"
        }
        
        # 3. Construir el diccionario de datos
        datos_dict = {
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'avg_glucose_level': avg_glucose,
            'bmi': bmi,
            'gender_Male': 1 if gender_map[gender] == "Male" else 0,
            'ever_married_Yes': 1 if married_map[ever_married] == "Yes" else 0,
            'work_type_Never_worked': 1 if work_map[work_type] == "Never_worked" else 0,
            'work_type_Private': 1 if work_map[work_type] == "Private" else 0,
            'work_type_Self-employed': 1 if work_map[work_type] == "Self-employed" else 0,
            'work_type_children': 1 if work_map[work_type] == "children" else 0,
            'Residence_type_Urban': 1 if residence_map[residence] == "Urban" else 0,
            'smoking_status_formerly smoked': 1 if smoking_map[smoking] == "formerly smoked" else 0,
            'smoking_status_never smoked': 1 if smoking_map[smoking] == "never smoked" else 0,
            'smoking_status_smokes': 1 if smoking_map[smoking] == "smokes" else 0,
        }
        
        # 4. Preprocesar y predecir
        df_input = pd.DataFrame([datos_dict])
        df_input = df_input.reindex(columns=columnas, fill_value=0)
        datos_escalados = scaler.transform(df_input)
        
        # Modelo de probabilidad
        prob = modelo_prob.predict(datos_escalados, verbose=0)[0][0]
        porcentaje = prob * 100
        
        # Ensemble de categoría
        predicciones_ens = []
        for m in modelos_ensemble:
            predicciones_ens.append(m.predict(datos_escalados, verbose=0))
        promedio_ens = np.mean(predicciones_ens, axis=0)
        cat_idx = np.argmax(promedio_ens, axis=1)[0]
        categorias = ['Bajo', 'Medio', 'Alto']
        categoria = categorias[cat_idx]
        confianza = promedio_ens[0][cat_idx] * 100
        
        # ==========================================
        # MOSTRAR RESULTADOS
        # ==========================================
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

# ==========================================
# PIE DE PÁGINA
# ==========================================
st.markdown("---")
st.caption("Proyecto académico - Universidad de Córdoba - Facultad de Ingeniería - Ingeniería de Sistemas - 2026-2")
