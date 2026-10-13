import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import base64
from datetime import datetime

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide"
)

# ==========================================
# ESTADO DE SESIÓN
# ==========================================
if "historial" not in st.session_state:
    st.session_state.historial = []

if "modo_oscuro" not in st.session_state:
    st.session_state.modo_oscuro = True

# ==========================================
# BARRA LATERAL
# ==========================================
with st.sidebar:
    try:
        with open("logou.png", "rb") as f:
            logo_data = f.read()
        logo_base64 = base64.b64encode(logo_data).decode()
        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        ">
            <img src="data:image/png;base64,{logo_base64}" style="width: 100%; max-width: 220px;" />
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("### Universidad de Córdoba")

    st.markdown("---")
    st.markdown("## Información del Proyecto")
    st.markdown("---")

    st.markdown("### Integrantes")
    st.markdown("""
    - Dilan Estiven Ortega Pinzón
    - Juan David Andrade Llanos
    - Andrés Camilo Gonzalez Echavarria
    """)

    st.markdown("---")
    st.markdown("### Información Académica")
    st.markdown("""
    - **Programa:** Ingeniería de Sistemas
    - **Facultad:** Facultad de Ingeniería
    - **Universidad:** Universidad de Córdoba
    - **Periodo:** 2026-2
    - **Asignatura:** Simulación
    - **Docente:** Luis Garcia Cuida
    """)

    st.markdown("---")
    st.markdown("### Descripción")
    st.caption("""
    Sistema de predicción de riesgo cardiovascular basado en redes neuronales artificiales. 
    Utiliza un dataset de 5,110 registros clínicos con 16 variables predictoras para estimar 
    la probabilidad de riesgo (regresión) y clasificar al paciente en tres categorías: 
    Bajo, Medio o Alto. Emplea un ensemble de 5 modelos con data augmentation para mejorar 
    la precisión del sistema.
    """)

    st.markdown("---")
    st.markdown("### Preferencias")
    modo = st.toggle("Modo oscuro", value=st.session_state.modo_oscuro)
    if modo != st.session_state.modo_oscuro:
        st.session_state.modo_oscuro = modo
        st.rerun()

# ==========================================
# FONDO CON IMAGEN + OVERLAY (SOLO MODO OSCURO)
# ==========================================
try:
    with open("fondo1.png", "rb") as f:
        img_data = f.read()
    img_base64 = base64.b64encode(img_data).decode()
    fondo_css = f"""
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)),
            url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """
except FileNotFoundError:
    fondo_css = "background-color: #0A192F;"

# ==========================================
# ESTILOS DINÁMICOS SEGÚN MODO
# ==========================================
if st.session_state.modo_oscuro:
    color_texto = "#ffffff"
    color_texto_secundario = "#a8d5e5"
    color_sidebar_bg = "rgba(10, 25, 47, 0.95)"
    color_card_bg = "rgba(255, 255, 255, 0.08)"
    color_card_border = "rgba(255, 255, 255, 0.15)"
    color_input_bg = "rgba(255, 255, 255, 0.92)"
    color_input_text = "#000000"
else:
    color_texto = "#0A192F"
    color_texto_secundario = "#1A5276"
    color_sidebar_bg = "rgba(240, 244, 248, 0.98)"
    color_card_bg = "rgba(255, 255, 255, 0.95)"
    color_card_border = "rgba(10, 25, 47, 0.15)"
    color_input_bg = "rgba(255, 255, 255, 1.0)"
    color_input_text = "#000000"
    fondo_css = "background-color: #F0F4F8;"

st.markdown(f"""
<style>
    .stApp {{
        {fondo_css}
    }}
    
    h1, h2, h3, p, label, .stMarkdown, .stCaption {{
        color: {color_texto} !important;
    }}
    
    .stNumberInput label, .stSelectbox label {{
        color: {color_texto} !important;
        font-weight: bold;
    }}
    
    .stNumberInput input, .stSelectbox select {{
        background-color: {color_input_bg} !important;
        color: {color_input_text} !important;
    }}
    
    .stMetric label, .stMetric div {{
        color: {color_texto} !important;
    }}
    
    .stProgress > div > div > div > div {{
        background-color: #4CAF50 !important;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {color_sidebar_bg} !important;
        border-right: 1px solid {color_card_border};
    }}
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stCaption {{
        color: {color_texto} !important;
    }}

    section[data-testid="stSidebar"] hr {{
        border-color: {color_card_border};
    }}

    .streamlit-expanderHeader {{
        color: {color_texto} !important;
        background-color: {color_card_bg} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {color_card_bg} !important;
        color: {color_texto} !important;
    }}

    /* Tarjetas de resultado */
    .tarjeta-resultado {{
        background: {color_card_bg};
        border-radius: 16px;
        padding: 25px;
        border: 1px solid {color_card_border};
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        text-align: center;
        margin-bottom: 15px;
    }}
    .tarjeta-resultado .titulo {{
        font-size: 1.1rem;
        color: {color_texto_secundario};
        margin-bottom: 8px;
    }}
    .tarjeta-resultado .valor {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {color_texto};
    }}

    /* Disclaimer */
    .disclaimer {{
        background: {color_card_bg};
        border-left: 4px solid #D4AF37;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 30px;
        font-size: 0.85rem;
        color: {color_texto_secundario};
        font-style: italic;
    }}

    /* Factores de riesgo */
    .factor-item {{
        background: {color_card_bg};
        border-radius: 8px;
        padding: 8px 14px;
        margin: 4px 0;
        border-left: 3px solid #E74C3C;
        color: {color_texto};
        font-size: 0.92rem;
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
# BANNER SUPERIOR (ADAPTATIVO AL MODO)
# ==========================================
if st.session_state.modo_oscuro:
    banner_bg = "linear-gradient(135deg, #0A2540 0%, #1A5276 100%)"
    banner_titulo = "#ffffff"
    banner_subtitulo = "#a8d5e5"
    banner_desc = "#d0e8f2"
    banner_border = "rgba(255, 255, 255, 0.15)"
else:
    banner_bg = "linear-gradient(135deg, #E8F0F8 0%, #D0E0F0 100%)"
    banner_titulo = "#0A2540"
    banner_subtitulo = "#1A5276"
    banner_desc = "#2C3E50"
    banner_border = "rgba(10, 25, 47, 0.15)"

st.markdown(f"""
<div style="
    background: {banner_bg};
    border-radius: 18px;
    padding: 35px 40px;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    border: 1px solid {banner_border};
    text-align: center;
">
    <h1 style="
        color: {banner_titulo} !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-align: center;
    ">
        Predicción de Riesgo Cardiovascular
    </h1>
    <p style="
        color: {banner_subtitulo} !important;
        margin-top: 12px;
        margin-bottom: 0;
        font-size: 1.1rem;
        text-align: center;
        font-style: italic;
    ">
        Proyecto de Simulación — Redes Neuronales Artificiales
    </p>
    <p style="
        color: {banner_desc} !important;
        margin-top: 18px;
        margin-bottom: 0;
        font-size: 0.95rem;
        text-align: center;
    ">
        Sistema basado en redes neuronales artificiales que estima la probabilidad y la categoría 
        de riesgo cardiovascular (Bajo / Medio / Alto) a partir de datos clínicos del paciente.
    </p>
</div>
""", unsafe_allow_html=True)

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
        # Spinner de carga
        with st.spinner("Analizando datos del paciente..."):
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
            
            # Factores de riesgo heurísticos
            factores = []
            if age > 60:
                factores.append(f"Edad avanzada ({age} años, mayor a 60)")
            if hypertension == 1:
                factores.append("Hipertensión arterial")
            if heart_disease == 1:
                factores.append("Enfermedad cardíaca previa")
            if avg_glucose > 140:
                factores.append(f"Glucosa elevada ({avg_glucose:.1f} mg/dL, mayor a 140)")
            if bmi > 30:
                factores.append(f"IMC elevado ({bmi:.1f} kg/m², mayor a 30)")
            if smoking_map[smoking] == "smokes":
                factores.append("Tabaquismo activo")
            elif smoking_map[smoking] == "formerly smoked":
                factores.append("Historial de tabaquismo")
        
        # ==========================================
        # GUARDAR EN HISTORIAL
        # ==========================================
        st.session_state.historial.append({
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Edad": age,
            "Sexo": gender,
            "IMC": f"{bmi:.1f}",
            "Glucosa": f"{avg_glucose:.1f}",
            "Probabilidad": f"{porcentaje:.2f}%",
            "Categoría": categoria
        })
        
        # ==========================================
        # MOSTRAR RESULTADOS
        # ==========================================
        st.markdown("---")
        st.header("Resultado de la Predicción")
        
        # --- Tarjetas ---
        if categoria == "Bajo":
            color = "#4CAF50"
        elif categoria == "Medio":
            color = "#FF9800"
        else:
            color = "#E74C3C"
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown(f"""
            <div class="tarjeta-resultado">
                <div class="titulo">Probabilidad de Riesgo</div>
                <div class="valor">{porcentaje:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_r2:
            st.markdown(f"""
            <div class="tarjeta-resultado">
                <div class="titulo">Categoría de Riesgo</div>
                <div class="valor" style="color: {color};">{categoria.upper()}</div>
                <div style="font-size: 0.85rem; color: {color_texto_secundario};">
                    Confianza: {confianza:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.progress(float(prob))
        
        # --- Interpretación ---
        st.markdown("### Interpretación")
        if categoria == "Bajo":
            st.info("El paciente tiene bajo riesgo cardiovascular. Se recomienda mantener hábitos saludables.")
        elif categoria == "Medio":
            st.warning("El paciente tiene riesgo moderado. Se recomienda consulta médica y seguimiento.")
        else:
            st.error("El paciente tiene alto riesgo cardiovascular. Se recomienda evaluación médica inmediata.")
        
        # --- Factores de riesgo ---
        st.markdown("### Factores de Riesgo Identificados")
        if factores:
            for f in factores:
                st.markdown(f'<div class="factor-item">{f}</div>', unsafe_allow_html=True)
        else:
            st.success("No se identificaron factores de riesgo clínicos relevantes.")
        
        # --- Datos ingresados ---
        with st.expander("Ver datos ingresados"):
            st.dataframe(df_input.T.rename(columns={0: "Valor"}))

# ==========================================
# HISTORIAL DE PREDICCIONES
# ==========================================
if st.session_state.historial:
    st.markdown("---")
    st.header("Historial de Predicciones (Sesión Actual)")
    df_hist = pd.DataFrame(st.session_state.historial)
    st.dataframe(df_hist, use_container_width=True)
    
    if st.button("Limpiar historial"):
        st.session_state.historial = []
        st.rerun()

# ==========================================
# DISCLAIMER MÉDICO
# ==========================================
st.markdown("""
<div class="disclaimer">
    <strong>Aviso médico:</strong> Esta herramienta es un proyecto académico desarrollado 
    con fines educativos. No reemplaza la evaluación, diagnóstico o tratamiento de un profesional 
    de la salud. Los resultados son estimaciones basadas en un modelo de redes neuronales y no 
    deben interpretarse como diagnóstico clínico. Consulte siempre a un médico ante cualquier 
    duda sobre su salud cardiovascular.
</div>
""", unsafe_allow_html=True)

# ==========================================
# PIE DE PÁGINA
# ==========================================
st.markdown("---")
st.caption("Proyecto académico - Universidad de Córdoba - Facultad de Ingeniería - Ingeniería de Sistemas - 2026-2")
