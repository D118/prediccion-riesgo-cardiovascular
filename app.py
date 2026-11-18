import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
import base64
from datetime import datetime
import time
import io # NUEVO: Para generar el PDF en memoria
from fpdf import FPDF # NUEVO: Para el PDF
import plotly.graph_objects as go # NUEVO: Para el gráfico de radar
from groq import Groq

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Predicción de Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide"
)

# ==========================================
# CONFIGURACIÓN LLM (GROQ) Y RL
# ==========================================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = "FALTA_CONFIGURAR_SECRET"

client = Groq(api_key=GROQ_API_KEY)

if "ejemplos_exitosos" not in st.session_state:
    st.session_state.ejemplos_exitosos = []
if "feedback_dado" not in st.session_state:
    st.session_state.feedback_dado = False

def generar_recomendacion_llm(categoria, factores, edad, sexo):
    contexto_paciente = f"Paciente de {edad} años, sexo {sexo}. "
    contexto_paciente += f"Nivel de riesgo cardiovascular: {categoria}. "
    
    if factores:
        contexto_paciente += f"Algunos puntos a mejorar: {', '.join(factores)}. "
    else:
        contexto_paciente += "El paciente no tiene factores de riesgo graves. "

    instruccion_rl = ""
    if st.session_state.ejemplos_exitosos:
        instruccion_rl = "Ten en cuenta que a los usuarios les han gustado recomendaciones con este estilo:\n"
        for ej in st.session_state.ejemplos_exitosos[-2:]: 
            instruccion_rl += f"- \"{ej}\"\n"

    prompt = f"""
    Actúa como un asistente virtual de salud muy amable, empático y cercano. 
    Tu tarea es generar un consejo o recomendación basándote en estos datos clínicos:
    {contexto_paciente}
    {instruccion_rl}
    
    Reglas ESTRICTAS para tu respuesta:
    1. NO digas "soy tu médico" ni te asignes títulos médicos. Eres un sistema de apoyo. Puedes iniciar con un saludo cálido o refiriéndote directamente al paciente.
    2. Háblale de "tú" (o "usted" si es mayor de 60 años), de forma cálida y tranquila. Que no parezca un robot.
    3. EXPLICA LAS COSAS EN PALABRAS SENCILLAS. No uses términos médicos complicados (por ejemplo, en vez de decir "dislipidemia", di "colesterol alto"; en vez de "evento cardiovascular", di "problemas del corazón").
    4. Dale 2 o 3 consejos prácticos y realistas que pueda aplicar en su día a día (ej. caminar, cambiar la dieta, descansar).
    5. El texto debe tener más de 200 caracteres, pero no debe ser un testamento. Máximo 4 o 5 oraciones.
    6. No uses viñetas ni formato Markdown, solo texto corrido en un párrafo.
    """
    
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Eres un asistente virtual de salud empático y muy bueno explicando la salud en palabras sencillas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8, 
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error al conectar con el LLM: {str(e)}. (Verifica tu API Key en los Secrets de Streamlit)."

# ==========================================
# NUEVO: FUNCIÓN PARA GENERAR EL PDF
# ==========================================
def generar_pdf(res, recomendacion):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Reporte de Riesgo Cardiovascular", ln=True, align='C')
    pdf.ln(5)
    
    # Fecha
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='R')
    pdf.ln(5)
    
    # Datos del paciente
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Datos del Paciente:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Edad: {res['age']} años", ln=True)
    pdf.cell(0, 8, f"Sexo: {res['gender']}", ln=True)
    pdf.cell(0, 8, f"IMC: {res['bmi']:.1f} kg/m2", ln=True)
    pdf.cell(0, 8, f"Glucosa promedio: {res['avg_glucose']:.1f} mg/dL", ln=True)
    pdf.ln(5)
    
    # Resultados de la predicción
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resultados de la Prediccion:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"Probabilidad de Riesgo: {res['porcentaje']:.2f}%", ln=True)
    pdf.cell(0, 8, f"Categoria de Riesgo: {res['categoria']}", ln=True)
    pdf.cell(0, 8, f"Confianza del modelo: {res['confianza']:.1f}%", ln=True)
    pdf.ln(5)
    
    # Factores de riesgo
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Factores de Riesgo Identificados:", ln=True)
    pdf.set_font("Arial", '', 11)
    if res['factores']:
        for _, texto in res['factores']:
            pdf.cell(0, 8, f"- {texto}", ln=True)
    else:
        pdf.cell(0, 8, "No se identificaron factores de riesgo relevantes.", ln=True)
    pdf.ln(5)
    
    # Recomendación LLM
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Recomendacion Generada por IA:", ln=True)
    pdf.set_font("Arial", '', 11)
    # Usamos multi_cell para textos largos
    pdf.multi_cell(0, 8, recomendacion.encode('latin-1', 'replace').decode('latin-1'))
    
    return bytes(pdf.output())

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
# ESTILOS DINÁMICOS SEGÚN MODO (CON GLASSMORPHISM)
# ==========================================
if st.session_state.modo_oscuro:
    color_texto = "#ffffff"
    color_texto_secundario = "#a8d5e5"
    color_sidebar_bg = "rgba(10, 25, 47, 0.85)"
    color_card_bg = "rgba(255, 255, 255, 0.05)"
    color_card_border = "rgba(255, 255, 255, 0.15)"
    color_input_bg = "rgba(255, 255, 255, 0.92)"
    color_input_text = "#000000"
    glass_filter = "backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);"
else:
    color_texto = "#0A192F"
    color_texto_secundario = "#1A5276"
    color_sidebar_bg = "rgba(240, 244, 248, 0.85)"
    color_card_bg = "rgba(255, 255, 255, 0.7)"
    color_card_border = "rgba(10, 25, 47, 0.1)"
    color_input_bg = "rgba(255, 255, 255, 1.0)"
    color_input_text = "#000000"
    fondo_css = "background-color: #F0F4F8;"
    glass_filter = "backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); box-shadow: 0 4px 15px rgba(0,0,0,0.05);"

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

    section[data-testid="stSidebar"] {{
        background-color: {color_sidebar_bg} !important;
        border-right: 1px solid {color_card_border};
        {glass_filter}
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

    /* Tarjetas de resultado con Glassmorphism */
    .tarjeta-resultado {{
        background: {color_card_bg};
        {glass_filter}
        border-radius: 16px;
        padding: 25px;
        border: 1px solid {color_card_border};
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        text-align: center;
        margin-bottom: 15px;
        transition: transform 0.3s ease;
    }}
    .tarjeta-resultado:hover {{
        transform: translateY(-5px);
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
        {glass_filter}
        border-left: 4px solid #D4AF37;
        border-radius: 8px;
        padding: 15px 20px;
        margin-top: 30px;
        font-size: 0.85rem;
        color: {color_texto_secundario};
        font-style: italic;
    }}

    /* Factores de riesgo con hover */
    .factor-item {{
        background: {color_card_bg};
        {glass_filter}
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #E74C3C;
        color: {color_texto};
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
        transition: all 0.2s;
    }}
    .factor-item:hover {{
        background: rgba(231, 76, 60, 0.15);
        border-left: 3px solid #FF5252;
    }}

    /* Barra de progreso personalizada */
    .custom-progress-container {{
        width: 100%;
        background-color: rgba(150, 150, 150, 0.2);
        border-radius: 10px;
        margin: 15px 0;
        height: 16px;
        overflow: hidden;
    }}
    .custom-progress-bar {{
        height: 100%;
        border-radius: 10px;
        transition: width 1.5s ease-in-out, background-color 0.5s;
    }}

    /* SKELETON LOADER para el LLM */
    .skeleton-loader {{
        background: linear-gradient(90deg, rgba(255,255,255,0.05) 25%, rgba(255,255,255,0.15) 50%, rgba(255,255,255,0.05) 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 8px;
        height: 100px;
        width: 100%;
        margin-bottom: 15px;
    }}
    @keyframes loading {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
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
# BANNER SUPERIOR
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
        Proyecto de Simulación — Redes Neuronales Artificiales + LLM + RL
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

with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad (años)", min_value=None, max_value=120, value=None, step=1, placeholder="Ej: 50")
        gender = st.selectbox("Sexo", options=[None, "Hombre", "Mujer"], format_func=lambda x: "Seleccione..." if x is None else x)
        weight = st.number_input("Peso (kg)", min_value=None, max_value=250.0, value=None, step=0.1, placeholder="Ej: 70.0")
        height = st.number_input("Altura (cm)", min_value=None, max_value=250.0, value=None, step=0.1, placeholder="Ej: 170.0")
        hypertension = st.selectbox("¿Tiene hipertensión?", options=[None, 0, 1], format_func=lambda x: "Seleccione..." if x is None else ("No" if x == 0 else "Sí"))
    
    with col2:
        heart_disease = st.selectbox("¿Tiene enfermedad cardíaca?", options=[None, 0, 1], format_func=lambda x: "Seleccione..." if x is None else ("No" if x == 0 else "Sí"))
        
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        col_titulo, col_toggle = st.columns([3, 2])
        with col_titulo:
            st.markdown("**Glucosa promedio (mg/dL)**")
        with col_toggle:
            no_glucosa = st.toggle("No la conozco", help="Activa si no sabes tu nivel de glucosa.")
            
        if no_glucosa:
            st.markdown(f"""
            <div style="
                background-color: rgba(240, 242, 246, 0.5); 
                border: 1px solid rgba(250, 250, 250, 0.2); 
                border-radius: 0.5rem; 
                padding: 0.5rem 1rem; 
                color: #888888;
                margin-bottom: 15px;
            ">
                100.0
            </div>
            """, unsafe_allow_html=True)
            avg_glucose = 100.0
        else:
            avg_glucose = st.number_input("", min_value=None, max_value=300.0, value=None, step=0.1, placeholder="Ej: 100.0", label_visibility="collapsed")

        smoking = st.selectbox("Estado de tabaquismo", options=[None, "Nunca ha fumado", "Fumó anteriormente", "Fuma actualmente", "Desconocido"], format_func=lambda x: "Seleccione..." if x is None else x)
    
    st.markdown("---")
    boton = st.button("Predecir", use_container_width=True)

# ==========================================
# PREDICCIÓN
# ==========================================
if boton:
    campos_faltantes = []
    if age is None: campos_faltantes.append("Edad")
    if gender is None: campos_faltantes.append("Sexo")
    if weight is None: campos_faltantes.append("Peso")
    if height is None: campos_faltantes.append("Altura")
    if hypertension is None: campos_faltantes.append("Hipertensión")
    if heart_disease is None: campos_faltantes.append("Enfermedad cardíaca")
    if not no_glucosa and avg_glucose is None: campos_faltantes.append("Glucosa promedio")
    if smoking is None: campos_faltantes.append("Tabaquismo")
    
    if campos_faltantes:
        st.error(f"Por favor complete todos los campos. Faltan: {', '.join(campos_faltantes)}")
    else:
        with st.spinner("Analizando datos del paciente..."):
            altura_metros = height / 100.0
            bmi = weight / (altura_metros ** 2)

            gender_map = {"Hombre": "Male", "Mujer": "Female"}
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
                'ever_married_Yes': 0, 
                'work_type_Never_worked': 0,
                'work_type_Private': 1,
                'work_type_Self-employed': 0,
                'work_type_children': 0,
                'Residence_type_Urban': 1,
                'smoking_status_formerly smoked': 1 if smoking_map[smoking] == "formerly smoked" else 0,
                'smoking_status_never smoked': 1 if smoking_map[smoking] == "never smoked" else 0,
                'smoking_status_smokes': 1 if smoking_map[smoking] == "smokes" else 0,
            }
            
            df_input = pd.DataFrame([datos_dict])
            df_input = df_input.reindex(columns=columnas, fill_value=0)
            datos_escalados = scaler.transform(df_input)
            
            predicciones_ens = []
            for m in modelos_ensemble:
                predicciones_ens.append(m.predict(datos_escalados, verbose=0))
            promedio_ens = np.mean(predicciones_ens, axis=0)
            cat_idx = np.argmax(promedio_ens, axis=1)[0]
            categorias = ['Bajo', 'Medio', 'Alto']
            categoria = categorias[cat_idx]
            confianza = promedio_ens[0][cat_idx] * 100
            
            if categoria == "Bajo":
                prob = (confianza / 100) * 0.33
            elif categoria == "Medio":
                prob = 0.33 + ((confianza / 100) * 0.33)
            else:
                prob = 0.66 + ((confianza / 100) * 0.34)
                
            porcentaje = prob * 100
            
            factores = []
            if age > 60:
                factores.append(("⚠️", f"Edad avanzada ({age} años, mayor a 60)"))
            if hypertension == 1:
                factores.append(("🩸", "Hipertensión arterial"))
            if heart_disease == 1:
                factores.append(("❤️", "Enfermedad cardíaca previa"))
            if avg_glucose > 140:
                factores.append(("🍬", f"Glucosa elevada ({avg_glucose:.1f} mg/dL)"))
            if bmi > 30:
                factores.append(("⚖️", f"IMC elevado ({bmi:.1f} kg/m²)"))
            if smoking_map[smoking] == "smokes":
                factores.append(("🚬", "Tabaquismo activo"))
            elif smoking_map[smoking] == "formerly smoked":
                factores.append(("🚬", "Historial de tabaquismo"))
        
        st.session_state.resultado_prediccion = {
            "prob": prob,
            "porcentaje": porcentaje,
            "categoria": categoria,
            "confianza": confianza,
            "factores": factores,
            "df_input": df_input,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "avg_glucose": avg_glucose
        }
        
        st.session_state.historial.append({
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Edad": age,
            "Sexo": gender,
            "Peso (kg)": f"{weight:.1f}",
            "Altura (cm)": f"{height:.1f}",
            "IMC": f"{bmi:.1f}",
            "Glucosa": f"{avg_glucose:.1f}",
            "Probabilidad": f"{porcentaje:.2f}%",
            "Categoría": categoria
        })
        
        if "recomendacion_actual" in st.session_state:
            del st.session_state.recomendacion_actual
        st.session_state.feedback_dado = False
        st.rerun()

# ==========================================
# MOSTRAR RESULTADOS
# ==========================================
if "resultado_prediccion" in st.session_state:
    res = st.session_state.resultado_prediccion
    
    st.markdown("---")
    st.header("Resultado de la Predicción")
    
    if res['categoria'] == "Bajo":
        color = "#4CAF50"
        color_barra = "#4CAF50"
    elif res['categoria'] == "Medio":
        color = "#FF9800"
        color_barra = "#FF9800"
    else:
        color = "#E74C3C"
        color_barra = "#E74C3C"
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.markdown(f"""
        <div class="tarjeta-resultado">
            <div class="titulo">Probabilidad de Riesgo</div>
            <div class="valor">{res['porcentaje']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r2:
        st.markdown(f"""
        <div class="tarjeta-resultado">
            <div class="titulo">Categoría de Riesgo</div>
            <div class="valor" style="color: {color};">{res['categoria'].upper()}</div>
            <div style="font-size: 0.85rem; color: {color_texto_secundario};">
                Confianza: {res['confianza']:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="custom-progress-container">
        <div class="custom-progress-bar" style="width: {res['porcentaje']}%; background-color: {color_barra}; box-shadow: 0 0 10px {color_barra};"></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Interpretación")
    if res['categoria'] == "Bajo":
        st.info("El paciente tiene bajo riesgo cardiovascular. Se recomienda mantener hábitos saludables.")
    elif res['categoria'] == "Medio":
        st.warning("El paciente tiene riesgo moderado. Se recomienda consulta médica y seguimiento.")
    else:
        st.error("El paciente tiene alto riesgo cardiovascular. Se recomienda evaluación médica inmediata.")
    
    # --- SECCIÓN DE FACTORES Y GRÁFICO RADAR ---
    col_fact, col_radar = st.columns([1, 1])
    
    with col_fact:
        st.markdown("### Factores de Riesgo")
        if res['factores']:
            for icono, texto in res['factores']:
                st.markdown(f'<div class="factor-item"><span style="font-size:1.2rem;">{icono}</span> {texto}</div>', unsafe_allow_html=True)
        else:
            st.success("No se identificaron factores de riesgo clínicos relevantes.")
    
    with col_radar:
        st.markdown("### Perfil de Riesgo (Radar)")
        # Normalizamos los valores para el gráfico (escala 0 a 100)
        # Edad (max 80), Glucosa (max 200), IMC (max 40), Presión (0 o 100), Corazón (0 o 100)
        val_edad = min((res['age'] / 80) * 100, 100)
        val_glucosa = min((res['avg_glucose'] / 200) * 100, 100)
        val_imc = min((res['bmi'] / 40) * 100, 100)
        val_presion = 100 if res['factores'] and any("Hipertensión" in f[1] for f in res['factores']) else 0
        val_corazon = 100 if res['factores'] and any("cardíaca" in f[1] for f in res['factores']) else 0
        
        categorias_radar = ['Edad', 'Glucosa', 'IMC', 'Hipertensión', 'Enf. Cardíaca']
        valores_radar = [val_edad, val_glucosa, val_imc, val_presion, val_corazon]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=valores_radar,
            theta=categorias_radar,
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.3)', # Rojo translúcido
            line=dict(color='#E74C3C', width=2),
            name='Perfil del Paciente'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                bgcolor='rgba(0,0,0,0)' # Fondo transparente
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
