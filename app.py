import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Config general de la app
# ----------------------------
# App inputs/predicción.

st.set_page_config(
    page_title="Predicción de demanda de bicis",
    page_icon="🚲",
    layout="wide"
)

st.markdown("""
<style>
div.stButton > button {
    background-color: #2e7d32;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    height: 3em;
    border: none;
}
div.stButton > button:hover {
    background-color: #256428;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1e293b, #0f172a);
}

.main .block-container {
    background: transparent;
}

[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# Base dir para que funcione aunque ejecutes desde otra carpeta
# (evita problemas de rutas relativas).
BASE_DIR = Path(__file__).resolve().parent

# Columnas que NO se usan para predecir (mismo criterio que en entrenamiento)
TARGET = "cnt"
DROP_COLS = ["instant", "dteday", "casual", "registered"]

# ----------------------------
# Cargas con cache
# ----------------------------
# cache_resource: objetos pesados (modelo) que no cambian en cada interacción

@st.cache_resource
def load_model():
    # Pipeline guardado desde train.py (prep + modelo)
    return joblib.load(BASE_DIR / "models" / "gradient_boosting.pkl")

# cache_data: datos (DataFrame) que no querés recargar siempre
@st.cache_data
def load_data():
    # Dataset histórico: lo usamos para contexto (promedios, distribución, etc.)
    return pd.read_csv(BASE_DIR / "data" / "raw" / "day.csv")

@st.cache_data
def load_importance():
    """
    Este CSV es opcional. Si no existe, devolvemos DataFrame vacío
    para que la app NO se rompa.
    """
    path = BASE_DIR / "reports" / "feature_importance.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["feature", "importance"])


model = load_model()
df = load_data()
imp = load_importance()

# Lista de FEATURES exactamente igual a entrenamiento:
# todo menos target y columnas descartadas.
FEATURES = [c for c in df.columns if c not in ([TARGET] + DROP_COLS)]

# ----------------------------
# Estadísticas de contexto
# ----------------------------
# Objetivo: mostrarle al usuario qué es "normal" en el dataset.

avg_cnt = float(df["cnt"].mean())
p25 = float(df["cnt"].quantile(0.25))
p50 = float(df["cnt"].quantile(0.50))
p75 = float(df["cnt"].quantile(0.75))


def clean_feature_name(name: str) -> str:
    """
    Limpia nombres técnicos (ej: 'num__temp') y los hace más legibles.
    Útil si el CSV de importancias viene del pipeline con prefijos.
    """
    name = name.replace("num__", "").replace("cat__", "")

    mappings = {
        "temp": "Temperatura",
        "atemp": "Sensación térmica",
        "hum": "Humedad",
        "windspeed": "Viento",
        "yr_0": "Año 2011",
        "yr_1": "Año 2012",
        "season_1": "Primavera",
        "season_2": "Verano",
        "season_3": "Otoño",
        "season_4": "Invierno",
        "weathersit_1": "Clima despejado",
        "weathersit_2": "Clima nublado",
        "weathersit_3": "Lluvia ligera",
        "weathersit_4": "Tormenta fuerte",
        "holiday_1": "Feriado",
        "workingday_1": "Día laborable",
    }
    return mappings.get(name, name)

# ----------------------------
# UI principal
# ----------------------------
st.markdown("""
<div style="
    background-color: rgba(0,0,0,0.6);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 30px;
">
    <h1 style="color: white; margin: 0;">
        🚲 Predicción de demanda de bicicletas
    </h1>
</div>
""", unsafe_allow_html=True)
st.caption(
    "Modelo entrenado con datos de Washington D.C. (2011–2012). "
    "Ingresá condiciones del día y estimamos alquileres totales."
)

with st.expander("ℹ️ ¿Qué hace esta app? (leer)"):
    st.write(
        """
- Estima cuántos **alquileres totales** (cnt) habrá en un día.
- Usa variables como **estación, clima, temperatura, humedad y viento**.
- La predicción es una **estimación**: puede variar por eventos especiales, obras, vacaciones, etc.
        """
    )

# ----------------------------
# Diccionarios (labels -> códigos)
# ----------------------------
# Objetivo: que el usuario vea texto entendible y el modelo reciba números del dataset.

SEASON_LABELS = {1: "Primavera", 2: "Verano", 3: "Otoño", 4: "Invierno"}

WEATHER_LABELS = {
    1: "Despejado / pocas nubes",
    2: "Niebla / nublado",
    3: "Lluvia ligera / nieve ligera",
    4: "Lluvia fuerte / tormenta",
}

WEEKDAY_LABELS = {
    0: "Domingo",
    1: "Lunes",
    2: "Martes",
    3: "Miércoles",
    4: "Jueves",
    5: "Viernes",
    6: "Sábado",
}

# ----------------------------
# Inputs
# ----------------------------
# SIDEBAR – Configuración
# ----------------------------

st.sidebar.title("🎛 Configuración del día")

# Calendario
st.sidebar.subheader("📅 Calendario")

season_label = st.sidebar.selectbox("Estación", list(SEASON_LABELS.values()), index=2)
season = [k for k, v in SEASON_LABELS.items() if v == season_label][0]

weekday_label = st.sidebar.selectbox("Día de la semana", list(WEEKDAY_LABELS.values()), index=1)
weekday = [k for k, v in WEEKDAY_LABELS.items() if v == weekday_label][0]

mnth = st.sidebar.slider("Mes", 1, 12, 6)

yr_label = st.sidebar.selectbox("Año ", ["2011", "2012"], index=1)
yr = 0 if yr_label == "2011" else 1

holiday = st.sidebar.toggle("Feriado", value=False)
workingday = st.sidebar.toggle("Día laborable", value=True)

# Clima categórico
st.sidebar.subheader("🌤 Clima")

weathersit_label = st.sidebar.selectbox("Clima", list(WEATHER_LABELS.values()), index=0)
weathersit = [k for k, v in WEATHER_LABELS.items() if v == weathersit_label][0]

# Clima real (numérico)
st.sidebar.subheader("🌡 Variables ambientales")

temp_c = st.sidebar.slider("Temperatura (°C)", 0, 41, 20)
atemp_c = st.sidebar.slider("Sensación térmica (°C)", 0, 50, 22)
hum_percent = st.sidebar.slider("Humedad (%)", 0, 100, 60)
windspeed_kmh = st.sidebar.slider("Viento (km/h)", 0, 67, 15)

st.sidebar.caption("Ingresá valores reales. La app los convierte internamente.")

# ----------------------------
# Normalización a escala dataset (0–1)
# ----------------------------
# En el dataset original: temp/atemp/hum/windspeed normalizados.

temp = temp_c / 41
atemp = atemp_c / 50
hum = hum_percent / 100
windspeed = windspeed_kmh / 67

# Regla lógica: si es feriado, normalmente no es laborable
if holiday:
    workingday = False

# ----------------------------
# Construcción de input para el modelo
# ----------------------------
# Objetivo CLAVE: evitar errores de columnas.
# Creamos un dict con todas las FEATURES y pisamos solo lo que el usuario define.

defaults = {c: 0 for c in FEATURES}

# Algunas columnas del dataset existen pero tu UI no las pide (ej: 'w' si existiera).
# Con defaults=0 queda estable. Si el dataset trae 'hum' etc ya las pisamos.
user_values = {
    "season": season,
    "yr": yr,
    "mnth": mnth,
    "holiday": 1 if holiday else 0,
    "weekday": weekday,
    "workingday": 1 if workingday else 0,
    "weathersit": weathersit,
    "temp": temp,
    "atemp": atemp,
    "hum": hum,
    "windspeed": windspeed,
}

defaults.update(user_values)

# DataFrame 1 fila con el orden exacto de FEATURES
X = pd.DataFrame([defaults], columns=FEATURES)

# ----------------------------
# Predicción
# ----------------------------
st.markdown("## ")

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button("🔮 Predecir demanda", use_container_width=True)

pred = None  # evita NameError

if predict:
    pred = float(model.predict(X)[0])

    delta = pred - p50
    low = pred * 0.88
    high = pred * 1.12

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🚲 Resultado de la predicción")
        st.metric(
            label="Demanda estimada",
            value=f"{pred:,.0f} alquileres",
            delta=f"{delta:,.0f} vs mediana"
        )
        st.write(f"Rango estimado: {low:,.0f} – {high:,.0f}")
        st.caption("Nota: el rango es orientativo (no es intervalo estadístico formal).")
        
    percentile = (df["cnt"] < pred).mean() * 100

    st.markdown(
        f"📈 Tu predicción está en el **percentil {percentile:.1f}** histórico."
    )
    # EL GRÁFICO
    st.markdown("### 📊 Demanda histórica vs tu predicción")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.hist(df["cnt"], bins=30, alpha=0.6)

    ax.axvline(p50, linestyle="--", linewidth=2)
    ax.axvline(pred, linewidth=3)

    ax.set_xlabel("Alquileres diarios (cnt)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución histórica de demanda")

    ax.grid(alpha=0.3)

    st.pyplot(fig)
    

# ----------------------------
# Contexto vs dataset
# ----------------------------


c1, c2, c3, c4 = st.columns(4)
c1.metric("Promedio histórico", f"{avg_cnt:,.0f}")
c2.metric("P25", f"{p25:,.0f}")
c3.metric("Mediana (P50)", f"{p50:,.0f}")
c4.metric("P75", f"{p75:,.0f}")

if pred is not None:
    if pred < p25:
        st.warning("Interpretación: **Demanda baja** (por debajo del 25% de los días históricos).")
    elif pred < p75:
        st.info("Interpretación: **Demanda normal** (rango típico histórico).")
    else:
        st.success("Interpretación: **Demanda alta** (por encima del 75% de los días históricos).")
else:
    st.info("Pulsa **🔮 Predecir demanda** para ver la interpretación comparada con el histórico.")



# ----------------------------
# Distribución del dataset
# ----------------------------
st.subheader("📊 Contexto del dataset")
st.caption("Para entender si la predicción está en un orden razonable, mirá la distribución real de alquileres diarios.")

bins = st.slider("Nivel de detalle del gráfico", 8, 30, 16)
counts, edges = pd.cut(df["cnt"], bins=bins, retbins=True, include_lowest=True)
hist = counts.value_counts().sort_index()

midpoints = [interval.mid for interval in hist.index]
hist_df = pd.DataFrame({"cnt_aprox": midpoints, "frecuencia": hist.values}).sort_values("cnt_aprox")

st.bar_chart(hist_df.set_index("cnt_aprox")["frecuencia"])
st.caption("Eje X: alquileres diarios aproximados (punto medio de cada rango).")

with st.expander("Ver una muestra de datos"):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ----------------------------
# Importancia de variables (opcional)
# ----------------------------
st.subheader("🧠 ¿Qué variables influyen más en el modelo?")

if imp.empty:
    st.info(
        "No se encontró `reports/feature_importance.csv`. "
        "Si querés, lo generamos desde el modelo entrenado y lo guardamos."
    )
else:
    st.caption(
        "Importancia de variables según el archivo de importancias generado en entrenamiento. "
        "Sirve para entender qué factores empujan la predicción."
    )

    top_n = st.slider("Mostrar Top N variables", 5, 20, 10)

    top = imp.copy()
    top["feature"] = top["feature"].apply(clean_feature_name)
    top = top.sort_values("importance", ascending=False).head(top_n)
    top["importance"] = top["importance"].round(3)

    st.bar_chart(top.set_index("feature")["importance"])

    dominante = top.iloc[0]
    st.caption(f"Variable dominante: **{dominante['feature']}** (importancia {dominante['importance']}).")

    imp_pretty = imp.copy()
    imp_pretty["feature"] = imp_pretty["feature"].apply(clean_feature_name)
    st.dataframe(imp_pretty, use_container_width=True)

    with st.expander("Ver tabla completa (raw)"):
        st.dataframe(imp, use_container_width=True)


        st.markdown("---")

st.markdown("""
<div style="
    text-align: center;
    padding: 25px;
    font-size: 15px;
    color: #e2e8f0;
    margin-top: 40px;
">
    🚲 Proyecto desarrollado por <strong>Juan Cruz Mogordoy</strong><br>
    Machine Learning | Modelado predictivo | Streamlit App
</div>
""", unsafe_allow_html=True)
 
        