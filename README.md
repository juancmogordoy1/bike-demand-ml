🚲 Predicción de Demanda de Bicicletas

Bike Sharing – Washington D.C. (2011–2012)

## 📷 Vista de la aplicación

![Vista de la app](app_screenshot.png)

Proyecto de Machine Learning para predecir la demanda diaria de bicicletas compartidas (cnt) utilizando variables de calendario y condiciones climáticas.
Incluye:

Comparación de múltiples modelos de regresión
Pipeline con preprocesamiento integrado
Prevención de data leakage
Evaluación con train/test split
Modelo exportado listo para producción
Aplicación interactiva desarrollada con Streamlit
Interpretabilidad mediante importancia de variables y percentiles históricos

📌 Objetivo del Proyecto

Predecir el total de alquileres diarios (cnt) para un día dado en función de:
Variables de calendario: estación, año, mes, día de semana, día laboral / feriado
Condiciones climáticas: clima general, temperatura, sensación térmica, humedad y viento

Aplicación práctica:
Planificación operativa, redistribución de bicicletas y estimación de demanda esperada según condiciones previstas.
El problema se define como regresión supervisada.

🧾 Dataset

Bike Sharing Dataset – Washington D.C. (2011–2012)
Fuente: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Archivo utilizado:
data/raw/day.csv (agregado diario)

Se eligió la versión diaria (day.csv) porque:
Reduce ruido respecto al nivel horario
Permite un modelo más interpretable
Facilita una defensa clara del enfoque

🧹 Selección de Variables y Prevención de Data Leakage

Variable objetivo:
y = cnt (total de alquileres por día)

Variables utilizadas (features):
Calendario
season
yr
mnth
weekday
holiday
workingday
Clima
weathersit
temp
atemp
hum
windspeed

Columnas descartadas:
instant → identificador sin valor predictivo
dteday → fecha literal
casual y registered → forman parte de cnt (su uso generaría fuga de información)
Esta decisión garantiza que el modelo solo utilice información disponible antes de que ocurra el día a predecir.

⚙️ Preparación de Datos

Se implementó un ColumnTransformer para:
Aplicar OneHotEncoder a variables categóricas
Dejar variables numéricas como passthrough
Todo el preprocesamiento forma parte del mismo Pipeline que el modelo, asegurando consistencia entre entrenamiento e inferencia en producción.

🧠 Modelado y Comparación de Modelos

Se evaluaron distintos enfoques para contrastar supuestos lineales y no lineales:

Modelo	Tipo	MAE	R²
GradientBoostingRegressor	Ensemble (boosting)	442.90	0.897
RandomForestRegressor	Ensemble (bagging)	442.99	0.883
Ridge	Lineal regularizado	579.70	0.842
KNN (k=25)	Basado en distancia	713.31	0.804

Split utilizado: 80% entrenamiento / 20% test.

El modelo seleccionado fue:

GradientBoostingRegressor (configuración default)
Motivo: mejor desempeño en MAE y R², capturando relaciones no lineales entre clima y demanda.

🏋️ Entrenamiento Final

Proceso:
Split 80/20 para evaluación
Selección del mejor modelo
Reentrenamiento con el 100% del dataset
Exportación del pipeline completo

Archivo exportado:
models/gradient_boosting.pkl
Esto permite utilizar el modelo sin necesidad de reentrenar en cada ejecución.

📊 Interpretabilidad

Se generaron dos mecanismos de interpretación:

1️⃣ Importancia de Variables
Exportada en:
reports/feature_importance.csv
Permite identificar qué factores influyen más en la demanda.

2️⃣ Contexto Histórico con Percentiles
Se calcularon:
Media histórica
Percentiles P25, P50, P75

La predicción se clasifica como:
Baja demanda → menor a P25
Demanda normal → entre P25 y P75
Alta demanda → mayor a P75
Esto transforma una predicción numérica en una interpretación operativa.

🖥️ Aplicación Web (Streamlit)
La aplicación permite:
Ingresar condiciones del día (clima + calendario)
Obtener predicción de demanda (cnt)
Mostrar rango orientativo ±12%
Comparar contra percentiles históricos

Arquitectura:
La app NO entrena el modelo
Solo carga el pipeline exportado (gradient_boosting.pkl)
Garantiza consistencia con el entrenamiento

▶️ Ejecución Local
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Entrenamiento opcional
python -m src.train

# Ejecutar app
streamlit run src/app.py
📁 Estructura del Proyecto
bike-demand-ml/
├── data/raw/day.csv
├── models/
│   ├── gradient_boosting.pkl
│   ├── gradient_boosting_hour_circular.pkl
│   └── model.pkl
├── reports/
│   ├── model_comparison.csv
│   └── feature_importance.csv
├── notebooks/
│   ├── 01_eda_modeling.ipynb
│   └── 02_hour_modeling.ipynb
├── src/
│   ├── train_compare.py
│   ├── train.py
│   └── app.py
├── requirements.txt
└── README.md
🎯 Resultado Final

Con datos históricos de bicicletas y variables climáticas:
Se entrenó un modelo de ML sin fuga de información
Se evaluó correctamente con train/test split
Se seleccionó el mejor modelo según métricas objetivas
Se exportó para producción
Se implementó una app interactiva
Se agregó contexto histórico para interpretación
<<<<<<< HEAD

=======
>>>>>>> 24e0992 (Mejorar el README y agregar la descripción del proyecto)


Autor: Juan Cruz Mogordoy
