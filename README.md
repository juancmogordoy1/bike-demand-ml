🚲 Predicción de demanda de bicicletas (Bike Sharing - Washington D.C.)

App y modelo de Machine Learning para estimar la demanda diaria de bicicletas compartidas (cnt) usando variables de calendario y clima.

Incluye:

Comparación de modelos
Pipeline con preprocesamiento integrado
Modelo exportado listo para producción
App en Streamlit para inferencia interactiva

📌 Problema

Predecir cuántos alquileres totales (cnt) habrá en un día dado utilizando:
Estación, año, mes, día de semana
Día laboral / feriado
Condición climática
Temperatura, humedad y viento
Uso práctico: planificación operativa y redistribución de bicicletas.

🧾 Dataset

Bike Sharing Dataset (Washington D.C., 2011–2012)
Archivo utilizado:
data/raw/day.csv
Fuente:
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
No se utilizan casual ni registered como features porque generan fuga de información (son parte de cnt).

🧹 Preparación de datos

Columnas descartadas:
instant
dteday
casual
registered
Variables utilizadas:

Categóricas:
season, yr, mnth, holiday, weekday, workingday, weathersit

Numéricas:
temp, atemp, hum, windspeed

Preprocesamiento con ColumnTransformer:
OneHotEncoder para categóricas
Passthrough para numéricas
El preprocesamiento forma parte del Pipeline exportado.

🧠 Modelado

Modelos evaluados:
Ridge
RandomForestRegressor
GradientBoostingRegressor
KNeighborsRegressor

Resultados (split 80/20)
Modelo	MAE	R²
GradientBoosting	442.90	0.897
RandomForest	442.99	0.883
Ridge	579.70	0.842
KNN (k=25)	713.31	0.804

Modelo final: GradientBoostingRegressor (default)
Motivo: mejor MAE y mejor R².

🏋️ Entrenamiento final

Script: src/train.py

Proceso:
Split 80/20 para evaluación
Reentrenamiento con todo el dataset

Export del pipeline completo:
models/gradient_boosting.pkl

🖥️ App (Streamlit)

La app permite:
Ingresar condiciones del día
Predecir cnt
Mostrar rango orientativo ±12%
Comparar contra percentiles históricos

Arquitectura:
Streamlit NO entrena
Solo carga gradient_boosting.pkl

▶️ Ejecutar localmente
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train  # opcional
streamlit run src/app.py



📁 Estructura
bike-demand-ml/
├── data/raw/day.csv
├── models/gradient_boosting.pkl
├── reports/model_comparison.csv
├── src/
│   ├── train_compare.py
│   ├── train.py
│   └── app.py
├── requirements.txt
└── README.md



Autor: Juan Cruz Mogordoy
