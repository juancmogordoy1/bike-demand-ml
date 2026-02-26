# importe librerías

import pandas as pd
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor # modelo de árboles secuenciales (boosting)

# Configuración de rutas y columnas, mi proyecto tiene una estructura, y desde este archivo voy 2 niveles arriba para encontrar data/raw/day.csv
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "day.csv"
# lo que querés predecir (demanda total)
TARGET = "cnt" 
# DROP_COLs: sacás columnas que no sirven o generan leakage (casual, registered
DROP_COLS = ["instant", "dteday", "casual", "registered"]
#columnas que trato como categóricas (aunque sean números) para one-hot.
CAT_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]

# Función make_preprocessor: armar el preprocesamiento
# Generás la lista de columnas numéricas “por descarte”: todo lo que no está en CAT_COLS se considera numérico.
# Ventaja: si mañana agrego una columna numérica nueva, entra sola sin tocar código.
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS), # si en producción aparece una categoría nueva, no revienta el modelo, la ignora.
            ("num", "passthrough", num_cols), # dejás los números tal cual (no escalás), sirve para modelo de arbol
        ]
    )

# main(): flujo completo de entrenamiento
def main():
    df = pd.read_csv(DATA_PATH)
# despues de cargar el dataset, separo x (features) de y (target), y armo el preprocesador con make_preprocessor.  
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    pre = make_preprocessor(X)

# Luego armo un pipeline que primero hace el preprocesamiento y después entrena el modelo.
# “prep” transforma los datos (one-hot + num igual).
# “model” entrena GradientBoosting.
# Ventaja enorme: cuando guardás el modelo, guardás todo (transformaciones + modelo).
# En producción, no te olvidás de “hacer one-hot igual que en training”.
    pipe = Pipeline(
        steps=[
            ("prep", pre),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    # 1) Métricas con split (para que puedas reportar números) 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds) # MAE: error promedio en “cantidad de alquileres” (más fácil de interpretar).
    r2 = r2_score(y_test, preds)  # R²: proporción de varianza explicada (qué tan bien explica el modelo).
    print(f"GradientBoosting -> MAE: {mae:.2f} | R2: {r2:.3f}")

    # 2) Entrenamiento final con TODO el dataset (mejor para producción)
    pipe.fit(X, y)

    # 3) Guardar modelo (pipeline completo: prepro + modelo)
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gradient_boosting.pkl"
    joblib.dump(pipe, model_path)
    print(f"Modelo guardado en: {model_path}")

# Importar funciones de train.py desde otro módulo sin que vuelva a entrenar todo
# si ejecuto este archivo directamente, corré main(), pero si lo importo desde otro lado, no se ejecuta main() automáticamente.
if __name__ == "__main__":
    main()