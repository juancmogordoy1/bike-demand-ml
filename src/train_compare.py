# importamos librerias necesarias
import pandas as pd
from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Configuración de rutas y columnas, mi proyecto tiene una estructura, y desde este archivo voy 2 niveles arriba para encontrar data/raw/day.csv
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "day.csv"

# lo que querés predecir (demanda total)
TARGET = "cnt"
DROP_COLS = ["instant", "dteday", "casual", "registered"] #sacás columnas que no sirven o generan leakage (casual, registered
# define cuáles tratás como categóricas para hacer OneHot
CAT_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]

# Prosesador general: lo uso para Ridge, RandomForest y GradientBoosting.
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", num_cols),
        ]
    )

# preprocesador específico para KNN: estandariza numéricos porque KNN es sensible a escala.
def make_preprocessor_knn(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), num_cols),
        ]
    )

# función que estandariza la evaluación/ Evita repetir código para cada modelo, garantiza q todos se midan igual y Devuelve un dict fácil de transformar en DataFrame
def evaluate(model_name: str, pipe: Pipeline, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": model_name, "mae": mae, "r2": r2}

# main(): flujo completo de comparación, donde separamos x de y haciendo el mismo train/test para todos
def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = make_preprocessor(X) # sin escalado
    pre_knn = make_preprocessor_knn(X) # con StandardScaler en numéricas
# Lista model definiste una lista de tuplas modelo por modelo
    models = [
        ("Ridge", Pipeline([("prep", pre), ("model", Ridge(alpha=1.0))])), # controla la regularización (evita overfitting).
        ("RandomForest", Pipeline([("prep", pre), ("model", RandomForestRegressor(
            n_estimators=400, random_state=42, n_jobs=-1
        ))])), # promedia árboles
        ("GradientBoosting", Pipeline([("prep", pre), ("model", GradientBoostingRegressor(
            random_state=42
        ))])), # árboles secuenciales corrigiendo error
        ("KNN(k=25)", Pipeline([("prep", pre_knn), ("model", KNeighborsRegressor(
            n_neighbors=25
        ))])), # Predice promediando los 25 vecinos, usa prep porq nesesita escalado
    ]
    # --- Cross Validation 5-Fold ---
    from sklearn.model_selection import KFold, cross_validate

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results_cv = []

    for name, pipe in models:
        scores = cross_validate(
            pipe,
            X, y,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "r2": "r2"
            },
            n_jobs=-1
        )

        mae_mean = (-scores["test_mae"]).mean()
        mae_std  = (-scores["test_mae"]).std()
        r2_mean  = scores["test_r2"].mean()
        r2_std   = scores["test_r2"].std()

        results_cv.append({
            "model": name,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std
        })

    res_cv = pd.DataFrame(results_cv).sort_values("mae_mean")

    print("\n=== CV 5-Fold (menor MAE_mean es mejor) ===")
    print(res_cv.to_string(
        index=False,
        formatters={
            "mae_mean": "{:.2f}".format,
            "mae_std": "{:.2f}".format,
            "r2_mean": "{:.3f}".format,
            "r2_std": "{:.3f}".format,
        }
    ))  


# Loop de evaluación: compara todos, recorre cada modelo, entrena con train, evalua en test, guarda MAE y R².
    results = []
    for name, pipe in models:
        results.append(evaluate(name, pipe, X_train, X_test, y_train, y_test))

# Arma ranking en DataFrame y lo imprime prolijo 
# convierto a tabla y ordeno por MAE (menor = mejor), y lo imprimo con formato para 2 decimales en MAE y 3 en R², sin mostrar índices.
    res = pd.DataFrame(results).sort_values("mae")
    print("\n=== Resultados (menor MAE es mejor) ===")
    print(res.to_string(
        index=False,
        formatters={"mae": "{:.2f}".format, "r2": "{:.3f}".format}
    ))

    out_path = BASE_DIR / "reports" / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()