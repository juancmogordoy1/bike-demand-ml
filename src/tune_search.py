import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint, uniform


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "day.csv"

TARGET = "cnt"
DROP_COLS = ["instant", "dteday", "casual", "registered"]
CAT_COLS = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", "passthrough", num_cols),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    pre = make_preprocessor(X)

    # Split solo para reportar un numerito final comparable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # =========================================================
    # A) RANDOMIZED SEARCH  -> RandomForest
    # =========================================================
    rf_pipe = Pipeline(
        steps=[
            ("prep", pre),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    rf_param_dist = {
        "model__n_estimators": randint(200, 801),          # 200..800
        "model__max_depth": [None, 10, 20, 30, 40],
        "model__min_samples_split": randint(2, 21),        # 2..20
        "model__min_samples_leaf": randint(1, 11),         # 1..10
        "model__max_features": ["sqrt", "log2", None],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=rf_param_dist,
        n_iter=25,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    rf_search.fit(X_train, y_train)
    print("\n=== RandomizedSearchCV (RandomForest) ===")
    print("Mejor MAE CV:", -rf_search.best_score_)
    print("Mejores params:", rf_search.best_params_)

    # =========================================================
    # B) GRID SEARCH  -> GradientBoosting
    # =========================================================
    gb_pipe = Pipeline(
        steps=[
            ("prep", pre),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )

    gb_param_grid = {
        "model__n_estimators": [150, 250, 350],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__subsample": [0.8, 1.0],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 3, 5],
    }

    gb_search = GridSearchCV(
        estimator=gb_pipe,
        param_grid=gb_param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    gb_search.fit(X_train, y_train)
    print("\n=== GridSearchCV (GradientBoosting) ===")
    print("Mejor MAE CV:", -gb_search.best_score_)
    print("Mejores params:", gb_search.best_params_)

    # =========================================================
    # Elegir ganador por MAE de CV (más bajo es mejor)
    # =========================================================
    best_search = gb_search if (-gb_search.best_score_ < -rf_search.best_score_) else rf_search
    best_name = "GradientBoosting(Grid)" if best_search is gb_search else "RandomForest(Randomized)"
    best_pipe = best_search.best_estimator_

    # Reporte en test (para que tengas un número fácil de explicar)
    preds = best_pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n=== Ganador: {best_name} ===")
    print(f"TEST -> MAE: {mae:.2f} | R2: {r2:.3f}")

    # Reentrenar ganador con TODO el dataset y guardar (para tu app)
    best_pipe.fit(X, y)

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "gradient_boosting.pkl"  # mantiene compatibilidad con tu Streamlit
    joblib.dump(best_pipe, model_path)
    print(f"\nModelo final guardado en: {model_path}")


if __name__ == "__main__":
    main()