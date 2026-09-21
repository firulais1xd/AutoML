"""
Zoológico de modelos, entrenamiento, validación cruzada, métricas y
comparación de resultados para clasificación y regresión.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso,
    ElasticNet, SGDClassifier, HuberRegressor,
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    r2_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, explained_variance_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, cross_validate,
    RandomizedSearchCV, learning_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .etl import build_preprocessor, get_feature_names

warnings.filterwarnings("ignore")

# Librerías opcionales -------------------------------------------------------
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

RANDOM_STATE = 42


# =========================================================================== #
# Detección del tipo de problema
# =========================================================================== #
def detect_task(y: pd.Series, force: str | None = None) -> str:
    """Devuelve 'clasificacion_binaria' | 'clasificacion_multiclase' | 'regresion'."""
    if force in ("clasificacion", "regresion"):
        if force == "regresion":
            return "regresion"
        return ("clasificacion_binaria" if y.nunique(dropna=True) == 2
                else "clasificacion_multiclase")

    y = y.dropna()
    nu = y.nunique()
    if nu <= 1:
        return "regresion"
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "clasificacion_binaria" if nu == 2 else "clasificacion_multiclase"
    # numérica: pocos valores enteros -> clasificación
    is_int = pd.api.types.is_integer_dtype(y) or (y % 1 == 0).all()
    if is_int and nu <= 20 and nu / max(len(y), 1) < 0.05:
        return "clasificacion_binaria" if nu == 2 else "clasificacion_multiclase"
    if nu == 2:
        return "clasificacion_binaria"
    return "regresion"


def is_classification(task: str) -> bool:
    return task.startswith("clasificacion")


# =========================================================================== #
# Catálogo de modelos
# =========================================================================== #
def model_zoo(task: str, n_classes: int = 2, class_weight: str | None = None,
              fast: bool = False) -> dict[str, Any]:
    """Diccionario {nombre: estimador} según el tipo de problema."""
    cw = "balanced" if class_weight == "balanced" else None
    jobs = -1

    if is_classification(task):
        zoo: dict[str, Any] = {
            "Regresión Logística": LogisticRegression(
                max_iter=2000, class_weight=cw, n_jobs=jobs),
            "Árbol de Decisión": DecisionTreeClassifier(
                random_state=RANDOM_STATE, class_weight=cw, max_depth=12),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=jobs,
                class_weight=cw, min_samples_leaf=1),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=jobs,
                class_weight=cw),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=RANDOM_STATE),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                random_state=RANDOM_STATE),
            "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
            "K-Vecinos (KNN)": KNeighborsClassifier(n_neighbors=15, n_jobs=jobs),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True, random_state=RANDOM_STATE, class_weight=cw),
            "Red Neuronal (MLP)": MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                random_state=RANDOM_STATE, early_stopping=True),
            "Ridge Classifier": RidgeClassifier(class_weight=cw),
            "Baseline (mayoritaria)": DummyClassifier(strategy="most_frequent"),
        }
        if HAS_XGB:
            zoo["XGBoost"] = XGBClassifier(
                n_estimators=400, learning_rate=0.08, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                n_jobs=jobs, tree_method="hist", eval_metric="logloss",
                objective="binary:logistic" if n_classes == 2 else "multi:softprob",
            )
        if HAS_LGBM:
            zoo["LightGBM"] = LGBMClassifier(
                n_estimators=400, learning_rate=0.08, num_leaves=31,
                random_state=RANDOM_STATE, n_jobs=jobs, verbose=-1,
                class_weight=cw)
        if HAS_CATBOOST:
            zoo["CatBoost"] = CatBoostClassifier(
                iterations=400, learning_rate=0.08, depth=6, verbose=0,
                random_seed=RANDOM_STATE, allow_writing_files=False)
    else:
        zoo = {
            "Regresión Lineal": LinearRegression(n_jobs=jobs),
            "Ridge": Ridge(random_state=RANDOM_STATE),
            "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
            "Elastic Net": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
            "Árbol de Decisión": DecisionTreeRegressor(
                random_state=RANDOM_STATE, max_depth=12),
            "Random Forest": RandomForestRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=jobs),
            "Extra Trees": ExtraTreesRegressor(
                n_estimators=300, random_state=RANDOM_STATE, n_jobs=jobs),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                random_state=RANDOM_STATE),
            "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
            "K-Vecinos (KNN)": KNeighborsRegressor(n_neighbors=15, n_jobs=jobs),
            "SVM": SVR(),
            "Red Neuronal (MLP)": MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=600,
                random_state=RANDOM_STATE, early_stopping=True),
            "Baseline (media)": DummyRegressor(strategy="mean"),
        }
        if HAS_XGB:
            zoo["XGBoost"] = XGBRegressor(
                n_estimators=400, learning_rate=0.08, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE,
                n_jobs=jobs, tree_method="hist")
        if HAS_LGBM:
            zoo["LightGBM"] = LGBMRegressor(
                n_estimators=400, learning_rate=0.08, num_leaves=31,
                random_state=RANDOM_STATE, n_jobs=jobs, verbose=-1)
        if HAS_CATBOOST:
            zoo["CatBoost"] = CatBoostRegressor(
                iterations=400, learning_rate=0.08, depth=6, verbose=0,
                random_seed=RANDOM_STATE, allow_writing_files=False)

    if fast:
        keep_c = ["Regresión Logística", "Random Forest", "HistGradientBoosting",
                  "XGBoost", "LightGBM", "Baseline (mayoritaria)"]
        keep_r = ["Regresión Lineal", "Random Forest", "HistGradientBoosting",
                  "XGBoost", "LightGBM", "Baseline (media)"]
        keep = keep_c if is_classification(task) else keep_r
        zoo = {k: v for k, v in zoo.items() if k in keep}
    return zoo


DEFAULT_SELECTION = {
    "clasificacion": ["Regresión Logística", "Random Forest", "XGBoost",
                      "LightGBM", "HistGradientBoosting", "Árbol de Decisión",
                      "Baseline (mayoritaria)"],
    "regresion": ["Regresión Lineal", "Random Forest", "XGBoost", "LightGBM",
                  "HistGradientBoosting", "Ridge", "Baseline (media)"],
}


PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [None, 8, 14, 22],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5],
    },
    "Extra Trees": {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
    },
    "XGBoost": {
        "model__n_estimators": [200, 400, 800],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__reg_lambda": [0.5, 1.0, 3.0],
    },
    "LightGBM": {
        "model__n_estimators": [200, 400, 800],
        "model__num_leaves": [15, 31, 63, 127],
        "model__learning_rate": [0.02, 0.05, 0.1],
        "model__min_child_samples": [5, 20, 50],
    },
    "CatBoost": {
        "model__iterations": [200, 400, 800],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.08, 0.15],
    },
    "Gradient Boosting": {
        "model__n_estimators": [100, 250, 500],
        "model__learning_rate": [0.03, 0.08, 0.15],
        "model__max_depth": [2, 3, 5],
    },
    "HistGradientBoosting": {
        "model__max_iter": [150, 300, 500],
        "model__learning_rate": [0.03, 0.08, 0.15],
        "model__max_leaf_nodes": [15, 31, 63],
    },
    "Regresión Logística": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"],
    },
    "Ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1]},
    "Elastic Net": {"model__alpha": [0.001, 0.01, 0.1, 1],
                    "model__l1_ratio": [0.2, 0.5, 0.8]},
    "Árbol de Decisión": {
        "model__max_depth": [3, 5, 8, 12, None],
        "model__min_samples_leaf": [1, 5, 15, 30],
    },
    "K-Vecinos (KNN)": {
        "model__n_neighbors": [3, 5, 11, 21, 35],
        "model__weights": ["uniform", "distance"],
    },
    "SVM": {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]},
}


# =========================================================================== #
# Resultado
# =========================================================================== #
@dataclass
class ModelResult:
    nombre: str
    pipeline: Any
    metricas_test: dict = field(default_factory=dict)
    metricas_train: dict = field(default_factory=dict)
    metricas_cv: dict = field(default_factory=dict)
    tiempo_entrenamiento: float = 0.0
    y_pred: np.ndarray | None = None
    y_proba: np.ndarray | None = None
    mejores_parametros: dict | None = None
    error: str | None = None


@dataclass
class TrainingOutput:
    task: str
    target: str
    features: list[str]
    class_names: list[str] | None
    label_encoder: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    results: dict[str, ModelResult]
    leaderboard: pd.DataFrame
    best_model: str
    preprocessor_info: dict
    config: dict


# =========================================================================== #
# Métricas
# =========================================================================== #
def classification_metrics(y_true, y_pred, y_proba=None, n_classes: int = 2) -> dict:
    avg = "binary" if n_classes == 2 else "macro"
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            if n_classes == 2:
                p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                m["roc_auc"] = roc_auc_score(y_true, p)
                m["pr_auc"] = average_precision_score(y_true, p)
                m["log_loss"] = log_loss(y_true, np.clip(p, 1e-9, 1 - 1e-9))
            else:
                m["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                             average="macro")
                m["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            pass
    return {k: float(v) for k, v in m.items()}


def regression_metrics(y_true, y_pred, n_features: int = 1) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    denom = max(n - n_features - 1, 1)
    r2_adj = 1 - (1 - r2) * (n - 1) / denom

    nz = np.abs(y_true) > 1e-9
    mape = (float(mean_absolute_percentage_error(y_true[nz], y_pred[nz])) * 100
            if nz.any() else np.nan)
    smape = float(np.mean(
        2 * np.abs(y_pred - y_true) / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-9, None)
    ) * 100)
    rng = float(np.max(y_true) - np.min(y_true)) or 1.0

    return {
        "r2": float(r2),
        "r2_ajustado": float(r2_adj),
        "rmse": rmse,
        "mae": float(mae),
        "mape_%": float(mape) if mape == mape else None,
        "smape_%": smape,
        "mediana_error_abs": float(median_absolute_error(y_true, y_pred)),
        "varianza_explicada": float(explained_variance_score(y_true, y_pred)),
        "rmse_normalizado_%": rmse / rng * 100,
    }


PRIMARY_METRIC = {
    "clasificacion_binaria": "roc_auc",
    "clasificacion_multiclase": "f1_weighted",
    "regresion": "r2",
}

METRIC_LABELS = {
    "accuracy": "Accuracy", "balanced_accuracy": "Accuracy balanceada",
    "precision": "Precisión", "recall": "Recall (sensibilidad)",
    "f1": "F1", "f1_weighted": "F1 ponderado",
    "precision_macro": "Precisión macro", "recall_macro": "Recall macro",
    "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "log_loss": "Log Loss",
    "mcc": "Matthews (MCC)", "kappa": "Kappa de Cohen",
    "r2": "R²", "r2_ajustado": "R² ajustado", "rmse": "RMSE", "mae": "MAE",
    "mape_%": "MAPE (%)", "smape_%": "sMAPE (%)",
    "mediana_error_abs": "Error absoluto mediano",
    "varianza_explicada": "Varianza explicada",
    "rmse_normalizado_%": "RMSE normalizado (%)",
}

METRICS_LOWER_BETTER = {"log_loss", "rmse", "mae", "mape_%", "smape_%",
                        "mediana_error_abs", "rmse_normalizado_%"}


# =========================================================================== #
# Entrenamiento
# =========================================================================== #
def train_models(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None = None,
    task: str | None = None,
    selected_models: list[str] | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    do_cv: bool = True,
    tune: bool = False,
    tune_iter: int = 15,
    scaler: str = "standard",
    encoder: str = "onehot",
    num_impute: str = "median",
    cat_impute: str = "most_frequent",
    class_weight: str | None = None,
    stratify: bool = True,
    random_state: int = RANDOM_STATE,
    progress_cb=None,
) -> TrainingOutput:
    """Entrena y compara varios modelos sobre el mismo split."""
    data = df.dropna(subset=[target]).copy()
    feats = [c for c in (features or [c for c in data.columns if c != target])
             if c in data.columns and c != target]
    if not feats:
        raise ValueError("No hay variables predictoras seleccionadas.")

    X = data[feats]
    y_raw = data[target]
    task = task or detect_task(y_raw)
    clf = is_classification(task)

    label_encoder, class_names = None, None
    if clf:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_names = [str(c) for c in label_encoder.classes_]
        n_classes = len(class_names)
        if n_classes < 2:
            raise ValueError("La variable objetivo debe tener al menos 2 clases.")
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        X, y = X.loc[ok], y[ok]
        n_classes = 0

    strat = y if (clf and stratify and pd.Series(y).value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat)

    pre_template, pre_info = build_preprocessor(
        X, num_impute=num_impute, cat_impute=cat_impute,
        scaler=scaler, encoder=encoder)

    zoo = model_zoo(task, n_classes=max(n_classes, 2), class_weight=class_weight)
    names = selected_models or [n for n in DEFAULT_SELECTION[
        "clasificacion" if clf else "regresion"] if n in zoo]
    names = [n for n in names if n in zoo]
    if not names:
        names = list(zoo.keys())[:6]

    # Con pocos datos (o clases pequeñas) no se pueden pedir tantas particiones
    if clf:
        min_clase = int(pd.Series(y_train).value_counts().min())
        cv_folds = max(2, min(cv_folds, min_clase))
    else:
        cv_folds = max(2, min(cv_folds, len(X_train)))
    cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
          if clf else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state))
    scoring = _cv_scoring(task)

    results: dict[str, ModelResult] = {}
    for i, name in enumerate(names):
        if progress_cb:
            progress_cb(i / len(names), f"Entrenando {name}…")
        t0 = time.perf_counter()
        try:
            pipe = Pipeline([("pre", clone(pre_template)), ("model", clone(zoo[name]))])
            best_params = None

            if tune and name in PARAM_GRIDS:
                search = RandomizedSearchCV(
                    pipe, PARAM_GRIDS[name], n_iter=tune_iter, cv=cv,
                    scoring=scoring, n_jobs=-1, random_state=random_state,
                    error_score="raise", refit=True)
                search.fit(X_train, y_train)
                pipe = search.best_estimator_
                best_params = {k.replace("model__", ""): v
                               for k, v in search.best_params_.items()}
            else:
                pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_pred_tr = pipe.predict(X_train)
            y_proba = None
            if clf and hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_test)
                except Exception:
                    y_proba = None

            if clf:
                mt = classification_metrics(y_test, y_pred, y_proba, n_classes)
                mtr = classification_metrics(y_train, y_pred_tr, None, n_classes)
            else:
                nf = X_train.shape[1]
                mt = regression_metrics(y_test, y_pred, nf)
                mtr = regression_metrics(y_train, y_pred_tr, nf)

            mcv = {}
            if do_cv:
                try:
                    cvres = cross_validate(
                        clone(pipe), X_train, y_train, cv=cv, scoring=scoring,
                        n_jobs=-1, error_score=np.nan)
                    for key in scoring:
                        vals = cvres[f"test_{key}"]
                        mcv[key] = float(np.nanmean(vals))
                        mcv[f"{key}_std"] = float(np.nanstd(vals))
                except Exception:
                    pass

            results[name] = ModelResult(
                nombre=name, pipeline=pipe, metricas_test=mt, metricas_train=mtr,
                metricas_cv=mcv, tiempo_entrenamiento=time.perf_counter() - t0,
                y_pred=y_pred, y_proba=y_proba, mejores_parametros=best_params)
        except Exception as e:
            results[name] = ModelResult(nombre=name, pipeline=None,
                                        error=f"{type(e).__name__}: {e}",
                                        tiempo_entrenamiento=time.perf_counter() - t0)

    if progress_cb:
        progress_cb(1.0, "Listo")

    lb = build_leaderboard(results, task)
    best = lb.iloc[0]["modelo"] if len(lb) else ""

    return TrainingOutput(
        task=task, target=target, features=feats, class_names=class_names,
        label_encoder=label_encoder, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, results=results, leaderboard=lb,
        best_model=best, preprocessor_info=pre_info,
        config={"test_size": test_size, "cv_folds": cv_folds, "tune": tune,
                "scaler": scaler, "encoder": encoder, "class_weight": class_weight,
                "random_state": random_state, "n_train": len(X_train),
                "n_test": len(X_test)},
    )


def _cv_scoring(task: str) -> list[str]:
    if task == "clasificacion_binaria":
        return ["accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall"]
    if task == "clasificacion_multiclase":
        return ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]
    return ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]


def build_leaderboard(results: dict[str, ModelResult], task: str) -> pd.DataFrame:
    """Tabla comparativa ordenada por la métrica principal."""
    clf = is_classification(task)
    rows = []
    for name, r in results.items():
        if r.error:
            rows.append({"modelo": name, "estado": "error", "detalle": r.error})
            continue
        row = {"modelo": name, "estado": "ok"}
        row.update({METRIC_LABELS.get(k, k): round(v, 4)
                    for k, v in r.metricas_test.items() if v is not None})
        prim = PRIMARY_METRIC[task]
        if prim in r.metricas_cv:
            row["CV media"] = round(r.metricas_cv[prim], 4)
            row["CV desv"] = round(r.metricas_cv.get(f"{prim}_std", 0), 4)
        elif "r2" in r.metricas_cv:
            row["CV media"] = round(r.metricas_cv["r2"], 4)
        # Sobreajuste
        key = "accuracy" if clf else "r2"
        if key in r.metricas_train and key in r.metricas_test:
            row["sobreajuste"] = round(r.metricas_train[key] - r.metricas_test[key], 4)
        row["tiempo_s"] = round(r.tiempo_entrenamiento, 2)
        rows.append(row)

    lb = pd.DataFrame(rows)
    if lb.empty:
        return lb
    prim_label = METRIC_LABELS.get(PRIMARY_METRIC[task], PRIMARY_METRIC[task])
    sort_col = prim_label if prim_label in lb.columns else (
        "Accuracy" if "Accuracy" in lb.columns else ("R²" if "R²" in lb.columns else None))
    ok = lb[lb["estado"] == "ok"]
    bad = lb[lb["estado"] != "ok"]
    if sort_col and sort_col in ok.columns:
        ok = ok.sort_values(sort_col, ascending=False)
    ok = ok.reset_index(drop=True)
    ok.insert(0, "ranking", range(1, len(ok) + 1))
    return pd.concat([ok, bad], ignore_index=True)


# =========================================================================== #
# Diagnósticos del mejor modelo
# =========================================================================== #
def confusion(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def report_table(y_true, y_pred, class_names=None) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.round(4)
    df.index.name = "clase"
    return df.reset_index()


def roc_data(y_true, y_proba, n_classes: int, class_names=None) -> list[dict]:
    out = []
    if n_classes == 2:
        p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        fpr, tpr, _ = roc_curve(y_true, p)
        out.append({"clase": "positiva", "fpr": fpr, "tpr": tpr,
                    "auc": roc_auc_score(y_true, p)})
    else:
        for i in range(n_classes):
            yb = (np.asarray(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(yb, y_proba[:, i])
            nombre = class_names[i] if class_names else str(i)
            out.append({"clase": nombre, "fpr": fpr, "tpr": tpr,
                        "auc": roc_auc_score(yb, y_proba[:, i])})
    return out


def pr_data(y_true, y_proba, n_classes: int, class_names=None) -> list[dict]:
    out = []
    if n_classes == 2:
        p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        pr, rc, _ = precision_recall_curve(y_true, p)
        out.append({"clase": "positiva", "precision": pr, "recall": rc,
                    "ap": average_precision_score(y_true, p)})
    else:
        for i in range(n_classes):
            yb = (np.asarray(y_true) == i).astype(int)
            pr, rc, _ = precision_recall_curve(yb, y_proba[:, i])
            nombre = class_names[i] if class_names else str(i)
            out.append({"clase": nombre, "precision": pr, "recall": rc,
                        "ap": average_precision_score(yb, y_proba[:, i])})
    return out


def threshold_analysis(y_true, y_proba, steps: int = 101) -> pd.DataFrame:
    """Métricas en función del umbral de decisión (clasificación binaria)."""
    p = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    rows = []
    for t in np.linspace(0.01, 0.99, steps):
        pred = (p >= t).astype(int)
        rows.append({
            "umbral": round(float(t), 3),
            "accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "positivos_predichos_%": float(pred.mean() * 100),
        })
    return pd.DataFrame(rows).round(4)


def learning_curve_data(pipeline, X, y, cv=5, scoring=None, task="regresion"):
    scoring = scoring or ("accuracy" if is_classification(task) else "r2")
    sizes, train_s, val_s = learning_curve(
        clone(pipeline), X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), random_state=RANDOM_STATE)
    return pd.DataFrame({
        "n_muestras": sizes,
        "entrenamiento": train_s.mean(axis=1),
        "validacion": val_s.mean(axis=1),
        "entrenamiento_std": train_s.std(axis=1),
        "validacion_std": val_s.std(axis=1),
    })


def residuals_frame(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = y_true - y_pred
    return pd.DataFrame({
        "real": y_true, "predicho": y_pred, "residuo": res,
        "residuo_abs": np.abs(res),
        "error_%": np.where(np.abs(y_true) > 1e-9, res / y_true * 100, np.nan),
    })
