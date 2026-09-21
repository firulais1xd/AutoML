"""
Variables clave: importancia nativa, por permutación, coeficientes, SHAP
y dependencia parcial. Todo se devuelve agregado a las columnas ORIGINALES
(se suman las columnas one-hot de una misma variable).
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# --------------------------------------------------------------------------- #
# Utilidades de mapeo one-hot -> variable original
# --------------------------------------------------------------------------- #
def _feature_names(pipeline) -> list[str]:
    """Nombres de las columnas después del preprocesamiento, con respaldos."""
    pre = pipeline.named_steps.get("pre")
    try:
        return [str(f) for f in pre.get_feature_names_out()]
    except Exception:
        pass
    # Respaldo: reconstruir bloque por bloque
    try:
        names: list[str] = []
        for nombre, trans, cols in pre.transformers_:
            if trans in ("drop", None):
                continue
            if trans == "passthrough":
                names += [str(c) for c in cols]
                continue
            try:
                names += [str(f) for f in trans.get_feature_names_out(cols)]
            except Exception:
                try:
                    enc = trans.named_steps.get("encoder")
                    names += [str(f) for f in enc.get_feature_names_out(cols)]
                except Exception:
                    names += [str(c) for c in cols]
        if names:
            return names
    except Exception:
        pass
    n = getattr(pre, "n_features_in_", 0)
    return [f"f{i}" for i in range(n)]


def map_to_original(feature_names: list[str], original_cols: list[str]) -> list[str]:
    """Asocia cada columna transformada con su variable original."""
    out = []
    ordered = sorted(original_cols, key=len, reverse=True)
    for f in feature_names:
        base = f
        for oc in ordered:
            if f == oc or f.startswith(f"{oc}_") or f.startswith(f"{oc}__"):
                base = oc
                break
        out.append(base)
    return out


def _aggregate(values: np.ndarray, feature_names: list[str],
               original_cols: list[str]) -> pd.DataFrame:
    base = map_to_original(feature_names, original_cols)
    df = pd.DataFrame({"variable": base, "importancia": np.abs(np.asarray(values, dtype=float))})
    agg = df.groupby("variable", as_index=False)["importancia"].sum()
    total = agg["importancia"].sum()
    agg["importancia_%"] = (agg["importancia"] / total * 100) if total > 0 else 0.0
    agg = agg.sort_values("importancia", ascending=False).reset_index(drop=True)
    agg["acumulado_%"] = agg["importancia_%"].cumsum().round(2)
    agg["importancia"] = agg["importancia"].round(6)
    agg["importancia_%"] = agg["importancia_%"].round(2)
    return agg


# --------------------------------------------------------------------------- #
# 1) Importancia nativa del modelo
# --------------------------------------------------------------------------- #
def native_importance(pipeline, original_cols: list[str]) -> pd.DataFrame | None:
    model = pipeline.named_steps.get("model")
    names = _feature_names(pipeline)
    vals = None
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        vals = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    if vals is None:
        return None
    if len(vals) != len(names):
        names = [f"f{i}" for i in range(len(vals))]
    return _aggregate(vals, names, original_cols)


def signed_coefficients(pipeline, original_cols: list[str]) -> pd.DataFrame | None:
    """Coeficientes con signo para modelos lineales: dirección del efecto."""
    model = pipeline.named_steps.get("model")
    if not hasattr(model, "coef_"):
        return None
    names = _feature_names(pipeline)
    coef = np.asarray(model.coef_, dtype=float)
    vals = coef.mean(axis=0) if coef.ndim > 1 else coef
    if len(vals) != len(names):
        return None
    df = pd.DataFrame({"feature": names, "coeficiente": vals})
    df["variable"] = map_to_original(names, original_cols)
    df["efecto"] = np.where(df["coeficiente"] > 0, "aumenta el objetivo",
                            "disminuye el objetivo")
    return df.reindex(df["coeficiente"].abs().sort_values(ascending=False).index
                      ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2) Importancia por permutación (agnóstica al modelo)
# --------------------------------------------------------------------------- #
def permutation_based(pipeline, X: pd.DataFrame, y, scoring: str | None = None,
                      n_repeats: int = 8, random_state: int = 42,
                      max_rows: int = 4000) -> pd.DataFrame:
    if len(X) > max_rows:
        idx = np.random.RandomState(random_state).choice(len(X), max_rows, replace=False)
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(pipeline, X, y, scoring=scoring,
                               n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    df = pd.DataFrame({
        "variable": list(X.columns),
        "importancia": r.importances_mean,
        "desv_std": r.importances_std,
    })
    df["importancia"] = df["importancia"].clip(lower=0)
    total = df["importancia"].sum()
    df["importancia_%"] = (df["importancia"] / total * 100).round(2) if total > 0 else 0.0
    df = df.sort_values("importancia", ascending=False).reset_index(drop=True)
    df["acumulado_%"] = df["importancia_%"].cumsum().round(2)
    return df.round(6)


# --------------------------------------------------------------------------- #
# 3) SHAP
# --------------------------------------------------------------------------- #
def shap_values(pipeline, X: pd.DataFrame, original_cols: list[str],
                max_rows: int = 500, random_state: int = 42):
    """
    Devuelve (df_importancia_agregada, matriz_shap, X_transformada, feature_names).
    Devuelve (None, None, None, None) si SHAP no está disponible o falla.
    """
    if not HAS_SHAP:
        return None, None, None, None
    try:
        Xs = X.sample(min(max_rows, len(X)), random_state=random_state) if len(X) > max_rows else X
        pre = pipeline.named_steps["pre"]
        model = pipeline.named_steps["model"]
        Xt = pre.transform(Xs)
        Xt = np.asarray(Xt.todense()) if hasattr(Xt, "todense") else np.asarray(Xt)
        names = _feature_names(pipeline)
        if len(names) != Xt.shape[1]:
            names = [f"f{i}" for i in range(Xt.shape[1])]

        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xt, check_additivity=False)
        except Exception:
            bg = shap.sample(Xt, min(80, len(Xt)), random_state=random_state)
            explainer = shap.Explainer(model, bg)
            sv = explainer(Xt).values

        sv = np.asarray(sv)
        if sv.ndim == 3:                      # (n, f, clases) o (clases, n, f)
            sv = np.abs(sv).mean(axis=2) if sv.shape[2] <= sv.shape[0] else np.abs(sv).mean(axis=0)
        mean_abs = np.abs(sv).mean(axis=0)
        agg = _aggregate(mean_abs, names, original_cols)
        return agg, sv, pd.DataFrame(Xt, columns=names), names
    except Exception:
        return None, None, None, None


# --------------------------------------------------------------------------- #
# 4) Dependencia parcial (efecto marginal)
# --------------------------------------------------------------------------- #
def partial_dependence_curve(pipeline, X: pd.DataFrame, feature: str,
                             grid_resolution: int = 25, target_class: int = 1):
    """Curva PDP de una variable: cómo cambia la predicción al moverla."""
    try:
        kw = {}
        model = pipeline.named_steps.get("model")
        if hasattr(model, "predict_proba"):
            kw["response_method"] = "predict_proba"
            n_cls = len(getattr(model, "classes_", [0, 1]))
            if n_cls > 2:
                kw["target"] = target_class
        pd_res = partial_dependence(
            pipeline, X, features=[feature], grid_resolution=grid_resolution,
            kind="average", **kw)
        grid = np.asarray(pd_res["grid_values"][0])
        avg = np.asarray(pd_res["average"])
        y = avg[0] if avg.ndim > 1 else avg
        return pd.DataFrame({feature: grid, "prediccion_promedio": y})
    except Exception:
        return None


def top_features(imp_df: pd.DataFrame, k: int = 15) -> list[str]:
    if imp_df is None or imp_df.empty:
        return []
    return imp_df.head(k)["variable"].tolist()


def importance_consensus(frames: dict[str, pd.DataFrame], k: int = 20) -> pd.DataFrame:
    """Combina varios rankings (nativo, permutación, SHAP) en un consenso."""
    ranks = []
    for metodo, df in frames.items():
        if df is None or df.empty:
            continue
        t = df[["variable", "importancia_%"]].copy()
        t["rank"] = t["importancia_%"].rank(ascending=False, method="min")
        t = t.rename(columns={"importancia_%": f"%_{metodo}", "rank": f"rank_{metodo}"})
        ranks.append(t.set_index("variable"))
    if not ranks:
        return pd.DataFrame()
    merged = pd.concat(ranks, axis=1).fillna(0)
    rank_cols = [c for c in merged.columns if c.startswith("rank_")]
    merged["rank_promedio"] = merged[rank_cols].replace(0, np.nan).mean(axis=1)
    pct_cols = [c for c in merged.columns if c.startswith("%_")]
    merged["importancia_promedio_%"] = merged[pct_cols].mean(axis=1).round(2)
    out = (merged.sort_values("rank_promedio")
           .reset_index().rename(columns={"index": "variable"}).head(k))
    return out.round(3)
