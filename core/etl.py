"""
ETL: transformaciones reproducibles sobre el DataFrame + pipeline de
preprocesamiento para los modelos.

Cada transformación se registra como una "receta" (lista de dicts) que puede
re-aplicarse, exportarse a JSON o traducirse a código Python.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler,
    RobustScaler, PowerTransformer, FunctionTransformer,
)

from .schema import (
    is_numeric, is_texty, is_datetime, try_parse_numeric, try_parse_datetime,
)


# =========================================================================== #
# Operaciones individuales
# =========================================================================== #
def op_drop_columns(df: pd.DataFrame, columns: list[str], **_) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def op_rename(df: pd.DataFrame, mapping: dict, **_) -> pd.DataFrame:
    return df.rename(columns=mapping)


def op_normalize_names(df: pd.DataFrame, **_) -> pd.DataFrame:
    from .io_utils import normalize_column_names
    return normalize_column_names(df)


def op_drop_duplicates(df: pd.DataFrame, subset: list[str] | None = None,
                       keep: str = "first", **_) -> pd.DataFrame:
    subset = [c for c in (subset or []) if c in df.columns] or None
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def op_drop_empty_rows(df: pd.DataFrame, how: str = "all",
                       thresh_pct: float | None = None, **_) -> pd.DataFrame:
    if thresh_pct is not None:
        min_ok = int(np.ceil(df.shape[1] * (1 - thresh_pct / 100)))
        return df.dropna(thresh=min_ok).reset_index(drop=True)
    return df.dropna(how=how).reset_index(drop=True)


def op_convert_types(df: pd.DataFrame, mapping: dict[str, str], **_) -> pd.DataFrame:
    out = df.copy()
    for col, target in mapping.items():
        if col not in out.columns:
            continue
        s = out[col]
        try:
            if target == "numerica":
                conv = try_parse_numeric(s, min_ratio=0.0)
                out[col] = conv if conv is not None else pd.to_numeric(s, errors="coerce")
            elif target == "fecha":
                conv = try_parse_datetime(s, min_ratio=0.0)
                out[col] = conv if conv is not None else pd.to_datetime(s, errors="coerce")
            elif target == "texto":
                out[col] = s.astype(str).where(s.notna())
            elif target == "categorica":
                out[col] = s.astype("category")
            elif target == "booleana":
                out[col] = _to_bool(s)
            elif target == "entero":
                out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
        except Exception:
            pass
    return out


def _to_bool(s: pd.Series) -> pd.Series:
    truthy = {"1", "true", "si", "sí", "yes", "y", "t", "verdadero"}
    falsy = {"0", "false", "no", "n", "f", "falso"}
    t = s.astype(str).str.strip().str.lower()
    return t.map(lambda v: True if v in truthy else (False if v in falsy else None)).astype("boolean")


def op_clean_text(df: pd.DataFrame, columns: list[str] | None = None,
                  strip: bool = True, case: str = "none",
                  remove_accents: bool = False, collapse_spaces: bool = True,
                  **_) -> pd.DataFrame:
    import unicodedata
    out = df.copy()
    cols = columns or [c for c in out.columns if is_texty(out[c])]
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        if strip:
            s = s.str.strip()
        if collapse_spaces:
            s = s.str.replace(r"\s+", " ", regex=True)
        if case == "lower":
            s = s.str.lower()
        elif case == "upper":
            s = s.str.upper()
        elif case == "title":
            s = s.str.title()
        if remove_accents:
            s = s.map(lambda v: unicodedata.normalize("NFKD", v)
                      .encode("ascii", "ignore").decode() if isinstance(v, str) else v)
        out[c] = s
    return out


def op_replace_blanks_with_na(df: pd.DataFrame, **_) -> pd.DataFrame:
    from .quality import BLANK_TOKENS
    out = df.copy()
    for c in out.columns:
        if is_texty(out[c]):
            s = out[c].astype("string")
            mask = s.str.strip().str.lower().isin(list(BLANK_TOKENS))
            out[c] = s.mask(mask.fillna(False))
    return out


def op_impute(df: pd.DataFrame, strategy_num: str = "mediana",
              strategy_cat: str = "moda", columns: list[str] | None = None,
              constant_value: Any = None, add_flag: bool = False,
              knn_neighbors: int = 5, **_) -> pd.DataFrame:
    out = df.copy()
    cols = columns or list(out.columns)

    if add_flag:
        for c in cols:
            if c in out.columns and out[c].isna().any():
                out[f"{c}__faltante"] = out[c].isna().astype(int)

    num_cols = [c for c in cols if c in out.columns and is_numeric(out[c])]
    cat_cols = [c for c in cols if c in out.columns and not is_numeric(out[c])
                and not is_datetime(out[c])]

    if num_cols and strategy_num != "ninguna":
        if strategy_num == "knn":
            sub = out[num_cols]
            if sub.notna().any().any():
                imp = KNNImputer(n_neighbors=knn_neighbors)
                out[num_cols] = imp.fit_transform(sub)
        else:
            for c in num_cols:
                if strategy_num == "media":
                    v = out[c].mean()
                elif strategy_num == "mediana":
                    v = out[c].median()
                elif strategy_num == "cero":
                    v = 0
                elif strategy_num == "constante":
                    v = constant_value if constant_value is not None else 0
                elif strategy_num == "interpolar":
                    out[c] = out[c].interpolate(limit_direction="both")
                    continue
                elif strategy_num == "ffill":
                    out[c] = out[c].ffill().bfill()
                    continue
                else:
                    continue
                out[c] = out[c].fillna(v)

    if cat_cols and strategy_cat != "ninguna":
        for c in cat_cols:
            if strategy_cat == "moda":
                m = out[c].mode(dropna=True)
                v = m.iloc[0] if len(m) else "DESCONOCIDO"
            elif strategy_cat == "constante":
                v = constant_value if constant_value is not None else "DESCONOCIDO"
            elif strategy_cat == "ffill":
                out[c] = out[c].ffill().bfill()
                continue
            else:
                continue
            if isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].cat.add_categories([v]) if v not in out[c].cat.categories else out[c]
            out[c] = out[c].fillna(v)

    return out


def op_drop_na_rows(df: pd.DataFrame, columns: list[str] | None = None, **_) -> pd.DataFrame:
    cols = [c for c in (columns or []) if c in df.columns] or None
    return df.dropna(subset=cols).reset_index(drop=True)


def op_outliers(df: pd.DataFrame, columns: list[str] | None = None,
                method: str = "iqr", action: str = "winsorizar",
                factor: float = 1.5, lower_q: float = 0.01,
                upper_q: float = 0.99, **_) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in (columns or [c for c in out.columns if is_numeric(out[c])])
            if c in out.columns and is_numeric(out[c])]
    mask_total = pd.Series(False, index=out.index)

    for c in cols:
        s = out[c]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        elif method == "zscore":
            mu, sd = s.mean(), s.std(ddof=0)
            lo, hi = mu - factor * sd, mu + factor * sd
        else:  # percentil
            lo, hi = s.quantile(lower_q), s.quantile(upper_q)

        if action == "winsorizar":
            out[c] = s.clip(lower=lo, upper=hi)
        elif action == "nulo":
            out[c] = s.mask((s < lo) | (s > hi))
        else:  # eliminar filas
            mask_total |= ((s < lo) | (s > hi)).fillna(False)

    if action == "eliminar":
        out = out.loc[~mask_total].reset_index(drop=True)
    return out


def op_filter_rows(df: pd.DataFrame, query: str = "", **_) -> pd.DataFrame:
    if not query.strip():
        return df
    return df.query(query).reset_index(drop=True)


def op_new_column(df: pd.DataFrame, name: str = "nueva", expression: str = "",
                  **_) -> pd.DataFrame:
    out = df.copy()
    if not expression.strip():
        return out
    try:
        out[name] = out.eval(expression, engine="python")
    except Exception:
        env = {"np": np, "pd": pd, **{c: out[c] for c in out.columns}}
        out[name] = eval(expression, {"__builtins__": {}}, env)  # noqa: S307
    return out


def op_date_features(df: pd.DataFrame, columns: list[str] | None = None,
                     parts: list[str] | None = None, drop_original: bool = False,
                     **_) -> pd.DataFrame:
    out = df.copy()
    parts = parts or ["anio", "mes", "dia", "dia_semana", "trimestre"]
    cols = columns or [c for c in out.columns if is_datetime(out[c])]
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c] if is_datetime(out[c]) else pd.to_datetime(out[c], errors="coerce")
        if "anio" in parts:
            out[f"{c}__anio"] = s.dt.year
        if "mes" in parts:
            out[f"{c}__mes"] = s.dt.month
        if "dia" in parts:
            out[f"{c}__dia"] = s.dt.day
        if "dia_semana" in parts:
            out[f"{c}__dia_semana"] = s.dt.dayofweek
        if "semana" in parts:
            out[f"{c}__semana"] = s.dt.isocalendar().week.astype("Int64")
        if "trimestre" in parts:
            out[f"{c}__trimestre"] = s.dt.quarter
        if "hora" in parts:
            out[f"{c}__hora"] = s.dt.hour
        if "es_fin_semana" in parts:
            out[f"{c}__es_fin_semana"] = (s.dt.dayofweek >= 5).astype("Int64")
        if drop_original:
            out = out.drop(columns=[c])
    return out


def op_group_rare(df: pd.DataFrame, columns: list[str] | None = None,
                  min_freq_pct: float = 1.0, other_label: str = "OTROS",
                  **_) -> pd.DataFrame:
    out = df.copy()
    cols = columns or [c for c in out.columns if is_texty(out[c])]
    n = len(out)
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        freq = s.value_counts(normalize=True) * 100
        rare = set(freq[freq < min_freq_pct].index)
        if rare:
            out[c] = s.mask(s.isin(rare), other_label)
    return out


def op_transform_numeric(df: pd.DataFrame, columns: list[str] | None = None,
                         method: str = "log1p", **_) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in (columns or []) if c in out.columns and is_numeric(out[c])]
    for c in cols:
        s = out[c].astype(float)
        if method == "log1p":
            out[c] = np.log1p(s - min(0, s.min()))
        elif method == "sqrt":
            out[c] = np.sqrt(s - min(0, s.min()))
        elif method == "boxcox_yeo":
            pt = PowerTransformer(method="yeo-johnson")
            out[c] = pt.fit_transform(s.to_frame().fillna(s.median()))[:, 0]
        elif method == "zscore":
            out[c] = (s - s.mean()) / (s.std(ddof=0) or 1)
        elif method == "minmax":
            rng = (s.max() - s.min()) or 1
            out[c] = (s - s.min()) / rng
    return out


def op_binning(df: pd.DataFrame, column: str = "", bins: int = 5,
               method: str = "cuantiles", labels: list[str] | None = None,
               new_name: str | None = None, **_) -> pd.DataFrame:
    out = df.copy()
    if column not in out.columns or not is_numeric(out[column]):
        return out
    name = new_name or f"{column}__bin"
    try:
        if method == "cuantiles":
            out[name] = pd.qcut(out[column], q=bins, labels=labels, duplicates="drop")
        else:
            out[name] = pd.cut(out[column], bins=bins, labels=labels)
        out[name] = out[name].astype("string")
    except Exception:
        pass
    return out


def op_sample(df: pd.DataFrame, n: int | None = None, frac: float | None = None,
              random_state: int = 42, **_) -> pd.DataFrame:
    if n:
        return df.sample(n=min(n, len(df)), random_state=random_state).reset_index(drop=True)
    if frac:
        return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    return df


OPERATIONS: dict[str, Callable[..., pd.DataFrame]] = {
    "eliminar_columnas": op_drop_columns,
    "renombrar": op_rename,
    "normalizar_nombres": op_normalize_names,
    "eliminar_duplicados": op_drop_duplicates,
    "eliminar_filas_vacias": op_drop_empty_rows,
    "convertir_tipos": op_convert_types,
    "limpiar_texto": op_clean_text,
    "vacios_a_nulos": op_replace_blanks_with_na,
    "imputar": op_impute,
    "eliminar_filas_nulas": op_drop_na_rows,
    "outliers": op_outliers,
    "filtrar": op_filter_rows,
    "nueva_columna": op_new_column,
    "features_fecha": op_date_features,
    "agrupar_raras": op_group_rare,
    "transformar_numerica": op_transform_numeric,
    "discretizar": op_binning,
    "muestrear": op_sample,
}

OPERATION_LABELS = {
    "eliminar_columnas": "Eliminar columnas",
    "renombrar": "Renombrar columnas",
    "normalizar_nombres": "Normalizar nombres de columnas",
    "eliminar_duplicados": "Eliminar filas duplicadas",
    "eliminar_filas_vacias": "Eliminar filas vacías",
    "convertir_tipos": "Convertir tipos de dato",
    "limpiar_texto": "Limpiar texto",
    "vacios_a_nulos": "Convertir vacíos en nulos",
    "imputar": "Imputar valores faltantes",
    "eliminar_filas_nulas": "Eliminar filas con nulos",
    "outliers": "Tratar outliers",
    "filtrar": "Filtrar filas",
    "nueva_columna": "Crear columna calculada",
    "features_fecha": "Extraer componentes de fecha",
    "agrupar_raras": "Agrupar categorías infrecuentes",
    "transformar_numerica": "Transformar variable numérica",
    "discretizar": "Discretizar (binning)",
    "muestrear": "Muestrear filas",
}


# =========================================================================== #
# Receta
# =========================================================================== #
def apply_recipe(df: pd.DataFrame, recipe: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica la lista de operaciones en orden. Devuelve (df, bitácora)."""
    out = df.copy()
    log: list[dict] = []
    for i, step in enumerate(recipe, start=1):
        op = step.get("op")
        params = {k: v for k, v in step.items() if k != "op"}
        fn = OPERATIONS.get(op)
        if fn is None:
            log.append({"paso": i, "operacion": op, "estado": "desconocida"})
            continue
        before = out.shape
        try:
            out = fn(out, **params)
            log.append({
                "paso": i,
                "operacion": OPERATION_LABELS.get(op, op),
                "estado": "ok",
                "filas_antes": before[0], "filas_despues": out.shape[0],
                "cols_antes": before[1], "cols_despues": out.shape[1],
                "detalle": _short(params),
            })
        except Exception as e:
            log.append({"paso": i, "operacion": OPERATION_LABELS.get(op, op),
                        "estado": f"error: {e}", "detalle": _short(params)})
    return out, log


def _short(params: dict, limit: int = 90) -> str:
    s = ", ".join(f"{k}={v}" for k, v in params.items())
    return s[:limit] + ("…" if len(s) > limit else "")


def recipe_to_code(recipe: list[dict], var: str = "df") -> str:
    """Traduce la receta a código Python reproducible."""
    lines = [
        "# Código generado automáticamente por AutoML Studio",
        "import pandas as pd, numpy as np",
        "from core.etl import apply_recipe",
        "",
        f"{var} = pd.read_csv('tu_archivo.csv')  # o pd.read_excel(...)",
        "",
        "recipe = [",
    ]
    for step in recipe:
        lines.append(f"    {step!r},")
    lines += ["]", f"{var}, log = apply_recipe({var}, recipe)", f"print({var}.shape)"]
    return "\n".join(lines)


def auto_recipe(df: pd.DataFrame) -> list[dict]:
    """Receta de limpieza sugerida a partir del diagnóstico de calidad."""
    from .quality import dataset_diagnostics
    d = dataset_diagnostics(df)
    recipe: list[dict] = [{"op": "vacios_a_nulos"}]

    if d["filas_totalmente_vacias"] > 0:
        recipe.append({"op": "eliminar_filas_vacias", "how": "all"})
    if d["filas_duplicadas"] > 0:
        recipe.append({"op": "eliminar_duplicados", "keep": "first"})

    drop = list(d["columnas_constantes"])
    for grupo in d["columnas_duplicadas"]:
        drop += grupo[1:]
    drop += [c for c in d["columnas_muy_nulas"] if c not in drop]
    if drop:
        recipe.append({"op": "eliminar_columnas", "columns": sorted(set(drop))})

    if d["columnas_convertibles"]:
        recipe.append({"op": "convertir_tipos", "mapping": d["columnas_convertibles"]})

    recipe.append({"op": "limpiar_texto", "strip": True, "collapse_spaces": True})
    recipe.append({"op": "imputar", "strategy_num": "mediana", "strategy_cat": "moda"})
    return recipe


# =========================================================================== #
# Pipeline de preprocesamiento para modelos
# =========================================================================== #
def build_preprocessor(
    X: pd.DataFrame,
    num_impute: str = "median",
    cat_impute: str = "most_frequent",
    scaler: str = "standard",
    encoder: str = "onehot",
    max_onehot_cardinality: int = 30,
) -> tuple[ColumnTransformer, dict]:
    """
    Construye el ColumnTransformer y devuelve además el mapa de columnas usadas.
    """
    num_cols = [c for c in X.columns if is_numeric(X[c])]
    dt_cols = [c for c in X.columns if is_datetime(X[c])]
    cat_cols = [c for c in X.columns
                if c not in num_cols and c not in dt_cols]

    # Las fechas se convierten a ordinal (días desde época) dentro del pipeline
    low_card = [c for c in cat_cols if X[c].nunique(dropna=True) <= max_onehot_cardinality]
    high_card = [c for c in cat_cols if c not in low_card]

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "yeo-johnson": PowerTransformer(method="yeo-johnson"),
        "ninguno": "passthrough",
    }
    sc = scalers.get(scaler, StandardScaler())

    num_steps = [("imputer", SimpleImputer(strategy=num_impute))]
    if sc != "passthrough":
        num_steps.append(("scaler", sc))
    num_pipe = Pipeline(num_steps)

    if encoder == "ordinal":
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                min_frequency=0.01)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_impute, fill_value="DESCONOCIDO")),
        ("encoder", cat_enc),
    ])
    high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if low_card:
        transformers.append(("cat", cat_pipe, low_card))
    if high_card:
        transformers.append(("cat_alta", high_pipe, high_card))
    if dt_cols:
        transformers.append(("fecha", Pipeline([
            ("to_num", FunctionTransformer(_datetime_to_numeric, validate=False,
                                           feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
        ]), dt_cols))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    info = {"numericas": num_cols, "categoricas": low_card,
            "categoricas_alta_card": high_card, "fechas": dt_cols}
    return pre, info


def _datetime_to_numeric(X):
    df = pd.DataFrame(X).copy()
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        df[c] = s.astype("int64") / 86_400_000_000_000  # días desde época
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df.to_numpy(dtype=float)


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return [str(f) for f in preprocessor.get_feature_names_out()]
    except Exception:
        return [f"f{i}" for i in range(getattr(preprocessor, "n_features_in_", 0))]
