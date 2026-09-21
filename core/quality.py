"""
Diagnóstico de calidad de datos: nulos, vacíos, duplicados, outliers,
cardinalidad, constantes, inconsistencias y un score global.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .schema import (
    infer_roles, is_numeric, is_texty, is_datetime, try_parse_numeric,
    try_parse_datetime,
)

BLANK_TOKENS = {"", " ", "  ", "-", "--", "n/a", "na", "null", "none", "nan",
                "sin dato", "desconocido", "?", "."}


# --------------------------------------------------------------------------- #
# Perfil por columna
# --------------------------------------------------------------------------- #
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con el diagnóstico completo de cada columna."""
    roles = infer_roles(df)
    n = len(df)
    rows = []

    for c in df.columns:
        s = df[c]
        r = roles[c]
        n_null = int(s.isna().sum())
        n_blank = _count_blanks(s)
        n_unique = int(s.nunique(dropna=True))
        n_dup_vals = int(n - n_unique - n_null) if n else 0

        rec: dict = {
            "columna": c,
            "tipo_dato": str(s.dtype),
            "rol": r.role,
            "n_registros": n,
            "nulos": n_null,
            "%_nulos": round(n_null / n * 100, 2) if n else 0.0,
            "vacios_texto": n_blank,
            "%_vacios": round(n_blank / n * 100, 2) if n else 0.0,
            "completitud_%": round((n - n_null - n_blank) / n * 100, 2) if n else 0.0,
            "unicos": n_unique,
            "%_unicos": round(n_unique / n * 100, 2) if n else 0.0,
            "valores_repetidos": max(n_dup_vals, 0),
            "constante": bool(n_unique <= 1),
            "memoria_kb": round(s.memory_usage(deep=True) / 1024, 1),
        }

        if is_numeric(s) and s.notna().any():
            desc = s.describe()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out_iqr = int(((s < lo) | (s > hi)).sum())
            std = float(s.std(ddof=0)) if s.notna().sum() > 1 else 0.0
            z = ((s - s.mean()) / std).abs() if std > 0 else pd.Series(0, index=s.index)
            rec.update({
                "min": _r(desc.get("min")),
                "p25": _r(q1),
                "mediana": _r(s.median()),
                "media": _r(s.mean()),
                "p75": _r(q3),
                "max": _r(desc.get("max")),
                "desv_std": _r(std),
                "cv_%": _r(abs(std / s.mean() * 100)) if s.mean() not in (0, np.nan) else None,
                "asimetria": _r(s.skew()),
                "curtosis": _r(s.kurtosis()),
                "ceros": int((s == 0).sum()),
                "negativos": int((s < 0).sum()),
                "outliers_iqr": out_iqr,
                "%_outliers": round(out_iqr / n * 100, 2) if n else 0.0,
                "outliers_z3": int((z > 3).sum()),
            })
        else:
            vc = s.dropna().astype(str).value_counts()
            rec.update({
                "moda": str(vc.index[0])[:60] if len(vc) else None,
                "frec_moda": int(vc.iloc[0]) if len(vc) else 0,
                "%_moda": round(float(vc.iloc[0]) / n * 100, 2) if len(vc) and n else 0.0,
                "long_min": int(s.dropna().astype(str).str.len().min()) if s.notna().any() else 0,
                "long_max": int(s.dropna().astype(str).str.len().max()) if s.notna().any() else 0,
                "espacios_sobrantes": _count_whitespace_issues(s),
                "may_min_inconsistente": _count_case_issues(s),
            })

        rec["alerta"] = _row_alert(rec, r.role)
        rec["recomendacion"] = r.suggestion or _row_reco(rec, r.role)
        rows.append(rec)

    prof = pd.DataFrame(rows)
    front = ["columna", "tipo_dato", "rol", "completitud_%", "nulos", "%_nulos",
             "vacios_texto", "unicos", "%_unicos", "alerta", "recomendacion"]
    order = front + [c for c in prof.columns if c not in front]
    return prof[order]


def _r(v, nd: int = 4):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return round(float(v), nd)
    except Exception:
        return None


def _count_blanks(s: pd.Series) -> int:
    if not is_texty(s):
        return 0
    try:
        t = s.dropna().astype(str).str.strip().str.lower()
        return int(t.isin(BLANK_TOKENS).sum())
    except Exception:
        return 0


def _count_whitespace_issues(s: pd.Series) -> int:
    if not is_texty(s):
        return 0
    try:
        t = s.dropna().astype(str)
        return int((t != t.str.strip()).sum())
    except Exception:
        return 0


def _count_case_issues(s: pd.Series) -> int:
    """Nº de valores que colisionan al normalizar mayúsculas/espacios."""
    if not is_texty(s):
        return 0
    try:
        t = s.dropna().astype(str)
        return int(t.nunique() - t.str.strip().str.lower().nunique())
    except Exception:
        return 0


def _row_alert(rec: dict, role: str) -> str:
    alerts = []
    if rec.get("constante"):
        alerts.append("CONSTANTE")
    if rec.get("%_nulos", 0) >= 50:
        alerts.append("NULOS CRÍTICOS")
    elif rec.get("%_nulos", 0) >= 20:
        alerts.append("NULOS ALTOS")
    if rec.get("%_vacios", 0) >= 10:
        alerts.append("VACÍOS")
    if role == "id":
        alerts.append("IDENTIFICADOR")
    if rec.get("%_outliers", 0) >= 10:
        alerts.append("OUTLIERS")
    if rec.get("may_min_inconsistente", 0) > 0:
        alerts.append("MAYÚS/MINÚS")
    if rec.get("espacios_sobrantes", 0) > 0:
        alerts.append("ESPACIOS")
    if role == "categorica" and rec.get("unicos", 0) > 50:
        alerts.append("ALTA CARDINALIDAD")
    if rec.get("%_moda", 0) >= 95 and not rec.get("constante"):
        alerts.append("CASI CONSTANTE")
    return " | ".join(alerts) if alerts else "OK"


def _row_reco(rec: dict, role: str) -> str:
    if rec.get("constante"):
        return "Eliminar: no aporta variabilidad."
    p = rec.get("%_nulos", 0)
    if p >= 50:
        return "Eliminar columna o imputar con indicador de faltante."
    if p >= 5:
        return ("Imputar con mediana" if role == "numerica"
                else "Imputar con la moda o categoría 'DESCONOCIDO'.")
    if rec.get("%_outliers", 0) >= 10:
        return "Winsorizar (recorte p1–p99) o aplicar transformación log."
    if role == "categorica" and rec.get("unicos", 0) > 50:
        return "Agrupar categorías infrecuentes en 'OTROS'."
    if rec.get("may_min_inconsistente", 0) > 0 or rec.get("espacios_sobrantes", 0) > 0:
        return "Normalizar texto: quitar espacios y unificar mayúsculas."
    return "Sin acción requerida."


# --------------------------------------------------------------------------- #
# Diagnóstico a nivel dataset
# --------------------------------------------------------------------------- #
def dataset_diagnostics(df: pd.DataFrame) -> dict:
    n, m = df.shape
    cells = max(n * m, 1)
    nulls = int(df.isna().sum().sum())
    blanks = int(sum(_count_blanks(df[c]) for c in df.columns))
    dup_rows = int(df.duplicated().sum())
    dup_cols = duplicated_columns(df)
    const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    empty_rows = int(df.isna().all(axis=1).sum())
    roles = infer_roles(df)
    id_cols = [c for c, r in roles.items() if r.role == "id"]
    high_null = [c for c in df.columns if df[c].isna().mean() >= 0.5]
    mixed = mixed_type_columns(df)
    convertibles = {
        c: r.convertible_a for c, r in roles.items() if r.convertible_a
    }

    return {
        "filas": n,
        "columnas": m,
        "celdas": cells,
        "celdas_nulas": nulls,
        "%_celdas_nulas": round(nulls / cells * 100, 2),
        "celdas_vacias_texto": blanks,
        "filas_duplicadas": dup_rows,
        "%_filas_duplicadas": round(dup_rows / max(n, 1) * 100, 2),
        "filas_totalmente_vacias": empty_rows,
        "columnas_duplicadas": dup_cols,
        "columnas_constantes": const_cols,
        "columnas_identificador": id_cols,
        "columnas_muy_nulas": high_null,
        "columnas_tipo_mixto": mixed,
        "columnas_convertibles": convertibles,
        "memoria_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 2),
        "roles": {c: r.role for c, r in roles.items()},
    }


def duplicated_columns(df: pd.DataFrame) -> list[list[str]]:
    """Grupos de columnas con contenido idéntico."""
    groups: dict[str, list[str]] = {}
    for c in df.columns:
        try:
            key = pd.util.hash_pandas_object(df[c].astype(str), index=False).sum()
        except Exception:
            key = hash(tuple(df[c].astype(str).tolist()[:1000]))
        groups.setdefault(str(key), []).append(c)
    return [g for g in groups.values() if len(g) > 1]


def mixed_type_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de texto donde conviven números y palabras."""
    out = []
    for c in df.columns:
        s = df[c]
        if not is_texty(s):
            continue
        sample = s.dropna().astype(str).head(1000)
        if sample.empty:
            continue
        numeric_like = pd.to_numeric(
            sample.str.replace(",", ".", regex=False), errors="coerce"
        ).notna().mean()
        if 0.15 < numeric_like < 0.85:
            out.append(c)
    return out


def quality_score(df: pd.DataFrame, diag: dict | None = None) -> dict:
    """
    Score 0–100 ponderado por dimensiones clásicas de calidad de datos.
    """
    d = diag or dataset_diagnostics(df)
    n, m = max(d["filas"], 1), max(d["columnas"], 1)

    completitud = 100 - d["%_celdas_nulas"] - (d["celdas_vacias_texto"] / d["celdas"] * 100)
    completitud = max(0.0, min(100.0, completitud))

    unicidad = 100 - d["%_filas_duplicadas"] - (len(d["columnas_duplicadas"]) / m * 100)
    unicidad = max(0.0, min(100.0, unicidad))

    penal_const = len(d["columnas_constantes"]) / m * 100
    penal_mixed = len(d["columnas_tipo_mixto"]) / m * 100
    consistencia = max(0.0, 100 - penal_const - penal_mixed)

    # Validez: outliers promedio
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    out_pct = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        out_pct.append(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).mean() * 100)
    validez = max(0.0, 100 - (float(np.mean(out_pct)) * 3 if out_pct else 0.0))

    total = (completitud * 0.35 + unicidad * 0.25 +
             consistencia * 0.20 + validez * 0.20)

    return {
        "score_total": round(total, 1),
        "completitud": round(completitud, 1),
        "unicidad": round(unicidad, 1),
        "consistencia": round(consistencia, 1),
        "validez": round(validez, 1),
        "nivel": _level(total),
    }


def _level(score: float) -> str:
    if score >= 90:
        return "Excelente"
    if score >= 75:
        return "Bueno"
    if score >= 60:
        return "Aceptable"
    if score >= 40:
        return "Deficiente"
    return "Crítico"


# --------------------------------------------------------------------------- #
# Hallazgos accionables
# --------------------------------------------------------------------------- #
def quality_findings(df: pd.DataFrame, diag: dict | None = None) -> list[dict]:
    """Lista priorizada de problemas con su acción sugerida."""
    d = diag or dataset_diagnostics(df)
    f: list[dict] = []

    def add(sev, titulo, detalle, accion):
        f.append({"severidad": sev, "hallazgo": titulo,
                  "detalle": detalle, "accion_sugerida": accion})

    if d["filas_duplicadas"] > 0:
        add("Alta", "Filas duplicadas",
            f"{d['filas_duplicadas']:,} filas ({d['%_filas_duplicadas']}%) están repetidas.",
            "Eliminar duplicados exactos en la pestaña ETL.")
    if d["filas_totalmente_vacias"] > 0:
        add("Alta", "Filas vacías",
            f"{d['filas_totalmente_vacias']:,} filas sin ningún dato.",
            "Eliminar esas filas.")
    for grupo in d["columnas_duplicadas"]:
        add("Media", "Columnas idénticas",
            f"Las columnas {grupo} tienen exactamente el mismo contenido.",
            f"Conservar solo '{grupo[0]}' y eliminar el resto.")
    if d["columnas_constantes"]:
        add("Media", "Columnas constantes",
            f"{d['columnas_constantes']} tienen un único valor.",
            "Eliminarlas: no aportan información al modelo.")
    if d["columnas_muy_nulas"]:
        add("Alta", "Columnas con exceso de nulos",
            f"{d['columnas_muy_nulas']} superan el 50% de valores faltantes.",
            "Eliminar o imputar creando una bandera de 'dato faltante'.")
    if d["columnas_identificador"]:
        add("Media", "Identificadores detectados",
            f"{d['columnas_identificador']} parecen llaves únicas.",
            "Excluirlas del entrenamiento para evitar sobreajuste.")
    if d["columnas_tipo_mixto"]:
        add("Media", "Tipos mixtos",
            f"{d['columnas_tipo_mixto']} mezclan números y texto.",
            "Estandarizar el tipo antes de modelar.")
    if d["columnas_convertibles"]:
        add("Baja", "Conversión de tipo disponible",
            f"{d['columnas_convertibles']} pueden convertirse automáticamente.",
            "Aplicar conversión de tipos en ETL.")
    if d["%_celdas_nulas"] > 5:
        add("Alta", "Nivel general de nulos",
            f"{d['%_celdas_nulas']}% de todas las celdas están vacías.",
            "Definir estrategia de imputación por columna.")
    if not f:
        add("Info", "Sin problemas críticos",
            "No se detectaron problemas graves de calidad.",
            "Puedes avanzar directamente al modelado.")

    orden = {"Alta": 0, "Media": 1, "Baja": 2, "Info": 3}
    return sorted(f, key=lambda x: orden.get(x["severidad"], 9))


def missing_matrix(df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
    """Matriz booleana de faltantes (submuestreada) para el heatmap."""
    step = max(1, len(df) // max_rows)
    return df.iloc[::step].isna().astype(int)


def missing_correlation(df: pd.DataFrame) -> pd.DataFrame | None:
    """Correlación entre patrones de ausencia: revela faltantes sistemáticos."""
    miss = df.isna().astype(int)
    miss = miss.loc[:, miss.sum() > 0]
    if miss.shape[1] < 2:
        return None
    return miss.corr().round(3)
