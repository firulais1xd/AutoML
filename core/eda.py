"""
Análisis exploratorio: estadística descriptiva, distribuciones, correlaciones
(Pearson/Spearman/Cramér's V), relación con la variable objetivo y detección
de asociaciones relevantes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .schema import is_numeric, is_texty, is_datetime

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Descriptiva
# --------------------------------------------------------------------------- #
def describe_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "variable": c, "n": int(s.size), "nulos": int(df[c].isna().sum()),
            "media": s.mean(), "mediana": s.median(),
            "moda": s.mode().iloc[0] if not s.mode().empty else np.nan,
            "desv_std": s.std(), "cv_%": abs(s.std() / s.mean() * 100) if s.mean() else np.nan,
            "min": s.min(), "p5": s.quantile(0.05), "p25": q1, "p75": q3,
            "p95": s.quantile(0.95), "max": s.max(), "rango": s.max() - s.min(),
            "iqr": q3 - q1, "asimetria": s.skew(), "curtosis": s.kurtosis(),
            "ceros_%": float((s == 0).mean() * 100),
            "distribucion": _shape_label(s),
        })
    return pd.DataFrame(rows).round(4)


def _shape_label(s: pd.Series) -> str:
    sk = s.skew()
    if not np.isfinite(sk):
        return "indeterminada"
    if abs(sk) < 0.5:
        base = "aproximadamente simétrica"
    elif sk >= 0.5:
        base = "sesgada a la derecha (cola alta)"
    else:
        base = "sesgada a la izquierda"
    k = s.kurtosis()
    if np.isfinite(k) and k > 3:
        base += ", leptocúrtica (colas pesadas)"
    return base


def describe_categorical(df: pd.DataFrame, columns: list[str] | None = None,
                         top: int = 3) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if is_texty(df[c])]
    rows = []
    n = len(df)
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        vc = s.value_counts()
        prop = vc / vc.sum()
        entropia = float(-(prop * np.log2(prop.clip(lower=1e-12))).sum())
        max_ent = np.log2(len(vc)) if len(vc) > 1 else 1
        rows.append({
            "variable": c, "n": int(s.size), "nulos": int(df[c].isna().sum()),
            "categorias": int(vc.size),
            "moda": str(vc.index[0]), "frec_moda": int(vc.iloc[0]),
            "%_moda": round(float(vc.iloc[0] / n * 100), 2),
            "top_categorias": " | ".join(
                f"{i} ({v}, {v/n*100:.1f}%)" for i, v in vc.head(top).items()),
            "entropia": round(entropia, 3),
            "balance": round(float(entropia / max_ent), 3) if max_ent else 1.0,
            "cardinalidad_%": round(float(vc.size / n * 100), 2),
        })
    return pd.DataFrame(rows)


def value_counts_table(df: pd.DataFrame, column: str, top: int = 20) -> pd.DataFrame:
    vc = df[column].astype(str).value_counts(dropna=False).head(top)
    out = pd.DataFrame({"categoria": vc.index.astype(str), "frecuencia": vc.to_numpy()})
    out["%"] = (out["frecuencia"] / len(df) * 100).round(2)
    out["%_acumulado"] = out["%"].cumsum().round(2)
    return out


# --------------------------------------------------------------------------- #
# Correlaciones
# --------------------------------------------------------------------------- #
def correlation_matrix(df: pd.DataFrame, method: str = "pearson",
                       columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr(method=method).round(3)


def top_correlations(corr: pd.DataFrame, threshold: float = 0.5,
                     top: int = 25) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame()
    m = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    if m.empty:
        return pd.DataFrame()
    df = m.reset_index()
    df.columns = ["variable_1", "variable_2", "correlacion"]
    df["abs"] = df["correlacion"].abs()
    df = df[df["abs"] >= threshold].sort_values("abs", ascending=False).head(top)
    df["fuerza"] = df["abs"].map(_corr_strength)
    df["sentido"] = np.where(df["correlacion"] > 0, "directa", "inversa")
    return df.drop(columns="abs").reset_index(drop=True)


def _corr_strength(a: float) -> str:
    if a >= 0.9:
        return "casi perfecta (posible redundancia)"
    if a >= 0.7:
        return "fuerte"
    if a >= 0.5:
        return "moderada"
    if a >= 0.3:
        return "débil"
    return "muy débil"


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Asociación entre dos variables categóricas (0 a 1)."""
    tab = pd.crosstab(x.astype(str), y.astype(str))
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def categorical_association_matrix(df: pd.DataFrame, columns: list[str] | None = None,
                                   max_cols: int = 15) -> pd.DataFrame:
    cols = (columns or [c for c in df.columns if is_texty(df[c])])[:max_cols]
    if len(cols) < 2:
        return pd.DataFrame()
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = cramers_v(df[a], df[b])
            m.loc[a, b] = m.loc[b, a] = round(v, 3) if v == v else np.nan
    return m


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta²: asociación entre una categórica y una numérica."""
    d = pd.DataFrame({"c": categories.astype(str), "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if d.empty or d["c"].nunique() < 2:
        return np.nan
    total_mean = d["v"].mean()
    ss_between = sum(len(g) * (g["v"].mean() - total_mean) ** 2 for _, g in d.groupby("c"))
    ss_total = ((d["v"] - total_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan


# --------------------------------------------------------------------------- #
# Relación con el objetivo
# --------------------------------------------------------------------------- #
def target_relationship(df: pd.DataFrame, target: str,
                        features: list[str] | None = None) -> pd.DataFrame:
    """
    Fuerza de relación de cada variable con el objetivo, con la prueba
    estadística adecuada según los tipos.
    """
    feats = [c for c in (features or df.columns) if c != target and c in df.columns]
    y = df[target]
    y_num = is_numeric(y)
    rows = []

    for c in feats:
        s = df[c]
        rec = {"variable": c, "tipo_variable": "numérica" if is_numeric(s) else "categórica"}
        try:
            if y_num and is_numeric(s):
                d = pd.DataFrame({"x": s, "y": y}).dropna()
                if len(d) < 3:
                    continue
                r, p = stats.pearsonr(d["x"], d["y"])
                rho, _ = stats.spearmanr(d["x"], d["y"])
                rec.update({"metrica": "Pearson r", "valor": round(float(r), 4),
                            "spearman": round(float(rho), 4),
                            "p_valor": float(p), "fuerza": _corr_strength(abs(r))})
            elif y_num and not is_numeric(s):
                eta = correlation_ratio(s, y)
                groups = [g.dropna().to_numpy() for _, g in
                          pd.DataFrame({"c": s.astype(str), "y": y}).dropna().groupby("c")["y"]]
                p = stats.f_oneway(*groups)[1] if len(groups) > 1 else np.nan
                rec.update({"metrica": "Eta (η)", "valor": round(float(eta), 4) if eta == eta else None,
                            "p_valor": float(p) if p == p else None,
                            "fuerza": _corr_strength(abs(eta) if eta == eta else 0)})
            elif not y_num and is_numeric(s):
                eta = correlation_ratio(y, s)
                groups = [g.dropna().to_numpy() for _, g in
                          pd.DataFrame({"c": y.astype(str), "x": s}).dropna().groupby("c")["x"]]
                p = stats.f_oneway(*groups)[1] if len(groups) > 1 else np.nan
                rec.update({"metrica": "Eta (η)", "valor": round(float(eta), 4) if eta == eta else None,
                            "p_valor": float(p) if p == p else None,
                            "fuerza": _corr_strength(abs(eta) if eta == eta else 0)})
            else:
                v = cramers_v(s, y)
                tab = pd.crosstab(s.astype(str), y.astype(str))
                p = stats.chi2_contingency(tab)[1] if tab.shape[0] > 1 and tab.shape[1] > 1 else np.nan
                rec.update({"metrica": "Cramér's V", "valor": round(float(v), 4) if v == v else None,
                            "p_valor": float(p) if p == p else None,
                            "fuerza": _corr_strength(abs(v) if v == v else 0)})
        except Exception:
            continue

        if rec.get("valor") is not None:
            pv = rec.get("p_valor")
            rec["significativo_95%"] = bool(pv is not None and pv == pv and pv < 0.05)
            rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs"] = out["valor"].abs()
    out = out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
    if "p_valor" in out.columns:
        out["p_valor"] = out["p_valor"].map(
            lambda v: f"{v:.2e}" if v is not None and v == v and v < 0.001 else
            (round(v, 4) if v is not None and v == v else None))
    return out


def group_summary(df: pd.DataFrame, by: str, value: str) -> pd.DataFrame:
    d = df[[by, value]].dropna()
    if d.empty:
        return pd.DataFrame()
    if is_numeric(d[value]):
        g = d.groupby(by)[value].agg(
            n="count", promedio="mean", mediana="median", desv="std",
            minimo="min", maximo="max")
        g["%_del_total"] = (g["n"] / g["n"].sum() * 100)
        return g.round(3).reset_index()
    tab = pd.crosstab(d[by], d[value], normalize="index") * 100
    return tab.round(2).reset_index()


def class_balance(y: pd.Series) -> pd.DataFrame:
    vc = y.astype(str).value_counts()
    out = pd.DataFrame({"clase": vc.index, "n": vc.to_numpy()})
    out["%"] = (out["n"] / out["n"].sum() * 100).round(2)
    out["ratio_vs_mayoritaria"] = (out["n"] / out["n"].max()).round(3)
    return out


def outlier_table(df: pd.DataFrame, columns: list[str] | None = None,
                  factor: float = 1.5) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        mask = (s < lo) | (s > hi)
        rows.append({
            "variable": c, "limite_inferior": round(float(lo), 4),
            "limite_superior": round(float(hi), 4),
            "n_outliers": int(mask.sum()),
            "%_outliers": round(float(mask.mean() * 100), 2),
            "min_outlier": round(float(s[mask].min()), 4) if mask.any() else None,
            "max_outlier": round(float(s[mask].max()), 4) if mask.any() else None,
        })
    return pd.DataFrame(rows).sort_values("%_outliers", ascending=False).reset_index(drop=True)


def normality_tests(df: pd.DataFrame, columns: list[str] | None = None,
                    max_n: int = 5000) -> pd.DataFrame:
    cols = columns or [c for c in df.columns if is_numeric(df[c])]
    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) < 20:
            continue
        sample = s.sample(min(len(s), max_n), random_state=42)
        try:
            w, p = stats.shapiro(sample) if len(sample) <= 5000 else (np.nan, np.nan)
        except Exception:
            w, p = np.nan, np.nan
        rows.append({
            "variable": c, "shapiro_W": round(float(w), 4) if w == w else None,
            "p_valor": f"{p:.2e}" if p == p and p < 0.001 else (round(float(p), 4) if p == p else None),
            "es_normal_95%": bool(p == p and p > 0.05),
            "asimetria": round(float(s.skew()), 3),
            "sugerencia": ("Distribución normal" if (p == p and p > 0.05)
                           else "No normal: considerar log/Yeo-Johnson o modelos no paramétricos"),
        })
    return pd.DataFrame(rows)


def timeseries_summary(df: pd.DataFrame, date_col: str, value_col: str,
                       freq: str = "ME") -> pd.DataFrame | None:
    """Agregación temporal con variación periodo a periodo."""
    if date_col not in df.columns or value_col not in df.columns:
        return None
    d = df[[date_col, value_col]].dropna().copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    if d.empty:
        return None
    g = d.set_index(date_col)[value_col].resample(freq).agg(["sum", "mean", "count"])
    g = g.rename(columns={"sum": "total", "mean": "promedio", "count": "n"})
    g["variacion_%"] = (g["total"].pct_change() * 100).round(2)
    return g.reset_index().round(3)
