"""
Inferencia de tipos semánticos de columnas.
Compatible con pandas 2.x y 3.x (dtype string dedicado).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_PATTERNS = re.compile(
    r"^(id|.*_id|id_.*|codigo|código|cod|cod_.*|key|uuid|guid|folio|nit|cedula|"
    r"cédula|documento|dni|rut|placa|serial|ticket|order_?id|customer_?id)$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Predicados robustos
# --------------------------------------------------------------------------- #
def is_bool(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    vals = set(str(v).strip().lower() for v in s.dropna().unique()[:50])
    return len(vals) > 0 and vals <= {"true", "false", "si", "sí", "no", "yes",
                                      "0", "1", "0.0", "1.0", "y", "n", "t", "f",
                                      "verdadero", "falso"}


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def is_categorical(s: pd.Series) -> bool:
    return isinstance(s.dtype, pd.CategoricalDtype)


def is_texty(s: pd.Series) -> bool:
    """object / string / category — todo lo que no es numérico ni fecha."""
    if is_categorical(s):
        return True
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
    ) and not pd.api.types.is_numeric_dtype(s)


def try_parse_datetime(s: pd.Series, min_ratio: float = 0.80) -> pd.Series | None:
    """Intenta convertir a fecha; devuelve None si no supera el umbral de éxito."""
    if is_datetime(s):
        return s
    if is_numeric(s):
        return None
    sample = s.dropna()
    if sample.empty:
        return None
    for dayfirst in (True, False):
        try:
            conv = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst,
                                  format="mixed")
        except Exception:
            try:
                conv = pd.to_datetime(sample, errors="coerce", dayfirst=dayfirst)
            except Exception:
                continue
        ok = conv.notna().mean()
        if ok >= min_ratio:
            try:
                return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst,
                                      format="mixed")
            except Exception:
                return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
    return None


ALPHA_RE = re.compile(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]")


def try_parse_numeric(s: pd.Series, min_ratio: float = 0.85) -> pd.Series | None:
    """
    Convierte texto tipo '1.234,56', '$ 1,200', '45%' a número.

    Rechaza códigos alfanuméricos ('CLI-000123', 'REF-99'): si buena parte de
    los valores contiene letras, no es una columna numérica disfrazada.
    """
    if is_numeric(s):
        return s
    sample = s.dropna().astype(str)
    if sample.empty:
        return None
    con_letras = sample.head(500).str.contains(ALPHA_RE, regex=True).mean()
    if con_letras > 0.15:
        return None
    cleaned_sample = _clean_numeric_strings(sample)
    conv = pd.to_numeric(cleaned_sample, errors="coerce")
    if conv.notna().mean() >= min_ratio:
        full = _clean_numeric_strings(s.astype(str).where(s.notna()))
        return pd.to_numeric(full, errors="coerce")
    return None


def _clean_numeric_strings(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"[^\d,.\-+eE%]", "", regex=True)
    pct = out.str.endswith("%")
    out = out.str.rstrip("%")
    # 1.234,56 -> 1234.56  |  1,234.56 -> 1234.56
    both = out.str.contains(r",") & out.str.contains(r"\.")
    comma_last = both & (out.str.rfind(",") > out.str.rfind("."))
    out = out.mask(comma_last, out.str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False))
    out = out.mask(both & ~comma_last, out.str.replace(",", "", regex=False))
    only_comma = ~both & out.str.contains(",")
    out = out.mask(only_comma, out.str.replace(",", ".", regex=False))
    num = pd.to_numeric(out, errors="coerce")
    num = num.mask(pct.fillna(False), num / 100.0)
    return num


# --------------------------------------------------------------------------- #
# Perfil semántico
# --------------------------------------------------------------------------- #
@dataclass
class ColumnRole:
    name: str
    dtype: str
    role: str                    # numerica | categorica | booleana | fecha | texto | id | constante
    n_unique: int = 0
    pct_missing: float = 0.0
    suggestion: str = ""
    convertible_a: str = ""
    ejemplos: list = field(default_factory=list)


def infer_roles(df: pd.DataFrame, cat_max_unique: int = 50) -> dict[str, ColumnRole]:
    """Asigna un rol semántico a cada columna."""
    n = max(len(df), 1)
    roles: dict[str, ColumnRole] = {}

    for c in df.columns:
        s = df[c]
        nu = int(s.nunique(dropna=True))
        miss = float(s.isna().mean() * 100)
        ejemplos = [str(v)[:40] for v in s.dropna().unique()[:4]]
        conv = ""

        if nu <= 1:
            role = "constante"
        elif is_datetime(s):
            role = "fecha"
        elif is_bool(s) and nu <= 2:
            role = "booleana"
        elif is_numeric(s):
            # numérica con pocos valores distintos y enteros -> puede ser categórica
            if nu <= min(cat_max_unique, 15) and pd.api.types.is_integer_dtype(s):
                role = "categorica"
                conv = "numerica"
            else:
                role = "numerica"
        else:
            # Los identificadores se detectan ANTES de intentar conversiones,
            # para no confundir un código como 'CLI-000123' con un número.
            if ID_PATTERNS.match(str(c)) and nu / n > 0.85:
                role = "id"
            elif nu / n > 0.92 and nu > 40:
                role = "id"
            elif try_parse_datetime(s) is not None:
                role, conv = "texto", "fecha"
            elif try_parse_numeric(s) is not None:
                role, conv = "texto", "numerica"
            elif nu <= cat_max_unique:
                role = "categorica"
            else:
                role = "texto"

        sug = _suggest(role, nu, miss, n, conv)
        roles[c] = ColumnRole(
            name=str(c), dtype=str(s.dtype), role=role, n_unique=nu,
            pct_missing=round(miss, 2), suggestion=sug, convertible_a=conv,
            ejemplos=ejemplos,
        )
    return roles


def _suggest(role: str, nu: int, miss: float, n: int, conv: str) -> str:
    if role == "constante":
        return "Columna constante: no aporta información, se recomienda eliminarla."
    if role == "id":
        return "Parece un identificador: excluir del modelo (genera sobreajuste)."
    if miss > 60:
        return f"{miss:.0f}% de nulos: evaluar eliminar la columna o imputar con cuidado."
    if conv == "fecha":
        return "Texto con formato de fecha: convertir a fecha y derivar año/mes/día."
    if conv == "numerica":
        return "Texto que representa números: convertir a numérica."
    if role == "categorica" and nu > 30:
        return f"Alta cardinalidad ({nu} categorías): agrupar las poco frecuentes en 'OTROS'."
    if role == "texto":
        return "Texto libre: requiere vectorización o exclusión del modelo."
    return ""


def split_by_role(roles: dict[str, ColumnRole]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c, r in roles.items():
        out.setdefault(r.role, []).append(c)
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_numeric(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 10_000) -> list[str]:
    return [c for c in df.columns if is_texty(df[c]) and df[c].nunique() <= max_unique]


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_datetime(df[c])]
