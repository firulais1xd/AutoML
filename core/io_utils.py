"""
Carga y detección automática de archivos (CSV / Excel).
Maneja separadores, codificaciones y hojas múltiples.
"""
from __future__ import annotations

import io
import os
import re
from typing import Any

import numpy as np
import pandas as pd

ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
SEPARATORS = [",", ";", "\t", "|"]

NA_STRINGS = [
    "", " ", "  ", "NA", "N/A", "n/a", "na", "NaN", "nan", "NULL", "null", "None",
    "none", "-", "--", "?", "#N/A", "#NULL!", "#DIV/0!", "sin dato", "SIN DATO",
    "Sin Dato", "desconocido", "DESCONOCIDO", "S/D", "s/d", ".",
]


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #
def list_excel_sheets(source: Any) -> list[str]:
    """Devuelve los nombres de hoja de un Excel."""
    try:
        xls = pd.ExcelFile(source)
        return list(xls.sheet_names)
    except Exception:
        return []


def _sniff_separator(sample: str) -> str:
    """Detecta el separador contando ocurrencias fuera de comillas."""
    best, best_score = ",", -1.0
    lines = [ln for ln in sample.splitlines() if ln.strip()][:25]
    if not lines:
        return ","
    for sep in SEPARATORS:
        counts = [ln.count(sep) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        mean = float(np.mean(counts))
        std = float(np.std(counts))
        # Preferimos el separador con muchas ocurrencias y consistente entre filas
        score = mean - std * 2
        if score > best_score:
            best_score, best = score, sep
    return best


def _detect_header_row(text: str, sep: str, max_scan: int = 12) -> int:
    """
    Localiza la fila del encabezado cuando el archivo trae líneas de título.

    Devuelve el índice de la primera línea cuyo número de separadores coincide
    con el del grueso de la tabla; 0 si el archivo ya empieza bien.
    """
    lines = [ln for ln in text.splitlines()[:max_scan + 40] if ln.strip()]
    if len(lines) < 3:
        return 0
    counts = [ln.count(sep) for ln in lines]
    cuerpo = counts[max_scan:] or counts[2:]
    if not cuerpo:
        return 0
    modal = max(set(cuerpo), key=cuerpo.count)
    if modal == 0:
        return 0
    for i, c in enumerate(counts[:max_scan]):
        if c == modal:
            return i
    return 0


def read_any(
    source: Any,
    filename: str | None = None,
    sheet_name: str | int | None = 0,
    sep: str | None = None,
    encoding: str | None = None,
    decimal: str | None = None,
    header_row: int = 0,
    na_values: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Lee un CSV/TSV/TXT/XLSX/XLS/Parquet/JSON desde ruta o buffer.

    Devuelve (DataFrame, metadatos_de_lectura).
    """
    name = (filename or getattr(source, "name", "") or str(source)).lower()
    ext = os.path.splitext(name)[1]
    na_vals = na_values if na_values is not None else NA_STRINGS
    meta: dict[str, Any] = {"archivo": os.path.basename(name), "extension": ext}

    def _rewind():
        try:
            source.seek(0)
        except Exception:
            pass

    # ---------------- Excel ----------------
    if ext in (".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"):
        _rewind()
        engine = None
        if ext == ".xlsb":
            engine = "pyxlsb"
        elif ext == ".ods":
            engine = "odf"
        df = pd.read_excel(
            source,
            sheet_name=sheet_name if sheet_name is not None else 0,
            header=header_row,
            na_values=na_vals,
            keep_default_na=True,
            engine=engine,
        )
        if isinstance(df, dict):  # varias hojas
            first = list(df.keys())[0]
            meta["hojas"] = list(df.keys())
            df = df[first]
            meta["hoja_usada"] = first
        else:
            meta["hoja_usada"] = sheet_name
        meta["motor"] = "excel"
        return _postprocess(df), meta

    # ---------------- Parquet / JSON ----------------
    if ext == ".parquet":
        _rewind()
        return _postprocess(pd.read_parquet(source)), {**meta, "motor": "parquet"}
    if ext in (".json", ".jsonl"):
        _rewind()
        df = pd.read_json(source, lines=(ext == ".jsonl"))
        return _postprocess(df), {**meta, "motor": "json"}

    # ---------------- Texto plano (csv/tsv/txt) ----------------
    raw: bytes
    if hasattr(source, "read"):
        _rewind()
        raw = source.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8", errors="replace")
    else:
        with open(source, "rb") as fh:
            raw = fh.read()

    used_encoding = encoding
    text = None
    if used_encoding:
        try:
            text = raw.decode(used_encoding)
        except Exception:
            text = None
    if text is None:
        for enc in ENCODINGS:
            try:
                text = raw.decode(enc)
                used_encoding = enc
                break
            except Exception:
                continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
        used_encoding = "utf-8 (con reemplazos)"

    used_sep = sep or _sniff_separator(text[:20000])

    # Encabezado desplazado: algunos exportes traen líneas de título antes de
    # la tabla. Si el usuario no indicó una fila, la buscamos.
    detected_header = header_row
    if header_row == 0:
        detected_header = _detect_header_row(text, used_sep)
        if detected_header:
            meta["fila_encabezado_detectada"] = detected_header
    header_row = detected_header

    # Decimal: si con punto quedan casi todas las columnas como texto, probamos coma
    candidates = [decimal] if decimal else [".", ","]
    best_df, best_meta, best_score = None, None, -1.0
    for dec in candidates:
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=used_sep,
                header=header_row,
                na_values=na_vals,
                keep_default_na=True,
                decimal=dec,
                thousands="," if dec == "." else ".",
                engine="python",
                on_bad_lines="warn",
            )
        except Exception:
            continue
        score = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
        if score > best_score:
            best_score = score
            best_df = df
            best_meta = {"separador": used_sep, "encoding": used_encoding, "decimal": dec}

    if best_df is None:
        raise ValueError(
            "No fue posible leer el archivo. Revisa el separador, la codificación "
            "o conviértelo a .xlsx."
        )

    meta.update(best_meta or {})
    meta["motor"] = "csv"
    return _postprocess(best_df), meta


# --------------------------------------------------------------------------- #
# Post-proceso común
# --------------------------------------------------------------------------- #
def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima e inocua: nombres de columna y columnas fantasma."""
    df = df.copy()
    df.columns = [clean_column_name(c, i) for i, c in enumerate(df.columns)]
    # Columnas 'Unnamed' completamente vacías que Excel suele agregar
    ghost = [
        c for c in df.columns
        if str(c).lower().startswith("unnamed") and df[c].isna().all()
    ]
    if ghost:
        df = df.drop(columns=ghost)
    # Filas totalmente vacías
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)
    return df


def clean_column_name(name: Any, idx: int = 0) -> str:
    s = str(name).strip()
    if s == "" or s.lower() == "nan":
        s = f"columna_{idx + 1}"
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case sin acentos ni caracteres especiales."""
    import unicodedata

    def norm(s: str) -> str:
        s = str(s).strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = re.sub(r"[^\w\s]", "_", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return s or "col"

    out = df.copy()
    new, seen = [], {}
    for c in out.columns:
        n = norm(c)
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        new.append(n)
    out.columns = new
    return out


def downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria sin perder información."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_integer_dtype(s):
            out[c] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[c] = pd.to_numeric(s, downcast="float")
    return out


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1024**2
