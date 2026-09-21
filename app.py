"""
AutoML Studio — plataforma de análisis y modelado a partir de un archivo.

Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import time
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from core import io_utils, quality, etl, eda, models, importance, clustering, insights, viz
from core.schema import infer_roles, is_numeric, is_texty, is_datetime

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoML Studio", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Estilos
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .kpi {background:#fcfcfb; border:1px solid #e8e7e3; border-radius:10px;
        padding:14px 16px; height:100%;}
  .kpi .lbl {font-size:12px; color:#52514e; text-transform:uppercase;
             letter-spacing:.04em; margin-bottom:4px;}
  .kpi .val {font-size:26px; font-weight:650; color:#0b0b0b; line-height:1.1;}
  .kpi .sub {font-size:12px; color:#88867e; margin-top:3px;}
  .pill {display:inline-block; padding:2px 9px; border-radius:20px; font-size:11.5px;
         font-weight:600; margin-right:5px;}
  .p-alta{background:#fbecec; color:#a32020;}
  .p-media{background:#fdf4e3; color:#8a5b00;}
  .p-baja{background:#eef4fc; color:#1c5cab;}
  .p-ok{background:#eaf5ef; color:#00631f;}
  .card {background:#fcfcfb; border:1px solid #e8e7e3; border-left:3px solid #2a78d6;
         border-radius:8px; padding:13px 16px; margin-bottom:10px;}
  .card h4 {margin:0 0 5px 0; font-size:14.5px; color:#0b0b0b;}
  .card p {margin:0; font-size:13.5px; color:#52514e; line-height:1.55;}
  div[data-testid="stMetricValue"] {font-size:24px;}
  .stTabs [data-baseweb="tab-list"] {gap:2px;}
  .stTabs [data-baseweb="tab"] {padding:9px 15px; font-size:14px;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Utilidades de interfaz
# --------------------------------------------------------------------------- #
def chart(fig, key: str | None = None):
    try:
        st.plotly_chart(fig, width="stretch", key=key,
                        config={"displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def table(df: pd.DataFrame, height: int | None = None, hide_index: bool = True):
    kw = {"hide_index": hide_index}
    if height is not None:
        kw["height"] = int(height)
    try:
        st.dataframe(df, width="stretch", **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kw)


def kpi(col, label: str, value, sub: str = ""):
    col.markdown(
        f"<div class='kpi'><div class='lbl'>{label}</div>"
        f"<div class='val'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def download_df(df: pd.DataFrame, nombre: str, label: str, key: str):
    c1, c2 = st.columns(2)
    c1.download_button(f"⬇ {label} (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       f"{nombre}.csv", "text/csv", key=f"{key}_csv")
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="datos")
        c2.download_button(f"⬇ {label} (Excel)", buf.getvalue(), f"{nombre}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"{key}_xlsx")
    except Exception:
        pass


def md2html(texto: str) -> str:
    """Convierte el markdown básico de las narrativas a HTML para las tarjetas."""
    import re as _re
    t = str(texto)
    t = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = _re.sub(r"(?<!\w)`(.+?)`(?!\w)", r"<code>\1</code>", t)
    return t


def sev_pill(sev: str) -> str:
    m = {"Alta": "p-alta", "Media": "p-media", "Baja": "p-baja", "Info": "p-ok"}
    return f"<span class='pill {m.get(sev, 'p-baja')}'>{sev}</span>"


# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
SS = st.session_state
for k, v in {
    "df_raw": None, "meta": {}, "recipe": [], "etl_log": [], "training": None,
    "cluster": None, "file_sig": None, "importancias": {}, "shap_cache": None,
}.items():
    SS.setdefault(k, v)


@st.cache_data(show_spinner=False)
def _load(data: bytes, filename: str, sheet, header_row: int, sep, decimal):
    return io_utils.read_any(io.BytesIO(data), filename=filename, sheet_name=sheet,
                             header_row=header_row, sep=sep or None,
                             decimal=decimal or None)


@st.cache_data(show_spinner=False)
def _profile(df: pd.DataFrame):
    return quality.column_profile(df)


@st.cache_data(show_spinner=False)
def _diag(df: pd.DataFrame):
    d = quality.dataset_diagnostics(df)
    return d, quality.quality_score(df, d), quality.quality_findings(df, d)


# Claves de widgets que dependen de los nombres de columna: al cambiar de
# archivo hay que borrarlas, o Streamlit conserva columnas que ya no existen.
COLUMN_BOUND_KEYS = [
    "m_target", "m_feats", "m_models", "m_task", "lb_m", "det_m",
    "eda_target", "dist_col", "dist_g", "corr_v", "corr_m",
    "bi_x", "bi_y", "bi_c", "ts_d", "ts_v", "out_col",
    "imp_m", "pdp_v", "cl_feats", "cl_algos", "cl_view", "cl_prof_algo",
    "cl_q", "cl_tgt", "cl_crossv", "cl_rows", "wf_m", "wf_sv",
    "op_m", "op_lev", "op_cls", "exp_m", "ks", "eps_sug", "kd_curve",
]
COLUMN_BOUND_PREFIXES = ("wf_", "ob_", "lv_", "lvn_")


def wkey(nombre: str) -> str:
    """
    Clave de widget versionada por conjunto de datos.

    Al cambiar de archivo cambia la versión, así que Streamlit crea widgets
    nuevos con sus valores por defecto en vez de conservar columnas que ya
    no existen.
    """
    return f"{nombre}__v{SS.get('data_version', 0)}"


def reset_column_state():
    """Invalida todos los widgets que dependen de los nombres de columna."""
    SS["data_version"] = SS.get("data_version", 0) + 1
    SS.pop("clx", None)
    for k in list(SS.keys()):
        if not isinstance(k, str):
            continue
        base = k.split("__v")[0]
        if base in COLUMN_BOUND_KEYS or base.startswith(COLUMN_BOUND_PREFIXES):
            SS.pop(k, None)


def working_df() -> pd.DataFrame | None:
    """DataFrame con la receta de ETL aplicada."""
    if SS.df_raw is None:
        return None
    if not SS.recipe:
        return SS.df_raw
    out, log = etl.apply_recipe(SS.df_raw, SS.recipe)
    SS.etl_log = log
    return out


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📊 AutoML Studio")
    st.caption("ETL · Calidad · Modelos · Clusters · Prescriptivo")
    st.divider()

    up = st.file_uploader("Sube tu archivo", type=["csv", "xlsx", "xls", "xlsm",
                                                    "txt", "tsv", "parquet", "json"],
                          help="CSV, Excel, Parquet o JSON")

    usar_demo = st.checkbox("Usar datos de ejemplo", value=False)
    demo_cual = None
    if usar_demo:
        demo_cual = st.radio("Conjunto", ["Churn de clientes (clasificación)",
                                          "Precios de inmuebles (regresión)"],
                             label_visibility="collapsed")

    with st.expander("⚙️ Opciones de lectura"):
        hoja = st.text_input("Hoja de Excel (nombre o índice)", value="0")
        fila_encabezado = st.number_input("Fila del encabezado (0 = primera)", 0, 50, 0)
        sep_manual = st.text_input("Separador (vacío = automático)", value="")
        dec_manual = st.selectbox("Separador decimal", ["automático", ".", ","], index=0)

    # --- Carga ---
    if up is not None:
        raw = up.getvalue()
        sig = (up.name, len(raw), hoja, fila_encabezado, sep_manual, dec_manual)
        if SS.file_sig != sig:
            try:
                sh = int(hoja) if str(hoja).strip().lstrip("-").isdigit() else hoja
                with st.spinner("Leyendo archivo…"):
                    df, meta = _load(raw, up.name, sh, int(fila_encabezado),
                                     sep_manual, None if dec_manual == "automático" else dec_manual)
                reset_column_state()
                SS.update(df_raw=df, meta=meta, file_sig=sig, recipe=[], training=None,
                          cluster=None, importancias={}, shap_cache=None)
                st.success(f"✔ {df.shape[0]:,} filas × {df.shape[1]} columnas")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
    elif usar_demo and demo_cual:
        sig = ("demo", demo_cual)
        if SS.file_sig != sig:
            import pathlib
            base = pathlib.Path(__file__).parent / "sample_data"
            ruta = (base / "clientes_churn.csv" if "Churn" in demo_cual
                    else base / "precios_inmuebles.xlsx")
            if not ruta.exists():
                import subprocess, sys
                subprocess.run([sys.executable, str(base / "generar_datos.py")], check=False)
            try:
                df, meta = io_utils.read_any(str(ruta))
                reset_column_state()
                SS.update(df_raw=df, meta=meta, file_sig=sig, recipe=[], training=None,
                          cluster=None, importancias={}, shap_cache=None)
            except Exception as e:
                st.error(f"No se pudo cargar el ejemplo: {e}")

    if SS.df_raw is not None:
        st.divider()
        d = working_df()
        st.markdown("**Conjunto en uso**")
        st.caption(f"{SS.meta.get('archivo', 'datos')}")
        c1, c2 = st.columns(2)
        c1.metric("Filas", f"{len(d):,}")
        c2.metric("Columnas", f"{d.shape[1]}")
        if SS.recipe:
            st.caption(f"🧪 {len(SS.recipe)} transformaciones activas "
                       f"(original: {len(SS.df_raw):,} × {SS.df_raw.shape[1]})")
        if st.button("↺ Reiniciar todo", width="stretch"):
            reset_column_state()
            for k in ["df_raw", "meta", "recipe", "training", "cluster", "file_sig",
                      "importancias", "shap_cache"]:
                SS[k] = None if k in ("df_raw", "training", "cluster", "file_sig",
                                      "shap_cache") else ([] if k == "recipe" else {})
            st.rerun()


# --------------------------------------------------------------------------- #
# Portada
# --------------------------------------------------------------------------- #
if SS.df_raw is None:
    st.title("AutoML Studio")
    st.markdown("#### Del archivo crudo a la decisión, en una sola herramienta")
    st.write("")
    c = st.columns(3)
    bloques = [
        ("🔍 Calidad de datos",
         "Nulos, vacíos, duplicados, constantes, outliers, tipos mixtos, "
         "cardinalidad y un score 0–100 con hallazgos priorizados."),
        ("🧪 ETL reproducible",
         "18 transformaciones encadenables, limpieza automática sugerida y "
         "exportación de la receta a código Python."),
        ("📊 Análisis descriptivo",
         "Estadística, distribuciones, correlaciones Pearson/Spearman/Cramér's V "
         "y fuerza de relación con el objetivo."),
        ("🤖 Comparación de modelos",
         "XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees, SVM, MLP y más, "
         "con accuracy, F1, ROC-AUC, R², RMSE, MAE y validación cruzada."),
        ("⭐ Variables clave",
         "Importancia nativa, por permutación, SHAP, coeficientes con signo, "
         "dependencia parcial y ranking de consenso."),
        ("🧩 Segmentación",
         "KMeans, DBSCAN, HDBSCAN, OPTICS, jerárquico, Gaussian Mixture, Birch, "
         "Mean Shift y Spectral, con codo, silueta y perfilado."),
    ]
    for i, (t, d) in enumerate(bloques):
        with c[i % 3]:
            st.markdown(f"<div class='card'><h4>{t}</h4><p>{d}</p></div>",
                        unsafe_allow_html=True)
    st.write("")
    st.info("**Para empezar:** sube un archivo CSV o Excel en la barra lateral, "
            "o marca *Usar datos de ejemplo*.")
    st.stop()


# --------------------------------------------------------------------------- #
df = working_df()
diag, score, findings = _diag(df)
roles = infer_roles(df)
num_cols = [c for c in df.columns if is_numeric(df[c])]
cat_cols = [c for c in df.columns if is_texty(df[c])]
date_cols = [c for c in df.columns if is_datetime(df[c])]

tabs = st.tabs(["📁 Datos", "🔍 Calidad", "🧪 ETL", "📊 Análisis", "🤖 Modelos",
                "⭐ Variables clave", "🧩 Clusters", "💡 Insights", "📤 Exportar"])

# =========================================================================== #
# 1 · DATOS
# =========================================================================== #
with tabs[0]:
    st.subheader("Vista general del conjunto")
    c = st.columns(5)
    kpi(c[0], "Registros", f"{len(df):,}")
    kpi(c[1], "Variables", f"{df.shape[1]}")
    kpi(c[2], "Completitud", f"{100 - diag['%_celdas_nulas']:.1f}%",
        f"{diag['celdas_nulas']:,} celdas vacías")
    kpi(c[3], "Duplicados", f"{diag['filas_duplicadas']:,}",
        f"{diag['%_filas_duplicadas']}% de las filas")
    kpi(c[4], "Memoria", f"{diag['memoria_mb']} MB")

    st.write("")
    tope_prev = max(5, min(200, len(df)))
    n_prev = (st.slider("Filas a mostrar", 5, tope_prev, min(25, tope_prev), key="prev_n")
              if tope_prev > 5 else tope_prev)
    table(df.head(n_prev), height=min(560, 38 * n_prev + 40), hide_index=False)

    st.markdown("#### Diccionario de variables")
    dicc = pd.DataFrame([{
        "columna": c,
        "rol_detectado": r.role,
        "tipo_pandas": r.dtype,
        "valores_únicos": r.n_unique,
        "%_nulos": r.pct_missing,
        "ejemplos": ", ".join(r.ejemplos),
        "observación": r.suggestion or "—",
    } for c, r in roles.items()])
    table(dicc, height=min(520, 38 * len(dicc) + 40))
    download_df(dicc, "diccionario_variables", "Diccionario", "dicc")

    if SS.meta:
        with st.expander("Detalles de lectura del archivo"):
            st.json(SS.meta)


# =========================================================================== #
# 2 · CALIDAD
# =========================================================================== #
with tabs[1]:
    st.subheader("Diagnóstico de calidad de los datos")
    c1, c2 = st.columns([1, 2])
    with c1:
        chart(viz.gauge_score(score["score_total"]), key="g_score")
        st.markdown(f"**Nivel: {score['nivel']}**")
    with c2:
        chart(viz.bar_dimensions(score), key="g_dims")

    st.markdown("#### Hallazgos priorizados")
    for f in findings:
        st.markdown(
            f"<div class='card'>{sev_pill(f['severidad'])}<b>{f['hallazgo']}</b>"
            f"<p style='margin-top:6px'>{md2html(f['detalle'])}<br>"
            f"<b>Acción:</b> {md2html(f['accion_sugerida'])}</p></div>", unsafe_allow_html=True)

    st.divider()
    sub = st.tabs(["Perfil por columna", "Faltantes", "Duplicados", "Outliers",
                   "Alertas"])

    with sub[0]:
        prof = _profile(df)
        solo_probl = st.checkbox("Mostrar solo columnas con alertas", value=False)
        vista = prof[prof["alerta"] != "OK"] if solo_probl else prof
        table(vista, height=min(560, 38 * len(vista) + 45))
        download_df(prof, "perfil_calidad", "Perfil de calidad", "perf")

    with sub[1]:
        prof = _profile(df)
        c1, c2 = st.columns([1, 1])
        with c1:
            chart(viz.bar_missing(prof), key="g_miss")
        with c2:
            chart(viz.heatmap_missing(df), key="g_missmap")
        mc = quality.missing_correlation(df)
        if mc is not None and len(mc) > 1:
            st.markdown("**Correlación entre patrones de ausencia** — valores altos "
                        "indican que los datos faltan juntos (faltante sistemático).")
            chart(viz.heatmap_corr(mc, "Correlación de faltantes", 0, 1), key="g_misscorr")
            table(mc, hide_index=False)

    with sub[2]:
        n_dup = diag["filas_duplicadas"]
        if n_dup:
            st.warning(f"Hay **{n_dup:,} filas duplicadas** ({diag['%_filas_duplicadas']}%).")
            st.markdown("**Ejemplos de filas repetidas:**")
            table(df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(50),
                  hide_index=False)
        else:
            st.success("No hay filas duplicadas exactas.")
        subset = st.multiselect("Buscar duplicados solo en estas columnas (llave de negocio)",
                                list(df.columns))
        if subset:
            nd = int(df.duplicated(subset=subset).sum())
            (st.warning if nd else st.success)(
                f"{nd:,} registros repiten la combinación {subset}.")
            if nd:
                table(df[df.duplicated(subset=subset, keep=False)]
                      .sort_values(subset).head(50), hide_index=False)
        if diag["columnas_duplicadas"]:
            st.markdown("**Columnas con contenido idéntico:**")
            for g in diag["columnas_duplicadas"]:
                st.write("· " + " = ".join(g))

    with sub[3]:
        if num_cols:
            ot = eda.outlier_table(df)
            table(ot)
            col_o = st.selectbox("Inspeccionar variable", num_cols, key=wkey("out_col"))
            c1, c2 = st.columns(2)
            with c1:
                chart(viz.histogram(df, col_o), key="g_outh")
            with c2:
                chart(viz.box_by_group(df, col_o), key="g_outb")
        else:
            st.info("No hay variables numéricas para analizar outliers.")

    with sub[4]:
        chart(viz.bar_alerts(_profile(df)), key="g_alerts")
        st.markdown("**Resumen del diagnóstico**")
        resumen = {k: v for k, v in diag.items() if k != "roles"}
        st.json(resumen, expanded=False)


# =========================================================================== #
# 3 · ETL
# =========================================================================== #
with tabs[2]:
    st.subheader("Transformación de datos")
    st.caption("Las transformaciones se encadenan en orden y siempre parten del "
               "archivo original, así que puedes eliminar cualquier paso sin perder nada.")

    c1, c2, c3 = st.columns([1, 1, 1])
    if c1.button("✨ Sugerir limpieza automática", width="stretch", type="primary"):
        SS.recipe = etl.auto_recipe(SS.df_raw)
        SS.training = None
        st.rerun()
    if c2.button("🗑 Vaciar receta", width="stretch"):
        SS.recipe = []
        SS.training = None
        st.rerun()
    if c3.button("↩ Deshacer último paso", width="stretch") and SS.recipe:
        SS.recipe = SS.recipe[:-1]
        SS.training = None
        st.rerun()

    st.markdown("#### Receta actual")
    if SS.recipe:
        for i, paso in enumerate(SS.recipe):
            cc = st.columns([0.06, 0.74, 0.20])
            cc[0].markdown(f"**{i+1}**")
            cc[1].markdown(
                f"**{etl.OPERATION_LABELS.get(paso['op'], paso['op'])}** — "
                f"`{ {k: v for k, v in paso.items() if k != 'op'} }`")
            if cc[2].button("Eliminar", key=f"del_{i}", width="stretch"):
                SS.recipe = [p for j, p in enumerate(SS.recipe) if j != i]
                SS.training = None
                st.rerun()
        if SS.etl_log:
            with st.expander("Bitácora de ejecución"):
                table(pd.DataFrame(SS.etl_log))
    else:
        st.info("Sin transformaciones. Usa la limpieza automática o agrega pasos abajo.")

    st.divider()
    st.markdown("#### Agregar transformación")
    op_label = st.selectbox("Operación", list(etl.OPERATION_LABELS.values()))
    op = [k for k, v in etl.OPERATION_LABELS.items() if v == op_label][0]
    params: dict = {}

    if op == "eliminar_columnas":
        params["columns"] = st.multiselect("Columnas a eliminar", list(df.columns))
    elif op == "renombrar":
        c = st.selectbox("Columna", list(df.columns))
        nuevo = st.text_input("Nuevo nombre", value=str(c))
        params["mapping"] = {c: nuevo}
    elif op == "eliminar_duplicados":
        params["subset"] = st.multiselect("Considerar solo estas columnas (opcional)",
                                          list(df.columns))
        params["keep"] = st.selectbox("Conservar", ["first", "last", False],
                                      format_func=lambda v: {"first": "el primero",
                                                             "last": "el último",
                                                             False: "ninguno"}[v])
    elif op == "eliminar_filas_vacias":
        modo = st.radio("Criterio", ["Filas totalmente vacías",
                                     "Filas con más de X% de nulos"], horizontal=True)
        if modo.startswith("Filas totalmente"):
            params["how"] = "all"
        else:
            params["thresh_pct"] = st.slider("% máximo de nulos permitido", 10, 95, 50)
    elif op == "convertir_tipos":
        cols = st.multiselect("Columnas", list(df.columns))
        destino = st.selectbox("Convertir a", ["numerica", "fecha", "texto",
                                               "categorica", "booleana", "entero"])
        params["mapping"] = {c: destino for c in cols}
    elif op == "limpiar_texto":
        params["columns"] = st.multiselect("Columnas (vacío = todas las de texto)", cat_cols)
        params["strip"] = st.checkbox("Quitar espacios al inicio/final", True)
        params["collapse_spaces"] = st.checkbox("Colapsar espacios múltiples", True)
        params["case"] = st.selectbox("Mayúsculas/minúsculas",
                                      ["none", "lower", "upper", "title"])
        params["remove_accents"] = st.checkbox("Quitar acentos", False)
    elif op == "imputar":
        params["columns"] = st.multiselect("Columnas (vacío = todas)", list(df.columns))
        c1, c2 = st.columns(2)
        params["strategy_num"] = c1.selectbox(
            "Numéricas", ["mediana", "media", "cero", "knn", "interpolar", "ffill",
                          "constante", "ninguna"])
        params["strategy_cat"] = c2.selectbox(
            "Categóricas", ["moda", "constante", "ffill", "ninguna"])
        params["add_flag"] = st.checkbox("Crear bandera de dato faltante", False)
        if "constante" in (params["strategy_num"], params["strategy_cat"]):
            params["constant_value"] = st.text_input("Valor constante", "DESCONOCIDO")
    elif op == "eliminar_filas_nulas":
        params["columns"] = st.multiselect("Eliminar filas con nulos en", list(df.columns))
    elif op == "outliers":
        params["columns"] = st.multiselect("Columnas (vacío = todas las numéricas)", num_cols)
        c1, c2, c3 = st.columns(3)
        params["method"] = c1.selectbox("Método", ["iqr", "zscore", "percentil"])
        params["action"] = c2.selectbox("Acción", ["winsorizar", "nulo", "eliminar"])
        params["factor"] = c3.number_input("Factor", 0.5, 6.0, 1.5, 0.5)
    elif op == "filtrar":
        params["query"] = st.text_input(
            "Condición (sintaxis pandas)",
            placeholder="ej: edad > 30 and ciudad == 'Bogotá'")
        st.caption(f"Columnas disponibles: {', '.join(map(str, df.columns[:15]))}…")
    elif op == "nueva_columna":
        params["name"] = st.text_input("Nombre de la nueva columna", "nueva_variable")
        params["expression"] = st.text_input(
            "Expresión", placeholder="ej: ingresos / (gastos + 1)")
    elif op == "features_fecha":
        params["columns"] = st.multiselect("Columnas de fecha", date_cols or list(df.columns))
        params["parts"] = st.multiselect(
            "Componentes", ["anio", "mes", "dia", "dia_semana", "semana", "trimestre",
                            "hora", "es_fin_semana"],
            default=["anio", "mes", "dia_semana", "trimestre"])
        params["drop_original"] = st.checkbox("Eliminar la columna original", False)
    elif op == "agrupar_raras":
        params["columns"] = st.multiselect("Columnas categóricas", cat_cols)
        params["min_freq_pct"] = st.slider("Frecuencia mínima (%)", 0.1, 20.0, 1.0, 0.1)
        params["other_label"] = st.text_input("Etiqueta para el resto", "OTROS")
    elif op == "transformar_numerica":
        params["columns"] = st.multiselect("Columnas", num_cols)
        params["method"] = st.selectbox("Transformación",
                                        ["log1p", "sqrt", "boxcox_yeo", "zscore", "minmax"])
    elif op == "discretizar":
        params["column"] = st.selectbox("Columna numérica", num_cols or list(df.columns))
        params["bins"] = st.slider("Número de intervalos", 2, 20, 5)
        params["method"] = st.selectbox("Método", ["cuantiles", "iguales"])
    elif op == "muestrear":
        modo = st.radio("Modo", ["Nº de filas", "Fracción"], horizontal=True)
        if modo == "Nº de filas":
            params["n"] = st.number_input("Filas", min_value=1, max_value=max(len(df), 1),
                                          value=min(1000, max(len(df), 1)))
        else:
            params["frac"] = st.slider("Fracción", 0.01, 1.0, 0.5)

    if st.button("➕ Agregar a la receta", type="primary"):
        SS.recipe = SS.recipe + [{"op": op, **params}]
        SS.training = None
        st.rerun()

    if SS.recipe:
        st.divider()
        st.markdown("#### Antes y después")
        c1, c2 = st.columns(2)
        c1.markdown("**Original**")
        c1.metric("Dimensiones", f"{SS.df_raw.shape[0]:,} × {SS.df_raw.shape[1]}")
        c1.metric("Celdas nulas", f"{int(SS.df_raw.isna().sum().sum()):,}")
        c2.markdown("**Transformado**")
        c2.metric("Dimensiones", f"{df.shape[0]:,} × {df.shape[1]}",
                  delta=f"{df.shape[0]-SS.df_raw.shape[0]:+,} filas")
        c2.metric("Celdas nulas", f"{int(df.isna().sum().sum()):,}",
                  delta=f"{int(df.isna().sum().sum()-SS.df_raw.isna().sum().sum()):+,}",
                  delta_color="inverse")
        with st.expander("🐍 Código Python equivalente"):
            st.code(etl.recipe_to_code(SS.recipe), language="python")
        st.download_button("⬇ Descargar receta (JSON)",
                           json.dumps(SS.recipe, ensure_ascii=False, indent=2, default=str),
                           "receta_etl.json", "application/json")


# =========================================================================== #
# 4 · ANÁLISIS EXPLORATORIO
# =========================================================================== #
with tabs[3]:
    st.subheader("Análisis exploratorio")
    sub = st.tabs(["Resumen estadístico", "Distribuciones", "Correlaciones",
                   "Relación con el objetivo", "Bivariado", "Temporal"])

    with sub[0]:
        if num_cols:
            st.markdown("**Variables numéricas**")
            dn = eda.describe_numeric(df)
            table(dn, height=min(460, 38 * len(dn) + 45))
            download_df(dn, "descriptiva_numerica", "Descriptiva numérica", "dn")
        if cat_cols:
            st.markdown("**Variables categóricas**")
            dc = eda.describe_categorical(df)
            table(dc, height=min(420, 38 * len(dc) + 45))
        if num_cols:
            with st.expander("Pruebas de normalidad"):
                table(eda.normality_tests(df))

    with sub[1]:
        c1, c2 = st.columns([2, 1])
        col_d = c1.selectbox("Variable", list(df.columns), key=wkey("dist_col"))
        agrupar = c2.selectbox("Separar por (opcional)", ["(ninguna)"] +
                               [c for c in cat_cols if df[c].nunique() <= 8], key=wkey("dist_g"))
        g = None if agrupar == "(ninguna)" else agrupar
        if is_numeric(df[col_d]):
            c1, c2 = st.columns(2)
            with c1:
                chart(viz.histogram(df, col_d, g), key="g_h1")
            with c2:
                chart(viz.box_by_group(df, col_d, g), key="g_b1")
            table(eda.describe_numeric(df, [col_d]))
        else:
            vc = eda.value_counts_table(df, col_d, top=25)
            c1, c2 = st.columns([3, 2])
            with c1:
                chart(viz.bar_categories(vc, titulo=f"Frecuencias de {col_d}"), key="g_c1")
            with c2:
                chart(viz.line_cumulative(vc), key="g_c2")
            table(vc)

    with sub[2]:
        if len(num_cols) >= 2:
            metodo = st.radio("Método", ["pearson", "spearman", "kendall"],
                              horizontal=True, key=wkey("corr_m"))
            sel = st.multiselect("Variables", num_cols, default=num_cols[:15], key=wkey("corr_v"))
            if len(sel) >= 2:
                corr = eda.correlation_matrix(df, metodo, sel)
                chart(viz.heatmap_corr(corr, f"Correlación ({metodo})"), key="g_corr")
                umbral = st.slider("Umbral para listar pares", 0.0, 1.0, 0.4, 0.05)
                tc = eda.top_correlations(corr, umbral)
                if not tc.empty:
                    table(tc)
                else:
                    st.info("No hay pares por encima del umbral.")
                with st.expander("Matriz en tabla"):
                    table(corr, hide_index=False)
        else:
            st.info("Se requieren al menos dos variables numéricas.")

        if len(cat_cols) >= 2:
            with st.expander("Asociación entre categóricas (Cramér's V)"):
                cm = eda.categorical_association_matrix(df)
                if not cm.empty:
                    chart(viz.heatmap_corr(cm, "Cramér's V", 0, 1), key="g_cram")
                    table(cm, hide_index=False)

    with sub[3]:
        objetivo = st.selectbox("Variable objetivo", list(df.columns), key=wkey("eda_target"))
        rel = eda.target_relationship(df, objetivo)
        if rel is not None and not rel.empty:
            chart(viz.bar_target_relation(rel), key="g_rel")
            table(rel)
            download_df(rel, "relacion_objetivo", "Relación con el objetivo", "rel")
        if not is_numeric(df[objetivo]) and df[objetivo].nunique() <= 30:
            st.markdown("**Balance de clases**")
            cb = eda.class_balance(df[objetivo])
            c1, c2 = st.columns([2, 3])
            with c1:
                table(cb)
            with c2:
                chart(viz.bar_categories(
                    cb.rename(columns={"clase": "categoria", "n": "frecuencia"}),
                    titulo="Distribución de la variable objetivo"), key="g_cb")

    with sub[4]:
        c1, c2, c3 = st.columns(3)
        vx = c1.selectbox("Eje X", list(df.columns), key=wkey("bi_x"))
        vy = c2.selectbox("Eje Y", [c for c in df.columns if c != vx], key=wkey("bi_y"))
        vc_ = c3.selectbox("Color por", ["(ninguno)"] +
                           [c for c in cat_cols if df[c].nunique() <= 8], key=wkey("bi_c"))
        col_color = None if vc_ == "(ninguno)" else vc_
        if is_numeric(df[vx]) and is_numeric(df[vy]):
            chart(viz.scatter_xy(df, vx, vy, col_color), key="g_sc")
            d = df[[vx, vy]].dropna()
            if len(d) > 2:
                st.caption(f"Correlación de Pearson: **{d[vx].corr(d[vy]):.4f}** · "
                           f"Spearman: **{d[vx].corr(d[vy], method='spearman'):.4f}**")
        elif is_numeric(df[vy]):
            chart(viz.box_by_group(df, vy, vx), key="g_bx")
            table(eda.group_summary(df, vx, vy))
        elif is_numeric(df[vx]):
            chart(viz.box_by_group(df, vx, vy), key="g_bx2")
            table(eda.group_summary(df, vy, vx))
        else:
            ct = pd.crosstab(df[vx], df[vy])
            chart(viz.heatmap_corr(ct.astype(float), f"{vx} × {vy}",
                                   0, float(ct.to_numpy().max())), key="g_ct")
            table(ct, hide_index=False)
            st.caption(f"Cramér's V: **{eda.cramers_v(df[vx], df[vy]):.4f}**")

    with sub[5]:
        if date_cols and num_cols:
            c1, c2, c3 = st.columns(3)
            dc_ = c1.selectbox("Columna de fecha", date_cols, key=wkey("ts_d"))
            vc2 = c2.selectbox("Variable", num_cols, key=wkey("ts_v"))
            fr = c3.selectbox("Frecuencia", ["D", "W", "ME", "QE", "YE"], index=2,
                              format_func=lambda f: {"D": "Diaria", "W": "Semanal",
                                                     "ME": "Mensual", "QE": "Trimestral",
                                                     "YE": "Anual"}[f], key=wkey("ts_f"))
            ts = eda.timeseries_summary(df, dc_, vc2, fr)
            if ts is not None and not ts.empty:
                chart(viz.line_timeseries(ts, dc_), key="g_ts")
                table(ts)
        else:
            st.info("Se necesita al menos una columna de fecha y una numérica. "
                    "Puedes convertir una columna a fecha en la pestaña ETL.")


# =========================================================================== #
# 5 · MODELOS
# =========================================================================== #
with tabs[4]:
    st.subheader("Entrenamiento y comparación de modelos")

    c1, c2, c3 = st.columns([2, 2, 2])
    target = c1.selectbox("Variable objetivo (a predecir)", list(df.columns),
                          index=len(df.columns) - 1, key=wkey("m_target"))
    tarea_auto = models.detect_task(df[target])
    forzar = c2.selectbox("Tipo de problema", ["Automático", "clasificacion", "regresion"],
                          key=wkey("m_task"))
    tarea = tarea_auto if forzar == "Automático" else models.detect_task(df[target], forzar)
    c2.caption(f"Detectado: **{tarea.replace('_', ' ')}**")

    excluir_sug = [c for c, r in roles.items() if r.role in ("id", "constante")
                   and c != target]
    feats = c3.multiselect("Variables predictoras",
                           [c for c in df.columns if c != target],
                           default=[c for c in df.columns
                                    if c != target and c not in excluir_sug],
                           key=wkey("m_feats"))
    if excluir_sug:
        c3.caption(f"Excluidas por defecto: {', '.join(excluir_sug[:5])}")

    zoo_nombres = list(models.model_zoo(tarea).keys())
    default_sel = [n for n in models.DEFAULT_SELECTION[
        "clasificacion" if models.is_classification(tarea) else "regresion"]
        if n in zoo_nombres]
    sel_models = st.multiselect("Modelos a entrenar", zoo_nombres, default=default_sel,
                                key=wkey("m_models"))

    with st.expander("⚙️ Configuración avanzada"):
        c1, c2, c3, c4 = st.columns(4)
        test_size = c1.slider("Tamaño del conjunto de prueba", 0.10, 0.40, 0.20, 0.05)
        cv_folds = c2.slider("Particiones de validación cruzada", 2, 10, 5)
        escalador = c3.selectbox("Escalado", ["standard", "minmax", "robust",
                                              "yeo-johnson", "ninguno"])
        codificador = c4.selectbox("Codificación de categóricas", ["onehot", "ordinal"])
        c1, c2, c3, c4 = st.columns(4)
        imp_num = c1.selectbox("Imputación numérica", ["median", "mean", "most_frequent"])
        imp_cat = c2.selectbox("Imputación categórica", ["most_frequent", "constant"])
        balanceo = c3.selectbox("Balanceo de clases", ["ninguno", "balanced"])
        semilla = c4.number_input("Semilla aleatoria", 0, 9999, 42)
        c1, c2, c3 = st.columns(3)
        hacer_cv = c1.checkbox("Ejecutar validación cruzada", True)
        tunear = c2.checkbox("Optimizar hiperparámetros (más lento)", False)
        n_iter = c3.slider("Combinaciones a probar", 5, 60, 15, disabled=not tunear)

    if st.button("🚀 Entrenar y comparar", type="primary", width="stretch"):
        if not feats:
            st.error("Selecciona al menos una variable predictora.")
        elif not sel_models:
            st.error("Selecciona al menos un modelo.")
        else:
            bar = st.progress(0.0, "Preparando…")
            t0 = time.perf_counter()
            try:
                SS.training = models.train_models(
                    df, target=target, features=feats, task=tarea,
                    selected_models=sel_models, test_size=test_size,
                    cv_folds=cv_folds, do_cv=hacer_cv, tune=tunear, tune_iter=n_iter,
                    scaler=escalador, encoder=codificador, num_impute=imp_num,
                    cat_impute=imp_cat,
                    class_weight=None if balanceo == "ninguno" else "balanced",
                    random_state=int(semilla),
                    progress_cb=lambda p, m: bar.progress(min(p, 1.0), m))
                SS.importancias = {}
                SS.shap_cache = None
                bar.empty()
                st.success(f"✔ {len(sel_models)} modelos entrenados en "
                           f"{time.perf_counter()-t0:.1f} s")
            except Exception as e:
                bar.empty()
                st.error(f"Error durante el entrenamiento: {e}")

    tr = SS.training
    if tr is None:
        st.info("Configura las opciones y pulsa **Entrenar y comparar**.")
    else:
        lb = tr.leaderboard
        ok = lb[lb["estado"] == "ok"]
        st.divider()
        st.markdown("### 🏆 Tabla comparativa")
        if not ok.empty:
            mejor = ok.iloc[0]
            c = st.columns(5)
            kpi(c[0], "Mejor modelo", mejor["modelo"])
            clf = models.is_classification(tr.task)
            if clf:
                kpi(c[1], "Accuracy", f"{mejor.get('Accuracy', float('nan')):.4f}")
                kpi(c[2], "F1", f"{mejor.get('F1', mejor.get('F1 ponderado', float('nan'))):.4f}")
                kpi(c[3], "ROC-AUC", f"{mejor.get('ROC-AUC', float('nan')):.4f}"
                    if "ROC-AUC" in mejor else "—")
            else:
                kpi(c[1], "R²", f"{mejor.get('R²', float('nan')):.4f}")
                kpi(c[2], "RMSE", f"{mejor.get('RMSE', float('nan')):,.3f}")
                kpi(c[3], "MAE", f"{mejor.get('MAE', float('nan')):,.3f}")
            kpi(c[4], "Registros de prueba", f"{tr.config['n_test']:,}")

        table(lb, height=min(480, 40 * len(lb) + 45))
        download_df(lb, "comparacion_modelos", "Tabla comparativa", "lb")

        metric_cols = [c for c in lb.columns if c in models.METRIC_LABELS.values()]
        c1, c2 = st.columns([3, 2])
        with c1:
            m_sel = st.selectbox("Métrica del gráfico", metric_cols,
                                 index=0 if metric_cols else None, key=wkey("lb_m"))
            if m_sel:
                chart(viz.bar_leaderboard(lb, m_sel), key="g_lb")
        with c2:
            st.markdown("**Cómo leer las métricas**")
            if models.is_classification(tr.task):
                st.caption(
                    "· **Accuracy**: % de aciertos. Engaña con clases desbalanceadas.\n\n"
                    "· **Accuracy balanceada / F1**: promedian el desempeño por clase.\n\n"
                    "· **Precisión**: de los que predije positivos, cuántos lo eran.\n\n"
                    "· **Recall**: de los positivos reales, cuántos detecté.\n\n"
                    "· **ROC-AUC**: capacidad de ordenar casos (0.5 = azar, 1 = perfecto).\n\n"
                    "· **MCC / Kappa**: robustos ante desbalance.\n\n"
                    "· **Sobreajuste**: diferencia entrenamiento − prueba; >0.10 es señal de alerta.")
            else:
                st.caption(
                    "· **R²**: proporción de la variabilidad explicada (1 = perfecto).\n\n"
                    "· **RMSE**: error típico en las unidades del objetivo; penaliza errores grandes.\n\n"
                    "· **MAE**: error absoluto promedio, más fácil de comunicar.\n\n"
                    "· **MAPE / sMAPE**: error en porcentaje.\n\n"
                    "· **RMSE normalizado**: error relativo al rango del objetivo.\n\n"
                    "· **Sobreajuste**: diferencia entrenamiento − prueba.")

        if len(ok) >= 2 and len(metric_cols) >= 3:
            c1, c2 = st.columns(2)
            with c1:
                chart(viz.grouped_metrics(lb, metric_cols[:6]), key="g_gm")
            with c2:
                chart(viz.radar_models(lb, metric_cols[:6]), key="g_rad")

        st.divider()
        st.markdown("### 🔬 Diagnóstico detallado")
        mod_sel = st.selectbox("Modelo", list(ok["modelo"]) if not ok.empty else [],
                               key=wkey("det_m"))
        if mod_sel:
            r = tr.results[mod_sel]
            if r.mejores_parametros:
                st.caption(f"Mejores hiperparámetros: `{r.mejores_parametros}`")

            if models.is_classification(tr.task):
                c1, c2 = st.columns(2)
                with c1:
                    norm = st.checkbox("Normalizar por clase real", True, key="cm_n")
                    cm = models.confusion(tr.y_test, r.y_pred,
                                          labels=list(range(len(tr.class_names))))
                    chart(viz.heatmap_confusion(cm, tr.class_names, norm), key="g_cm")
                with c2:
                    st.markdown("**Reporte por clase**")
                    table(models.report_table(tr.y_test, r.y_pred, tr.class_names))
                    st.caption("`support` = nº de casos reales de cada clase.")

                if r.y_proba is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        chart(viz.lines_roc(models.roc_data(
                            tr.y_test, r.y_proba, len(tr.class_names), tr.class_names)),
                            key="g_roc")
                    with c2:
                        prev = float(np.mean(np.asarray(tr.y_test) == 1)) \
                            if len(tr.class_names) == 2 else None
                        chart(viz.lines_pr(models.pr_data(
                            tr.y_test, r.y_proba, len(tr.class_names), tr.class_names),
                            prev), key="g_pr")

                    if len(tr.class_names) == 2:
                        st.markdown("**Análisis del umbral de decisión**")
                        th = models.threshold_analysis(tr.y_test, r.y_proba)
                        chart(viz.lines_threshold(th), key="g_th")
                        best_f1 = th.loc[th["f1"].idxmax()]
                        st.success(
                            f"Umbral que maximiza F1: **{best_f1['umbral']:.2f}** → "
                            f"precisión {best_f1['precision']:.3f}, "
                            f"recall {best_f1['recall']:.3f}, "
                            f"{best_f1['positivos_predichos_%']:.1f}% de casos marcados "
                            f"como positivos.")
                        with st.expander("Tabla completa de umbrales"):
                            table(th, height=340)
            else:
                res = models.residuals_frame(tr.y_test, r.y_pred)
                c1, c2 = st.columns(2)
                with c1:
                    chart(viz.scatter_pred_vs_real(res), key="g_pvr")
                with c2:
                    chart(viz.scatter_residuals(res), key="g_res")
                c1, c2 = st.columns([2, 1])
                with c1:
                    chart(viz.hist_residuals(res), key="g_hres")
                with c2:
                    st.markdown("**Errores más grandes**")
                    table(res.reindex(res["residuo_abs"].sort_values(ascending=False).index)
                          .head(12).round(3))

            with st.expander("Curva de aprendizaje (¿ayudarían más datos?)"):
                if st.button("Calcular curva de aprendizaje", key="btn_lc"):
                    with st.spinner("Calculando…"):
                        try:
                            lc = models.learning_curve_data(
                                r.pipeline, tr.X_train, tr.y_train,
                                cv=min(5, tr.config["cv_folds"]), task=tr.task)
                            chart(viz.lines_learning_curve(lc), key="g_lc")
                            brecha = lc.iloc[-1]["entrenamiento"] - lc.iloc[-1]["validacion"]
                            if brecha > 0.1:
                                st.warning(f"Brecha de {brecha:.3f} entre entrenamiento y "
                                           "validación: el modelo sobreajusta. Más datos o "
                                           "más regularización deberían ayudar.")
                            elif lc.iloc[-1]["validacion"] - lc.iloc[-3]["validacion"] > 0.01:
                                st.info("La curva de validación sigue subiendo: "
                                        "**más datos mejorarían el modelo**.")
                            else:
                                st.success("La curva se estabilizó: más datos del mismo tipo "
                                           "aportarían poco. El siguiente paso son mejores "
                                           "variables.")
                        except Exception as e:
                            st.error(f"No se pudo calcular: {e}")

            with st.expander("Métricas completas (prueba, entrenamiento y validación cruzada)"):
                comp = pd.DataFrame({
                    "métrica": [models.METRIC_LABELS.get(k, k) for k in r.metricas_test],
                    "prueba": list(r.metricas_test.values()),
                    "entrenamiento": [r.metricas_train.get(k) for k in r.metricas_test],
                })
                if r.metricas_cv:
                    cvdf = pd.DataFrame([{"métrica_cv": k, "valor": v}
                                         for k, v in r.metricas_cv.items()
                                         if not k.endswith("_std")])
                    c1, c2 = st.columns(2)
                    c1.dataframe(comp.round(4), hide_index=True)
                    c2.dataframe(cvdf.round(4), hide_index=True)
                else:
                    table(comp.round(4))


# =========================================================================== #
# 6 · VARIABLES CLAVE
# =========================================================================== #
with tabs[5]:
    st.subheader("Variables clave")
    tr = SS.training
    if tr is None:
        st.info("Entrena primero un modelo en la pestaña **Modelos**.")
    else:
        ok = tr.leaderboard[tr.leaderboard["estado"] == "ok"]
        mod = st.selectbox("Modelo a explicar", list(ok["modelo"]), key=wkey("imp_m"))
        r = tr.results[mod]
        feats = tr.features

        c1, c2, c3 = st.columns(3)
        calc_perm = c1.checkbox("Importancia por permutación", True)
        calc_shap = c2.checkbox("SHAP (más lento)", False)
        top_n = c3.slider("Variables a mostrar", 5, 40, 15)

        nat = importance.native_importance(r.pipeline, feats)
        frames = {"nativo": nat}

        sub = st.tabs(["Ranking", "Consenso", "SHAP", "Efecto marginal", "Coeficientes"])

        with sub[0]:
            c1, c2 = st.columns(2)
            with c1:
                chart(viz.bar_importance(nat, top_n, f"Importancia nativa · {mod}"),
                      key="g_nat")
                if nat is not None:
                    table(nat.head(top_n))
            with c2:
                if calc_perm:
                    ck = f"perm::{mod}"
                    if ck not in SS.importancias:
                        with st.spinner("Calculando importancia por permutación…"):
                            sc_ = ("roc_auc" if tr.task == "clasificacion_binaria"
                                   else ("f1_weighted" if models.is_classification(tr.task)
                                         else "r2"))
                            SS.importancias[ck] = importance.permutation_based(
                                r.pipeline, tr.X_test, tr.y_test, scoring=sc_)
                    perm = SS.importancias[ck]
                    frames["permutacion"] = perm
                    chart(viz.bar_importance(perm, top_n,
                                             "Importancia por permutación (impacto real)"),
                          key="g_perm")
                    table(perm.head(top_n))
                    st.caption("Mide cuánto empeora el modelo al desordenar cada variable: "
                               "es la medida más fiable de aporte real.")
                else:
                    st.info("Activa la casilla para calcular la importancia por permutación.")

            if nat is not None and not nat.empty:
                n80 = int((nat["acumulado_%"] <= 80).sum()) or 1
                st.success(f"**{n80} de {len(nat)} variables** concentran el 80% de la "
                           f"importancia: {', '.join(nat.head(n80)['variable'].tolist())}.")

        with sub[1]:
            if calc_shap and SS.shap_cache is not None:
                frames["shap"] = SS.shap_cache[0]
            cons = importance.importance_consensus(frames, k=top_n)
            if cons.empty:
                st.info("Activa más métodos para construir el consenso.")
            else:
                st.markdown("Ranking combinado de todos los métodos disponibles: "
                            "las variables que aparecen arriba en todos son las más sólidas.")
                chart(viz.bar_importance(
                    cons.rename(columns={"importancia_promedio_%": "importancia_%"}),
                    top_n, "Ranking de consenso"), key="g_cons")
                table(cons)

        with sub[2]:
            if not calc_shap:
                st.info("Marca **SHAP** arriba para calcular los valores de Shapley. "
                        "Explican el aporte de cada variable en cada predicción individual.")
            else:
                if SS.shap_cache is None or SS.shap_cache[-1] != mod:
                    with st.spinner("Calculando valores SHAP…"):
                        agg, sv, Xt, names = importance.shap_values(
                            r.pipeline, tr.X_test, feats, max_rows=400)
                        SS.shap_cache = (agg, sv, Xt, names, mod)
                agg, sv, Xt, names, _ = SS.shap_cache
                if agg is None and not importance.HAS_SHAP:
                    st.warning("La librería **shap** no está instalada en este entorno. "
                               "Instálala con `pip install shap` para habilitar esta "
                               "vista. Mientras tanto, la importancia por permutación "
                               "de la pestaña *Ranking* cumple la misma función.")
                elif agg is None:
                    st.warning("SHAP no pudo explicar este modelo. Prueba con un modelo "
                               "de árboles (Random Forest, XGBoost, LightGBM).")
                else:
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        chart(viz.bar_importance(agg, top_n, "Importancia SHAP media"),
                              key="g_shapbar")
                        table(agg.head(top_n))
                    with c2:
                        chart(viz.shap_beeswarm(sv, Xt, names, min(top_n, 15)),
                              key="g_shapbee")
                        st.caption("Cada punto es un registro. A la derecha del cero, la "
                                   "variable empuja la predicción hacia arriba; a la "
                                   "izquierda, hacia abajo. El color indica si el valor de "
                                   "la variable era alto o bajo.")

        with sub[3]:
            st.markdown("Muestra cómo cambia la predicción promedio al mover una sola "
                        "variable, manteniendo el resto constante.")
            base_rank = frames.get("permutacion", nat)
            opciones = (importance.top_features(base_rank, 25)
                        if base_rank is not None else feats)
            v_pdp = st.selectbox("Variable", [v for v in opciones if v in feats] or feats,
                                 key=wkey("pdp_v"))
            if st.button("Calcular efecto marginal", key="btn_pdp"):
                with st.spinner("Calculando…"):
                    pdp = importance.partial_dependence_curve(r.pipeline, tr.X_test, v_pdp)
                    if pdp is None:
                        st.warning("No se pudo calcular para esta variable.")
                    else:
                        chart(viz.line_pdp(pdp, v_pdp), key="g_pdp")
                        d = pdp["prediccion_promedio"]
                        tendencia = ("creciente" if d.iloc[-1] > d.iloc[0] else "decreciente")
                        st.info(f"Efecto **{tendencia}**: al pasar del valor mínimo al máximo "
                                f"de `{v_pdp}`, la predicción promedio cambia de "
                                f"{d.iloc[0]:.4f} a {d.iloc[-1]:.4f} "
                                f"({(d.iloc[-1]-d.iloc[0]):+.4f}).")
                        table(pdp.round(4))

        with sub[4]:
            coef = importance.signed_coefficients(r.pipeline, feats)
            if coef is None:
                st.info("Este modelo no es lineal. Selecciona Regresión Logística, "
                        "Lineal, Ridge, Lasso o Elastic Net para ver coeficientes con signo.")
            else:
                chart(viz.bar_coefficients(coef, top_n), key="g_coef")
                table(coef.head(top_n * 2).round(4))
                st.caption("Signo positivo = aumenta el valor (o la probabilidad) del "
                           "objetivo; negativo = lo reduce. Los valores están en la escala "
                           "de las variables ya estandarizadas.")


# =========================================================================== #
# 7 · CLUSTERS
# =========================================================================== #
with tabs[6]:
    st.subheader("Segmentación no supervisada (clusters)")
    st.caption("Descubre grupos naturales sin necesidad de una variable objetivo. "
               "Elige tus variables clave, pulsa **Analizar** y recorre las secciones: "
               "codo y silueta, K-Means, DBSCAN, árbol jerárquico, PCA y t-SNE.")

    # ------------------------------------------------------------------ #
    # Paso 1 · selección de variables clave
    # ------------------------------------------------------------------ #
    k_feats = wkey("cl_feats")
    candidatas_cl = [c for c in df.columns]

    def _set_feats(lista):
        SS[k_feats] = [c for c in lista if c in candidatas_cl]

    def _cb_sugeridas():
        excl = [SS.training.target] if SS.training is not None else []
        sel, _ = clustering.suggest_features(
            df, roles, exclude=excl,
            include_categorical=SS.get(wkey("cl_inc_cat"), False))
        _set_feats(sel)

    def _cb_numericas():
        _set_feats([c for c in num_cols if roles[c].role not in ("id", "constante")])

    def _cb_modelo():
        tr_ = SS.training
        if tr_ is None:
            return
        try:
            imp_ = importance.native_importance(
                tr_.results[tr_.best_model].pipeline, tr_.features)
            _set_feats([v for v in imp_["variable"].head(8) if v in df.columns])
        except Exception:
            pass

    def _cb_limpiar():
        _set_feats([])

    if k_feats not in SS:
        excl0 = [SS.training.target] if SS.training is not None else []
        SS[k_feats] = clustering.suggest_features(df, roles, exclude=excl0)[0]

    st.markdown("#### 1 · Variables clave para segmentar")
    b1, b2, b3, b4, b5 = st.columns([1.1, 1.1, 1.1, 0.8, 1.6])
    b1.button("✨ Sugeridas", on_click=_cb_sugeridas, width="stretch",
              help="Numéricas no redundantes y con más variabilidad; excluye IDs, "
                   "constantes y la variable objetivo.")
    b2.button("🔢 Todas las numéricas", on_click=_cb_numericas, width="stretch")
    b3.button("⭐ Top del modelo", on_click=_cb_modelo, width="stretch",
              disabled=SS.training is None,
              help="Las 8 variables más importantes del mejor modelo entrenado.")
    b4.button("🧹 Limpiar", on_click=_cb_limpiar, width="stretch")
    b5.checkbox("Incluir categóricas en la sugerencia", key=wkey("cl_inc_cat"))

    cl_feats = st.multiselect(
        "Variables seleccionadas (agrega o quita las que consideres clave)",
        candidatas_cl, key=k_feats)

    with st.expander("⚙️ Preprocesamiento"):
        p1, p2, p3 = st.columns(3)
        cl_scaler = p1.selectbox(
            "Escalado", ["standard", "robust", "minmax", "yeo-johnson"], key=wkey("cl_sc"),
            help="Indispensable: sin escalar, la variable de mayor magnitud domina las "
                 "distancias.")
        cl_enc = p2.selectbox("Codificación de categóricas", ["onehot", "ordinal"],
                              key=wkey("cl_enc"))
        max_rows = p3.number_input("Máximo de registros", 200, 100_000,
                                   int(min(len(df), 20_000)), step=500, key=wkey("cl_rows"))

    clave_actual = (f"{sorted(map(str, cl_feats))}|{cl_scaler}|{cl_enc}|{int(max_rows)}|"
                    f"{SS.get('data_version', 0)}|{json.dumps(SS.recipe, default=str)}")
    clx = SS.get("clx")
    vigente = clx is not None and clx.get("clave") == clave_actual

    ca, cb_ = st.columns([1, 3])
    lanzar = ca.button("🔍 Analizar clusters", type="primary", width="stretch",
                       disabled=not cl_feats)
    aviso = cb_.empty()   # se llena después de (posiblemente) calcular

    if lanzar and cl_feats:
        prog = st.progress(0.0, "Preparando la matriz…")
        try:
            M, Xu, pre = clustering.prepare_matrix(df, cl_feats, cl_scaler, cl_enc,
                                                   int(max_rows))
            prog.progress(0.10, "Midiendo la tendencia a formar grupos (Hopkins)…")
            hop = clustering.hopkins(M)
            prog.progress(0.18, "Barrido de k: método del codo y silueta…")
            ks = clustering.k_selection(M, 2, 10, "KMeans")
            prog.progress(0.45, "Proyección PCA…")
            idx_vis = clustering._sample_idx(len(M), 3000)
            pca_emb, pca_info = clustering.project(M, "PCA", 2)
            pca_var = clustering.pca_variance(M)
            try:
                nombres_t = list(pre.get_feature_names_out())
            except Exception:
                nombres_t = [f"f{i}" for i in range(M.shape[1])]
            loadings = clustering.pca_loadings(M, nombres_t, min(3, M.shape[1]))
            prog.progress(0.55, "Proyección t-SNE (la más lenta)…")
            tsne_emb, tsne_info = clustering.project(M[idx_vis], "t-SNE", 2)
            prog.progress(0.92, "Sugiriendo eps para DBSCAN…")
            eps_sug = clustering.suggest_eps(M, 5)
            kd = clustering.kdistance_curve(M, 5)
            k_rec = int(ks.loc[ks["silueta"].idxmax(), "k"]) if not ks.empty else 3
            SS.clx = {"clave": clave_actual, "M": M, "Xu": Xu, "pre": pre,
                      "feats": list(cl_feats), "hopkins": hop, "ks": ks,
                      "idx_vis": idx_vis, "pca_emb": pca_emb, "pca_info": pca_info,
                      "pca_var": pca_var, "loadings": loadings,
                      "tsne_emb": tsne_emb, "tsne_info": tsne_info,
                      "eps_sug": eps_sug, "kd": kd, "k_rec": k_rec,
                      "otros": {}}
            # los parámetros por algoritmo arrancan en los valores recomendados
            for k_ in ("km_k", "hc_k"):
                SS[wkey(k_)] = k_rec
            SS[wkey("db_eps")] = float(min(max(round(eps_sug, 3), 0.01), 100.0))
            prog.empty()
            clx, vigente = SS.clx, True
        except Exception as e:
            prog.empty()
            st.error(f"No se pudo analizar: {e}")

    if not cl_feats:
        aviso.warning("Selecciona al menos una variable.")
    elif clx is not None and not vigente:
        aviso.info("Cambiaste las variables o el preprocesamiento: pulsa **Analizar "
                   "clusters** para recalcular.")
    elif clx is None:
        aviso.info("Pulsa **Analizar clusters**: se calculan el codo, la silueta, PCA y "
                   "t-SNE una sola vez; luego ajustas cada algoritmo en vivo.")
    else:
        aviso.success(f"Análisis listo: {len(clx['feats'])} variables, "
                      f"{len(clx['M']):,} registros, k recomendado = {clx['k_rec']}. "
                      "Recorre las secciones de abajo.")

    # ------------------------------------------------------------------ #
    # Resultados
    # ------------------------------------------------------------------ #
    if clx is not None and vigente:
        M, Xu, feats_cl = clx["M"], clx["Xu"], clx["feats"]
        idx_vis = clx["idx_vis"]
        pca_vis = clx["pca_emb"].iloc[idx_vis].reset_index(drop=True)
        pca_vis.columns = ["PC1", "PC2"]
        tsne_vis = clx["tsne_emb"].copy()
        tsne_vis.columns = ["t-SNE 1", "t-SNE 2"]
        ck = clx["clave"]

        @st.cache_data(show_spinner=False, max_entries=40)
        def _km(_M, clave, k):
            return clustering.run_clustering(_M, "KMeans", n_clusters=int(k))

        @st.cache_data(show_spinner=False, max_entries=40)
        def _db(_M, clave, eps, ms):
            return clustering.run_clustering(_M, "DBSCAN", eps=float(eps),
                                             min_samples=int(ms))

        @st.cache_data(show_spinner=False, max_entries=40)
        def _hc(_M, clave, k, metodo):
            return clustering.hierarchical_fit(_M, int(k), metodo)

        def _vistas(labels, prefijo):
            """PCA y t-SNE lado a lado, coloreados con las mismas etiquetas."""
            lv = np.asarray(labels)[idx_vis]
            v1, v2 = st.columns(2)
            with v1:
                chart(viz.scatter_clusters_fast(pca_vis, lv, "Vista PCA"),
                      key=f"{prefijo}_pca")
            with v2:
                chart(viz.scatter_clusters_fast(tsne_vis, lv, "Vista t-SNE"),
                      key=f"{prefijo}_tsne")
            if len(M) > len(idx_vis):
                st.caption(f"Se grafican {len(idx_vis):,} de {len(M):,} registros "
                           "(muestra aleatoria) para que los mapas sean fluidos; las "
                           "métricas usan todos los registros.")

        def _kpis_cluster(met, extra=None):
            c = st.columns(4)
            kpi(c[0], "Grupos", met.get("n_clusters", "—"))
            sil = met.get("silueta")
            kpi(c[1], "Silueta", f"{sil:.3f}" if sil is not None else "—",
                "−1 a 1 · mayor es mejor")
            dbi = met.get("davies_bouldin")
            kpi(c[2], "Davies-Bouldin", f"{dbi:.3f}" if dbi is not None else "—",
                "menor es mejor")
            if extra:
                kpi(c[3], extra[0], extra[1], extra[2] if len(extra) > 2 else "")
            else:
                chi = met.get("calinski_harabasz")
                kpi(c[3], "Calinski-Harabasz", f"{chi:,.0f}" if chi is not None else "—",
                    "mayor es mejor")

        def _tamanos_y_centroides(labels, prefijo):
            sizes = clustering.cluster_sizes(labels)
            t1, t2 = st.columns([2, 3])
            with t1:
                chart(viz.bar_cluster_sizes(sizes), key=f"{prefijo}_sz")
            with t2:
                st.markdown("**Promedio de cada variable por grupo** (unidades originales)")
                cen = clustering.centroids_table(Xu, labels, feats_cl)
                if not cen.empty:
                    table(cen)
                else:
                    table(sizes)

        # --- resultados vigentes de los tres algoritmos principales ---
        km_k = int(SS.get(wkey("km_k"), clx["k_rec"]))
        db_eps = float(SS.get(wkey("db_eps"), clx["eps_sug"]))
        db_ms = int(SS.get(wkey("db_ms"), 5))
        hc_k = int(SS.get(wkey("hc_k"), clx["k_rec"]))
        hc_m = SS.get(wkey("hc_m"), "ward")

        sub = st.tabs(["① Variables clave", "② Codo y silueta", "③ K-Means",
                       "④ DBSCAN", "⑤ Árbol jerárquico", "⑥ PCA y t-SNE",
                       "⑦ Comparar y perfilar"])

        # ---------------- ① Variables clave ----------------
        with sub[0]:
            c = st.columns(4)
            kpi(c[0], "Variables", len(feats_cl), f"{M.shape[1]} columnas tras codificar")
            kpi(c[1], "Registros", f"{len(M):,}")
            h = clx["hopkins"]
            kpi(c[2], "Hopkins", f"{h:.3f}" if np.isfinite(h) else "—",
                clustering.hopkins_label(h))
            kpi(c[3], "k recomendado", clx["k_rec"], "por mejor silueta")
            st.write("")
            if np.isfinite(h) and h < 0.6:
                st.warning("El estadístico de Hopkins indica que estas variables casi no "
                           "forman grupos naturales. Prueba otras variables clave antes "
                           "de interpretar los segmentos.")
            elif np.isfinite(h):
                st.success(f"Hopkins = {h:.3f}: {clustering.hopkins_label(h)}. "
                           "Valores cercanos a 0.5 indicarían datos aleatorios.")

            excl = [SS.training.target] if SS.training is not None else []
            _, tabla_sug = clustering.suggest_features(df, roles, exclude=excl)
            tabla_sug["seleccionada"] = tabla_sug["variable"].isin(feats_cl)
            st.markdown("**Diagnóstico de cada variable** — por qué se sugiere o no")
            table(tabla_sug[["variable", "rol", "seleccionada", "sugerida", "motivo"]],
                  height=min(420, 36 * len(tabla_sug) + 45))

            num_sel = [f for f in feats_cl if is_numeric(df[f])]
            if len(num_sel) >= 2:
                corr_sel = eda.correlation_matrix(Xu, "pearson", num_sel)
                c1, c2 = st.columns([3, 2])
                with c1:
                    chart(viz.heatmap_corr(corr_sel, "Correlación entre las variables "
                                                     "elegidas"), key="clv_corr")
                with c2:
                    red = eda.top_correlations(corr_sel, 0.85)
                    if red.empty:
                        st.success("Sin redundancias fuertes (|r| > 0.85) entre las "
                                   "variables elegidas.")
                    else:
                        st.warning("Pares redundantes: pesan doble en la distancia. "
                                   "Considera quitar una de cada par.")
                        table(red)
                    st.markdown("**Resumen de las variables**")
                    table(eda.describe_numeric(Xu, num_sel)[
                        ["variable", "media", "mediana", "desv_std", "min", "max"]])

        # ---------------- ② Codo y silueta ----------------
        with sub[1]:
            r1, r2, r3 = st.columns([1, 1, 2])
            kmin = r1.number_input("k mínimo", 2, 20, 2, key=wkey("kmin"))
            kmax = r2.number_input("k máximo", 3, 30, 10, key=wkey("kmax"))
            if r3.button("Recalcular con este rango", key=wkey("btn_ks")):
                with st.spinner("Probando cada k…"):
                    clx["ks"] = clustering.k_selection(M, int(kmin), int(kmax), "KMeans")
            ks = clx["ks"]
            if ks is None or ks.empty:
                st.warning("No se pudo calcular el barrido de k.")
            else:
                g1, g2 = st.columns(2)
                with g1:
                    chart(viz.line_elbow(ks), key="cl_elbow")
                    st.caption("**Método del codo:** la inercia (distancia interna de los "
                               "grupos) siempre baja al aumentar k. El k adecuado es donde "
                               "la curva deja de bajar fuerte y se 'dobla'.")
                with g2:
                    chart(viz.line_silhouette_sweep(ks), key="cl_silsweep")
                    st.caption("**Silueta:** mide qué tan separado está cada grupo de los "
                               "demás (−1 a 1). Más alto es mejor: >0.5 separación "
                               "fuerte, 0.25–0.5 razonable, <0.25 grupos solapados.")
                g3, g4 = st.columns(2)
                with g3:
                    chart(viz.line_index_sweep(ks, "calinski_harabasz",
                                               "Calinski-Harabasz (mayor = mejor)"),
                          key="cl_ch")
                with g4:
                    chart(viz.line_index_sweep(ks, "davies_bouldin",
                                               "Davies-Bouldin (menor = mejor)", "min"),
                          key="cl_db")
                table(ks)
                rec = ks[ks["recomendado"] != ""]
                if not rec.empty:
                    st.success("**Recomendación:** " + " · ".join(
                        f"k = {int(r['k'])} ({r['recomendado']})" for _, r in rec.iterrows())
                        + ". Cuando los criterios no coinciden, prioriza la silueta y "
                          "elige el k que tenga sentido de negocio.")

        # ---------------- ③ K-Means ----------------
        with sub[2]:
            st.markdown("Divide los datos en **k grupos esféricos** alrededor de un "
                        "centro. Es el más usado: rápido e interpretable, pero exige "
                        "definir k y asume grupos de tamaño y forma parecidos.")
            st.number_input("Número de grupos (k)", 2, 30, key=wkey("km_k"),
                            help=f"Recomendado por silueta: {clx['k_rec']}")
            km_k = int(SS[wkey("km_k")])
            with st.spinner("Ajustando K-Means…"):
                res_km = _km(M, ck, km_k)
            if res_km.error:
                st.error(res_km.error)
            else:
                inercia = getattr(res_km.modelo, "inertia_", None)
                _kpis_cluster(res_km.metricas,
                              ("Inercia", f"{inercia:,.0f}" if inercia else "—",
                               "suma de distancias al centro"))
                _vistas(res_km.labels, "km")
                _tamanos_y_centroides(res_km.labels, "km")
                with st.expander("Silueta por registro"):
                    sil = clustering.silhouette_by_point(M, res_km.labels)
                    chart(viz.silhouette_plot(sil), key="km_silp")

        # ---------------- ④ DBSCAN ----------------
        with sub[3]:
            st.markdown("Agrupa por **densidad**: descubre solo el número de grupos, "
                        "admite formas irregulares y marca como **ruido** (−1) los "
                        "puntos aislados. Se controla con `eps` (radio de vecindad) y "
                        "`min_samples` (vecinos mínimos para formar un núcleo).")
            d1, d2 = st.columns(2)
            d1.number_input("eps (radio)", 0.01, 100.0, step=0.05, format="%.3f",
                            key=wkey("db_eps"),
                            help=f"Sugerido por la curva k-distancia: {clx['eps_sug']:.3f}")
            d2.number_input("min_samples", 2, 500, 5, key=wkey("db_ms"))
            db_eps = float(SS[wkey("db_eps")])
            db_ms = int(SS[wkey("db_ms")])
            chart(viz.line_kdistance(clx["kd"], db_eps), key="db_kd")
            st.caption(f"El codo de esta curva sugiere eps ≈ **{clx['eps_sug']:.3f}**. "
                       "Si subes eps, se fusionan grupos; si lo bajas, aumenta el ruido.")
            with st.spinner("Ajustando DBSCAN…"):
                res_db = _db(M, ck, db_eps, db_ms)
            if res_db.error:
                st.error(res_db.error)
            else:
                _kpis_cluster(res_db.metricas,
                              ("Ruido", f"{res_db.metricas.get('%_ruido', 0)}%",
                               f"{res_db.n_ruido:,} registros"))
                if res_db.n_clusters == 0:
                    st.warning("Todos los registros quedaron como ruido: **sube eps** o "
                               "baja min_samples.")
                elif res_db.n_clusters == 1:
                    st.warning("Todo quedó en un solo grupo: **baja eps**.")
                elif res_db.metricas.get("%_ruido", 0) > 30:
                    st.info("Más del 30% es ruido: considera subir eps un poco.")
                _vistas(res_db.labels, "db")
                _tamanos_y_centroides(res_db.labels, "db")

        # ---------------- ⑤ Jerárquico ----------------
        with sub[4]:
            st.markdown("Construye un **árbol** uniendo paso a paso los registros y grupos "
                        "más parecidos. Cortar el árbol a cierta altura define los "
                        "grupos: por eso no hay que fijar k de antemano, se decide "
                        "mirando el dendrograma.")
            metodos = {
                "ward": "Ward — minimiza la varianza interna (grupos compactos, el más usado)",
                "complete": "Completo — distancia entre los puntos más lejanos",
                "average": "Promedio — distancia media entre todos los pares",
                "single": "Simple — distancia entre los más cercanos (encadena grupos)",
            }
            h1, h2 = st.columns([2, 1])
            h1.selectbox("Método de enlace", list(metodos), key=wkey("hc_m"),
                         format_func=lambda m: metodos[m])
            h2.number_input("Número de grupos (corte)", 2, 30, key=wkey("hc_k"))
            hc_k = int(SS[wkey("hc_k")])
            hc_m = SS[wkey("hc_m")]
            with st.spinner("Construyendo el árbol…"):
                res_hc = _hc(M, ck, hc_k, hc_m)
            chart(viz.dendrogram_cut(res_hc["Z"], res_hc["corte"], hc_k), key="hc_dend")
            st.caption("Cada unión es una fusión; su altura es la distancia a la que se "
                       "unieron. Un salto vertical grande antes de la línea roja indica "
                       "un buen lugar para cortar."
                       + (f" El árbol se construye con {res_hc['n_muestra']:,} registros "
                          "y el resto se asigna al grupo más cercano."
                          if res_hc["n_muestra"] < len(M) else ""))
            _kpis_cluster(res_hc["metricas"])
            _vistas(res_hc["labels"], "hc")
            _tamanos_y_centroides(res_hc["labels"], "hc")

        # ---------------- ⑥ PCA y t-SNE ----------------
        with sub[5]:
            st.markdown("**PCA** es una proyección lineal: conserva las distancias "
                        "globales y sus ejes se interpretan con las cargas de las "
                        "variables. **t-SNE** es no lineal: preserva los vecindarios "
                        "locales, así que separa mejor los grupos visualmente, pero las "
                        "distancias entre grupos y los ejes no tienen significado.")
            st.markdown("##### PCA")
            p1, p2 = st.columns(2)
            with p1:
                chart(viz.bar_scree(clx["pca_var"]), key="pca_scree")
            with p2:
                chart(viz.line_cumvar(clx["pca_var"]), key="pca_cum")
            var = clx["pca_var"]
            n80 = int((var["acumulada_%"] < 80).sum()) + 1
            st.caption(f"Los 2 primeros componentes (los del mapa) explican "
                       f"**{var['acumulada_%'].iloc[min(1, len(var)-1)]:.1f}%** de la "
                       f"varianza; se necesitan **{min(n80, len(var))}** componentes "
                       "para llegar al 80%.")
            chart(viz.heatmap_loadings(clx["loadings"]), key="pca_load")

            st.markdown("##### Forma natural de los datos (sin agrupar)")
            f1, f2 = st.columns(2)
            with f1:
                chart(viz.scatter_plain(pca_vis, "PCA"), key="raw_pca")
            with f2:
                chart(viz.scatter_plain(tsne_vis, "t-SNE"), key="raw_tsne")

            with st.expander("Ajustar t-SNE"):
                q1, q2 = st.columns([2, 1])
                perp = q1.slider("Perplejidad", 5, 100, 30, key=wkey("tsne_perp"),
                                 help="Cuántos vecinos considera cada punto. Valores "
                                      "bajos resaltan grupos pequeños; altos, la "
                                      "estructura global.")
                if q2.button("Recalcular t-SNE", key=wkey("btn_tsne")):
                    with st.spinner("Recalculando t-SNE…"):
                        emb_, info_ = clustering.project(M[idx_vis], "t-SNE", 2,
                                                         perplexity=float(perp))
                        clx["tsne_emb"], clx["tsne_info"] = emb_, info_
                    st.rerun()

            st.markdown("##### Los tres algoritmos en ambas proyecciones")
            st.caption("Misma muestra de puntos en todas las vistas: compara cómo "
                       "agrupa cada algoritmo con los parámetros actuales.")
            filas_cmp = [("K-Means", _km(M, ck, km_k).labels),
                         ("DBSCAN", _db(M, ck, db_eps, db_ms).labels),
                         ("Jerárquico", _hc(M, ck, hc_k, hc_m)["labels"])]
            for nombre_a, labs in filas_cmp:
                if labs is None or len(labs) == 0:
                    continue
                lv = np.asarray(labs)[idx_vis]
                c1, c2 = st.columns(2)
                with c1:
                    chart(viz.scatter_clusters_fast(pca_vis, lv, f"{nombre_a} · PCA",
                                                    height=380), key=f"cmp_{nombre_a}_p")
                with c2:
                    chart(viz.scatter_clusters_fast(tsne_vis, lv, f"{nombre_a} · t-SNE",
                                                    height=380), key=f"cmp_{nombre_a}_t")

        # ---------------- ⑦ Comparar y perfilar ----------------
        with sub[6]:
            res_km = _km(M, ck, km_k)
            res_db = _db(M, ck, db_eps, db_ms)
            hc = _hc(M, ck, hc_k, hc_m)
            res_hc = clustering.ClusterResult(
                algoritmo=f"Jerárquico ({hc_m})", labels=hc["labels"],
                n_clusters=hc["metricas"]["n_clusters"], n_ruido=0,
                metricas=hc["metricas"], parametros={"k": hc_k, "enlace": hc_m})
            resultados = {"K-Means": res_km, "DBSCAN": res_db,
                          f"Jerárquico ({hc_m})": res_hc}

            with st.expander("➕ Probar otros algoritmos (Gaussian Mixture, HDBSCAN, "
                             "OPTICS, Birch, Mean Shift, Spectral…)"):
                otros_disp = [a for a in clustering.ALGORITHMS
                              if a not in ("KMeans", "DBSCAN", "Jerárquico (Ward)")
                              and (a != "HDBSCAN" or clustering.HAS_HDBSCAN)]
                sel_otros = st.multiselect("Algoritmos", otros_disp,
                                           default=["Gaussian Mixture"],
                                           key=wkey("cl_otros"))
                if st.button("Ejecutar", key=wkey("btn_otros")):
                    with st.spinner("Ejecutando…"):
                        _, extra = clustering.compare_algorithms(
                            M, sel_otros, n_clusters=km_k, eps=db_eps,
                            min_samples=db_ms)
                        clx["otros"] = extra
                resultados.update(clx.get("otros", {}))

            filas = []
            for nombre_a, r in resultados.items():
                if r.error:
                    filas.append({"algoritmo": nombre_a, "estado": "error",
                                  "detalle": r.error})
                    continue
                filas.append({"algoritmo": nombre_a, "estado": "ok",
                              "grupos": r.n_clusters, "ruido_%": r.metricas.get("%_ruido"),
                              "silueta": r.metricas.get("silueta"),
                              "davies_bouldin": r.metricas.get("davies_bouldin"),
                              "calinski_harabasz": r.metricas.get("calinski_harabasz"),
                              "parámetros": str(r.parametros)})
            comp = pd.DataFrame(filas)
            if "silueta" in comp.columns:
                comp = comp.sort_values("silueta", ascending=False,
                                        na_position="last").reset_index(drop=True)
            st.markdown("#### Comparación de algoritmos")
            table(comp)
            validos = [n for n, r in resultados.items()
                       if not r.error and r.n_clusters >= 1]
            if (not comp.empty and "silueta" in comp.columns
                    and pd.notna(comp.iloc[0].get("silueta"))):
                b = comp.iloc[0]
                st.success(f"Mejor separación con los parámetros actuales: "
                           f"**{b['algoritmo']}** — {int(b['grupos'])} grupos, "
                           f"silueta {b['silueta']}.")

            # contrato con Insights y Exportar
            SS.cluster = {"resultados": resultados, "M": M, "Xu": Xu,
                          "feats": feats_cl, "pre": clx["pre"], "tabla": comp}

            if validos:
                st.divider()
                algo_p = st.selectbox("Algoritmo a perfilar", validos, key=wkey("cl_prof"))
                res = resultados[algo_p]
                profile = clustering.profile_clusters(Xu, res.labels, feats_cl)
                sizes = clustering.cluster_sizes(res.labels)
                nombres_seg = clustering.name_clusters(profile, sizes)

                st.markdown("#### Identidad de cada segmento")
                for _, row in nombres_seg.iterrows():
                    st.markdown(
                        f"<div class='card'><h4>{row['nombre_sugerido']} · "
                        f"{row['n_registros']:,} registros ({row['%_del_total']}%)"
                        f"</h4><p>{md2html(row['caracteristicas'])}</p></div>",
                        unsafe_allow_html=True)
                chart(viz.heatmap_cluster_profile(profile), key="prof_heat")
                st.caption("Rojo = el segmento está por encima del promedio global en esa "
                           "variable; azul = por debajo (en desviaciones estándar).")

                st.markdown("#### Cruce con una variable de negocio")
                tgt_c = st.selectbox("Variable", list(df.columns), key=wkey("cl_tgt"))
                cross = clustering.target_by_cluster(df.loc[Xu.index], res.labels, tgt_c)
                if cross is not None:
                    table(cross)
                    numcols = [c for c in cross.columns if c != "cluster"
                               and pd.api.types.is_numeric_dtype(cross[c])]
                    if numcols:
                        colv = st.selectbox("Métrica a graficar", numcols,
                                            key=wkey("cl_crossv"))
                        chart(viz.bar_target_by_cluster(cross, colv), key="prof_cross")

                st.markdown("#### Acciones recomendadas por segmento")
                table(insights.segment_actions(profile, sizes))

                etiquetado = df.loc[Xu.index].copy()
                etiquetado["cluster"] = res.labels
                etiquetado = etiquetado.merge(
                    nombres_seg[["cluster", "nombre_sugerido"]], on="cluster", how="left")
                download_df(etiquetado, "datos_con_segmento",
                            "Datos con la etiqueta de segmento", "cletq")
    elif clx is None and cl_feats:
        excl = [SS.training.target] if SS.training is not None else []
        _, tabla_sug = clustering.suggest_features(df, roles, exclude=excl)
        with st.expander("¿Por qué estas variables sugeridas?", expanded=False):
            table(tabla_sug)


# =========================================================================== #
# 8 · INSIGHTS
# =========================================================================== #
with tabs[7]:
    st.subheader("Análisis descriptivo y prescriptivo")
    tr = SS.training
    imp_df = None
    if tr is not None:
        try:
            imp_df = importance.native_importance(
                tr.results[tr.best_model].pipeline, tr.features)
        except Exception:
            imp_df = None
    cinfo = None
    if SS.cluster:
        validos = [r for r in SS.cluster["resultados"].values()
                   if not r.error and r.n_clusters > 0]
        if validos:
            mejor_c = max(validos, key=lambda r: r.metricas.get("silueta") or -1)
            cinfo = {"n_clusters": mejor_c.n_clusters,
                     "algoritmo": mejor_c.algoritmo,
                     "silueta": mejor_c.metricas.get("silueta"),
                     "%_ruido": mejor_c.metricas.get("%_ruido")}

    st.markdown("### 📌 Resumen ejecutivo")
    st.markdown(f"<div class='card'><p>{md2html(insights.executive_summary(diag, score, tr, imp_df, cinfo))}</p></div>",
                unsafe_allow_html=True)

    sub = st.tabs(["Descriptivo", "Prescriptivo", "Simulador what-if",
                   "Optimizador de decisiones"])

    with sub[0]:
        obj = tr.target if tr is not None else None
        corr_top = None
        if len(num_cols) >= 2:
            corr_top = eda.top_correlations(eda.correlation_matrix(df), 0.3)
        trel = eda.target_relationship(df, obj) if obj else None
        for sec in insights.descriptive_report(df, diag, score, obj, corr_top, trel):
            st.markdown(f"<div class='card'><h4>{sec['titulo']}</h4>"
                        f"<p>{md2html(sec['texto'])}</p></div>", unsafe_allow_html=True)

    with sub[1]:
        plan = insights.prescriptive_actions(df, diag, score, findings, tr, imp_df,
                                             cinfo, tr.target if tr else None)
        st.markdown("### Plan de acción priorizado")
        for _, a in plan.iterrows():
            sev = a["prioridad"].split("·")[-1].strip()
            st.markdown(
                f"<div class='card'>{sev_pill(sev)}"
                f"<span class='pill p-baja'>{a['area']}</span>"
                f"<b> {md2html(a['accion'])}</b>"
                f"<p style='margin-top:7px'><b>Por qué:</b> {md2html(a['por_que'])}<br>"
                f"<b>Impacto esperado:</b> {md2html(a['impacto_esperado'])}<br>"
                f"<b>Cómo hacerlo:</b> {md2html(a['como_hacerlo'])}</p></div>",
                unsafe_allow_html=True)
        download_df(plan, "plan_de_accion", "Plan de acción", "plan")

    with sub[2]:
        if tr is None:
            st.info("Entrena un modelo para simular escenarios.")
        else:
            st.markdown("Construye un caso y observa la predicción del modelo. "
                        "Después mueve una variable para ver su efecto.")
            mod_w = st.selectbox("Modelo", list(
                tr.leaderboard[tr.leaderboard["estado"] == "ok"]["modelo"]), key=wkey("wf_m"))
            pipe = tr.results[mod_w].pipeline
            Xr = tr.X_train

            base = {}
            cols_w = st.columns(3)
            for i, f in enumerate(tr.features):
                col = cols_w[i % 3]
                s = Xr[f]
                if is_numeric(s):
                    lo, hi = float(s.min()), float(s.max())
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        # Columna constante: un deslizador necesita un rango real
                        base[f] = col.number_input(f, value=float(lo if np.isfinite(lo) else 0.0),
                                                   key=wkey(f"wf_{f}"))
                    else:
                        base[f] = col.slider(f, lo, hi, float(s.median()),
                                             (hi - lo) / 100, key=wkey(f"wf_{f}"))
                elif is_datetime(s):
                    base[f] = pd.to_datetime(col.date_input(
                        f, value=pd.to_datetime(s.median()), key=wkey(f"wf_{f}")))
                else:
                    opts = [str(v) for v in s.astype(str).value_counts().head(40).index]
                    base[f] = col.selectbox(f, opts, key=wkey(f"wf_{f}"))

            pred = insights.predict_row(pipe, base, tr.features, tr.label_encoder, tr.task)
            st.divider()
            c1, c2 = st.columns([1, 2])
            with c1:
                if models.is_classification(tr.task):
                    st.metric("Predicción", pred["prediccion"],
                              f"confianza {pred.get('confianza', 0):.1%}")
                else:
                    st.metric("Predicción", f"{pred['prediccion']:,.4f}")
            with c2:
                if pred.get("probabilidades"):
                    chart(viz.bar_probabilities(pred["probabilidades"]), key="g_prob")

            st.markdown("#### Sensibilidad a una variable")
            v_sens = st.selectbox("Variable a mover", tr.features, key=wkey("wf_sv"))
            s = Xr[v_sens]
            if is_numeric(s):
                vals = list(np.linspace(float(s.min()), float(s.max()), 25))
            else:
                vals = [str(v) for v in s.astype(str).value_counts().head(15).index]
            sens = insights.sensitivity_analysis(pipe, base, v_sens, vals, tr.features,
                                                 tr.label_encoder, tr.task)
            if not sens.empty:
                ycol = "probabilidad" if "probabilidad" in sens.columns else "prediccion"
                chart(viz.line_sensitivity(sens, v_sens, ycol), key="g_sens")
                table(sens.round(4))

    with sub[3]:
        if tr is None:
            st.info("Entrena un modelo para optimizar decisiones.")
        else:
            st.markdown("Elige las variables que **sí puedes controlar** (precio, plan, "
                        "descuento, canal, frecuencia de contacto…). El sistema prueba "
                        "todas las combinaciones y ordena los escenarios por el resultado "
                        "esperado.")
            mod_o = st.selectbox("Modelo", list(
                tr.leaderboard[tr.leaderboard["estado"] == "ok"]["modelo"]), key=wkey("op_m"))
            pipe = tr.results[mod_o].pipeline
            Xr = tr.X_train

            palancas_sel = st.multiselect("Variables accionables (máximo 4)", tr.features,
                                          max_selections=4, key=wkey("op_lev"))
            objetivo_dir = st.radio("Objetivo", ["maximizar", "minimizar"], horizontal=True,
                                    key=wkey("op_dir"))
            clase_pos = 1
            if models.is_classification(tr.task) and tr.class_names:
                clase_nom = st.selectbox("Clase de interés", tr.class_names,
                                         index=min(1, len(tr.class_names) - 1), key=wkey("op_cls"))
                clase_pos = tr.class_names.index(clase_nom)

            st.markdown("**Escenario base** (el resto de variables se mantiene fijo)")
            base_o = {}
            cols_o = st.columns(4)
            for i, f in enumerate(tr.features):
                s = Xr[f]
                if is_numeric(s):
                    base_o[f] = float(s.median())
                elif is_datetime(s):
                    base_o[f] = pd.to_datetime(s.median())
                else:
                    m = s.astype(str).mode()
                    base_o[f] = str(m.iloc[0]) if len(m) else ""
            with st.expander("Ajustar el escenario base"):
                cols_b = st.columns(3)
                for i, f in enumerate(tr.features):
                    if f in palancas_sel:
                        continue
                    s = Xr[f]
                    col = cols_b[i % 3]
                    if is_numeric(s):
                        base_o[f] = col.number_input(f, value=float(s.median()),
                                                     key=wkey(f"ob_{f}"))
                    elif not is_datetime(s):
                        opts = [str(v) for v in s.astype(str).value_counts().head(30).index]
                        base_o[f] = col.selectbox(f, opts, key=wkey(f"ob_{f}"))

            levers: dict[str, list] = {}
            if palancas_sel:
                st.markdown("**Valores a probar en cada palanca**")
                cols_l = st.columns(min(len(palancas_sel), 4))
                for i, f in enumerate(palancas_sel):
                    s = Xr[f]
                    with cols_l[i % len(cols_l)]:
                        if is_numeric(s):
                            vmin, vmax = float(s.min()), float(s.max())
                            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                                st.caption(f"`{f}` no varía en los datos: no sirve "
                                           "como palanca.")
                                levers[f] = [vmin if np.isfinite(vmin) else 0.0]
                            else:
                                lo = float(s.quantile(0.05))
                                hi = float(s.quantile(0.95))
                                lo, hi = max(vmin, min(lo, vmax)), min(vmax, max(hi, vmin))
                                if hi <= lo:
                                    lo, hi = vmin, vmax
                                rng = st.slider(f"{f} · rango", vmin, vmax, (lo, hi),
                                                key=wkey(f"lv_{f}"))
                                n_pts = st.number_input(f"{f} · puntos", 2, 15, 5,
                                                        key=wkey(f"lvn_{f}"))
                                levers[f] = list(np.round(
                                    np.linspace(rng[0], rng[1], int(n_pts)), 4))
                        else:
                            opts = [str(v) for v in s.astype(str).value_counts().head(20).index]
                            levers[f] = st.multiselect(f, opts, default=opts[:4],
                                                       key=wkey(f"lv_{f}"))

            if st.button("🎯 Buscar el mejor escenario", type="primary", key="btn_opt"):
                if not levers or any(not v for v in levers.values()):
                    st.error("Define al menos una palanca con valores a probar.")
                else:
                    with st.spinner("Simulando escenarios…"):
                        opt = insights.prescriptive_optimizer(
                            pipe, base_o, levers, tr.features, tr.task, objetivo_dir,
                            clase_pos, label_encoder=tr.label_encoder)
                    if "error" in opt.columns:
                        st.error(opt.iloc[0]["error"])
                    else:
                        ycol = ("probabilidad_objetivo" if "probabilidad_objetivo" in opt.columns
                                else "resultado_esperado")
                        base_val = opt.attrs.get("valor_base", float("nan"))
                        c1, c2, c3 = st.columns(3)
                        kpi(c1, "Escenario actual", f"{base_val:,.4f}")
                        kpi(c2, "Mejor escenario", f"{opt.iloc[0][ycol]:,.4f}")
                        kpi(c3, "Mejora", f"{opt.iloc[0]['vs_escenario_actual']:+,.4f}",
                            f"{opt.iloc[0]['mejora_%']:+.1f}%")
                        st.markdown("#### Recomendaciones")
                        for frase in insights.actionable_recommendations(opt, levers, ycol):
                            st.markdown(f"- {frase}")
                        chart(viz.bar_scenarios(opt, ycol, list(levers.keys())), key="g_opt")
                        table(opt.head(60))
                        download_df(opt, "escenarios_simulados", "Escenarios", "opt")


# =========================================================================== #
# 9 · EXPORTAR
# =========================================================================== #
with tabs[8]:
    st.subheader("Exportar resultados")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Datos")
        download_df(df, "datos_transformados", "Datos transformados", "exp_data")
        st.caption(f"{len(df):,} filas × {df.shape[1]} columnas, con las "
                   f"{len(SS.recipe)} transformaciones aplicadas.")
        st.write("")
        st.markdown("#### Reportes de análisis")
        download_df(_profile(df), "perfil_calidad", "Perfil de calidad", "exp_prof")
        if num_cols:
            download_df(eda.describe_numeric(df), "estadistica_descriptiva",
                        "Estadística descriptiva", "exp_desc")

    with c2:
        st.markdown("#### Modelo entrenado")
        if SS.training is None:
            st.info("Aún no has entrenado ningún modelo.")
        else:
            tr = SS.training
            ok = tr.leaderboard[tr.leaderboard["estado"] == "ok"]
            mod_e = st.selectbox("Modelo a exportar", list(ok["modelo"]), key=wkey("exp_m"))
            try:
                import joblib
                buf = io.BytesIO()
                joblib.dump({"pipeline": tr.results[mod_e].pipeline,
                             "features": tr.features, "target": tr.target,
                             "task": tr.task, "class_names": tr.class_names,
                             "label_encoder": tr.label_encoder,
                             "config": tr.config}, buf)
                st.download_button(f"⬇ Descargar «{mod_e}» (.joblib)", buf.getvalue(),
                                   f"modelo_{mod_e.replace(' ', '_').lower()}.joblib",
                                   "application/octet-stream")
                st.code(f"""import joblib, pandas as pd

art = joblib.load("modelo_{mod_e.replace(' ', '_').lower()}.joblib")
modelo, variables = art["pipeline"], art["features"]

nuevos = pd.read_csv("nuevos_datos.csv")
pred = modelo.predict(nuevos[variables])

# Para clasificación, recuperar las etiquetas originales:
if art["label_encoder"] is not None:
    pred = art["label_encoder"].inverse_transform(pred)
    proba = modelo.predict_proba(nuevos[variables])
""", language="python")
            except Exception as e:
                st.warning(f"No se pudo serializar el modelo: {e}")

            download_df(tr.leaderboard, "comparacion_modelos", "Tabla comparativa",
                        "exp_lb")

            preds = pd.DataFrame({"real": tr.y_test})
            for n, r in tr.results.items():
                if r.y_pred is not None:
                    preds[f"pred_{n}"] = r.y_pred
            if tr.label_encoder is not None:
                for c in preds.columns:
                    try:
                        preds[c] = tr.label_encoder.inverse_transform(
                            preds[c].astype(int))
                    except Exception:
                        pass
            download_df(preds, "predicciones_prueba", "Predicciones del conjunto de prueba",
                        "exp_pred")

    st.divider()
    st.markdown("#### Reporte ejecutivo completo (Markdown)")
    if st.button("Generar reporte", key="btn_rep"):
        tr = SS.training
        imp_df = None
        if tr is not None:
            try:
                imp_df = importance.native_importance(
                    tr.results[tr.best_model].pipeline, tr.features)
            except Exception:
                pass
        cinfo = None
        if SS.cluster:
            v = [r for r in SS.cluster["resultados"].values()
                 if not r.error and r.n_clusters > 0]
            if v:
                b = max(v, key=lambda r: r.metricas.get("silueta") or -1)
                cinfo = {"n_clusters": b.n_clusters, "algoritmo": b.algoritmo,
                         "silueta": b.metricas.get("silueta"),
                         "%_ruido": b.metricas.get("%_ruido")}

        partes = [f"# Reporte de análisis — {SS.meta.get('archivo', 'datos')}",
                  f"_Generado el {pd.Timestamp.now():%Y-%m-%d %H:%M}_", "",
                  "## Resumen ejecutivo",
                  insights.executive_summary(diag, score, tr, imp_df, cinfo), "",
                  "## Análisis descriptivo"]
        obj = tr.target if tr is not None else None
        ct = eda.top_correlations(eda.correlation_matrix(df), 0.3) if len(num_cols) >= 2 else None
        for sec in insights.descriptive_report(df, diag, score, obj, ct,
                                               eda.target_relationship(df, obj) if obj else None):
            partes += [f"### {sec['titulo']}", sec["texto"], ""]

        partes += ["## Calidad de los datos",
                   f"- Score global: **{score['score_total']}/100** ({score['nivel']})",
                   f"- Completitud: {score['completitud']} · Unicidad: {score['unicidad']} · "
                   f"Consistencia: {score['consistencia']} · Validez: {score['validez']}", ""]
        for f in findings:
            partes.append(f"- **[{f['severidad']}] {f['hallazgo']}** — {f['detalle']} "
                          f"→ _{f['accion_sugerida']}_")
        partes.append("")

        if SS.recipe:
            partes += ["## Transformaciones aplicadas"]
            for i, p in enumerate(SS.recipe, 1):
                partes.append(f"{i}. {etl.OPERATION_LABELS.get(p['op'], p['op'])} "
                              f"— `{ {k: v for k, v in p.items() if k != 'op'} }`")
            partes.append("")

        if tr is not None:
            partes += ["## Comparación de modelos", "",
                       tr.leaderboard.to_markdown(index=False), ""]
            if imp_df is not None:
                partes += ["## Variables clave", "",
                           imp_df.head(15).to_markdown(index=False), ""]

        if cinfo:
            partes += ["## Segmentación",
                       f"- Segmentos detectados: **{cinfo['n_clusters']}**",
                       f"- Silueta: {cinfo['silueta']}", ""]

        plan = insights.prescriptive_actions(df, diag, score, findings, tr, imp_df, cinfo,
                                             tr.target if tr else None)
        partes += ["## Plan de acción prescriptivo", "", plan.to_markdown(index=False)]

        reporte = "\n".join(partes)
        st.download_button("⬇ Descargar reporte (Markdown)", reporte.encode("utf-8"),
                           "reporte_automl.md", "text/markdown")
        with st.expander("Vista previa"):
            st.markdown(reporte)
