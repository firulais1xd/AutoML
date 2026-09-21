"""
Motor de análisis descriptivo y prescriptivo.

- Descriptivo: qué dicen los datos (narrativa automática sobre volumen,
  calidad, distribuciones, relaciones y segmentos).
- Prescriptivo: qué hacer al respecto (acciones sobre los datos, sobre el
  modelo y sobre el negocio, además de simulación what-if y búsqueda de
  la mejor combinación de palancas accionables).
"""
from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd

from .schema import is_numeric, is_texty
from .models import is_classification, METRIC_LABELS

warnings.filterwarnings("ignore")


def _fmt(v, nd: int = 2) -> str:
    try:
        f = float(v)
        if abs(f) >= 1000:
            return f"{f:,.0f}"
        return f"{f:,.{nd}f}"
    except Exception:
        return str(v)


# =========================================================================== #
# 1. DESCRIPTIVO
# =========================================================================== #
def descriptive_report(df: pd.DataFrame, diag: dict, score: dict,
                       target: str | None = None,
                       corr_top: pd.DataFrame | None = None,
                       target_rel: pd.DataFrame | None = None) -> list[dict]:
    """Narrativa descriptiva estructurada en secciones."""
    secciones: list[dict] = []
    n, m = diag["filas"], diag["columnas"]
    roles = diag.get("roles", {})
    conteo_roles = pd.Series(list(roles.values())).value_counts().to_dict()

    # --- Panorama -----------------------------------------------------------
    partes = [
        f"El conjunto contiene **{n:,} registros** y **{m} variables**, "
        f"ocupando {diag['memoria_mb']} MB en memoria.",
        "Composición: " + ", ".join(
            f"{v} {k}{'s' if v > 1 and not k.endswith('s') else ''}"
            for k, v in conteo_roles.items()) + ".",
    ]
    if n < 100:
        partes.append("⚠️ Con menos de 100 registros los resultados de cualquier "
                      "modelo serán inestables: se recomienda ampliar la muestra.")
    elif n < 1000:
        partes.append("La muestra es pequeña-moderada; conviene apoyarse en "
                      "validación cruzada más que en un único split.")
    secciones.append({"titulo": "Panorama general", "texto": " ".join(partes)})

    # --- Calidad ------------------------------------------------------------
    partes = [
        f"La calidad global se califica como **{score['nivel']}** "
        f"({score['score_total']}/100): completitud {score['completitud']}, "
        f"unicidad {score['unicidad']}, consistencia {score['consistencia']}, "
        f"validez {score['validez']}."
    ]
    if diag["%_celdas_nulas"] > 0:
        partes.append(f"Hay {diag['celdas_nulas']:,} celdas vacías "
                      f"({diag['%_celdas_nulas']}% del total).")
    else:
        partes.append("No hay valores faltantes.")
    if diag["filas_duplicadas"]:
        partes.append(f"Se detectaron {diag['filas_duplicadas']:,} filas duplicadas "
                      f"({diag['%_filas_duplicadas']}%).")
    if diag["columnas_constantes"]:
        partes.append(f"Columnas sin variabilidad: {', '.join(diag['columnas_constantes'][:6])}.")
    if diag["columnas_identificador"]:
        partes.append(f"Identificadores detectados: {', '.join(diag['columnas_identificador'][:6])} "
                      "(deben excluirse del modelo).")
    secciones.append({"titulo": "Calidad de los datos", "texto": " ".join(partes)})

    # --- Distribuciones -----------------------------------------------------
    num_cols = [c for c in df.columns if is_numeric(df[c])]
    if num_cols:
        sesgadas, dispersas = [], []
        for c in num_cols:
            s = df[c].dropna()
            if len(s) < 10:
                continue
            if abs(s.skew()) > 1.5:
                sesgadas.append(c)
            if s.mean() and abs(s.std() / s.mean()) > 1.5:
                dispersas.append(c)
        partes = [f"Se analizaron {len(num_cols)} variables numéricas."]
        if sesgadas:
            partes.append(f"Con fuerte asimetría: **{', '.join(sesgadas[:6])}** — "
                          "candidatas a transformación logarítmica o Yeo-Johnson.")
        if dispersas:
            partes.append(f"Alta dispersión relativa (CV>150%): {', '.join(dispersas[:6])}.")
        if not sesgadas and not dispersas:
            partes.append("Las distribuciones son razonablemente simétricas y estables.")
        secciones.append({"titulo": "Distribuciones numéricas", "texto": " ".join(partes)})

    cat_cols = [c for c in df.columns if is_texty(df[c])]
    if cat_cols:
        desbal, alta_card = [], []
        for c in cat_cols:
            vc = df[c].astype(str).value_counts(normalize=True)
            if len(vc) and vc.iloc[0] > 0.9:
                desbal.append(f"{c} ({vc.index[0]}: {vc.iloc[0]*100:.0f}%)")
            if df[c].nunique() > 50:
                alta_card.append(c)
        partes = [f"Hay {len(cat_cols)} variables categóricas o de texto."]
        if desbal:
            partes.append("Muy concentradas en un solo valor: " + "; ".join(desbal[:5]) + ".")
        if alta_card:
            partes.append(f"Alta cardinalidad (>50 niveles): {', '.join(alta_card[:5])} — "
                          "conviene agrupar las categorías poco frecuentes.")
        secciones.append({"titulo": "Variables categóricas", "texto": " ".join(partes)})

    # --- Relaciones ---------------------------------------------------------
    if corr_top is not None and not corr_top.empty:
        top3 = corr_top.head(3)
        frases = [f"{r['variable_1']} ↔ {r['variable_2']} (r={r['correlacion']:.2f}, {r['fuerza']})"
                  for _, r in top3.iterrows()]
        texto = "Las asociaciones más fuertes entre variables numéricas son: " + "; ".join(frases) + "."
        redundantes = corr_top[corr_top["correlacion"].abs() >= 0.9]
        if not redundantes.empty:
            pares = [f"{r['variable_1']}/{r['variable_2']}" for _, r in redundantes.iterrows()]
            texto += (f" Atención: {', '.join(pares[:5])} son casi redundantes "
                      "(multicolinealidad); conviene conservar solo una de cada par.")
        secciones.append({"titulo": "Relaciones entre variables", "texto": texto})

    # --- Objetivo -----------------------------------------------------------
    if target and target in df.columns:
        y = df[target]
        if is_numeric(y):
            s = y.dropna()
            texto = (f"La variable objetivo **{target}** es numérica: media {_fmt(s.mean())}, "
                     f"mediana {_fmt(s.median())}, rango [{_fmt(s.min())} – {_fmt(s.max())}], "
                     f"desviación {_fmt(s.std())}.")
            if abs(s.skew()) > 1:
                texto += (f" Su distribución está sesgada (asimetría {s.skew():.2f}); "
                          "transformarla con log suele mejorar los modelos lineales.")
        else:
            vc = y.astype(str).value_counts(normalize=True)
            texto = (f"La variable objetivo **{target}** tiene {len(vc)} clases. "
                     "Distribución: " +
                     ", ".join(f"{i} ({v*100:.1f}%)" for i, v in vc.head(6).items()) + ".")
            if len(vc) > 1 and vc.iloc[-1] < 0.10:
                texto += (f" ⚠️ Hay desbalance importante (la clase minoritaria es "
                          f"{vc.iloc[-1]*100:.1f}%): usa accuracy balanceada, F1 o ROC-AUC "
                          "en lugar de accuracy, y activa el ponderado de clases.")
        if target_rel is not None and not target_rel.empty:
            top = target_rel.head(5)
            frases = [f"{r['variable']} ({r['metrica']}={r['valor']})" for _, r in top.iterrows()]
            texto += " Las variables más asociadas al objetivo son: " + ", ".join(frases) + "."
        secciones.append({"titulo": f"Variable objetivo: {target}", "texto": texto})

    return secciones


# =========================================================================== #
# 2. PRESCRIPTIVO
# =========================================================================== #
def prescriptive_actions(df: pd.DataFrame, diag: dict, score: dict,
                         findings: list[dict] | None = None,
                         training=None, importance: pd.DataFrame | None = None,
                         cluster_info: dict | None = None,
                         target: str | None = None) -> pd.DataFrame:
    """
    Plan de acción priorizado: qué hacer, por qué y con qué impacto esperado.
    """
    acciones: list[dict] = []

    def add(prioridad, area, accion, motivo, impacto, como):
        acciones.append({"prioridad": prioridad, "area": area, "accion": accion,
                         "por_que": motivo, "impacto_esperado": impacto, "como_hacerlo": como})

    # ---------- Datos ----------
    for f in (findings or []):
        if f["severidad"] == "Alta":
            add("1 · Crítica", "Datos", f["accion_sugerida"], f["detalle"],
                "Evita sesgos y errores de entrenamiento",
                "Pestaña ETL → aplicar la limpieza sugerida")
        elif f["severidad"] == "Media":
            add("2 · Alta", "Datos", f["accion_sugerida"], f["detalle"],
                "Mejora la estabilidad del modelo",
                "Pestaña ETL")

    if score["score_total"] < 70:
        add("1 · Crítica", "Datos",
            "Ejecutar el plan de limpieza automático antes de modelar",
            f"El score de calidad es {score['score_total']}/100 ({score['nivel']}).",
            "Un dataset limpio suele mejorar las métricas entre 3 y 15 puntos",
            "Pestaña ETL → botón «Sugerir limpieza automática»")

    if diag["filas"] < 500:
        add("2 · Alta", "Datos", "Ampliar el volumen de datos",
            f"Solo hay {diag['filas']:,} registros.",
            "Reduce la varianza de las métricas y el sobreajuste",
            "Recolectar más histórico o unir fuentes adicionales")

    # ---------- Modelo ----------
    if training is not None and not training.leaderboard.empty:
        lb = training.leaderboard[training.leaderboard["estado"] == "ok"]
        if not lb.empty:
            best = lb.iloc[0]
            clf = is_classification(training.task)
            metric_col = ("ROC-AUC" if "ROC-AUC" in lb.columns else
                          ("Accuracy" if "Accuracy" in lb.columns else
                           ("R²" if "R²" in lb.columns else None)))

            if metric_col:
                val = best[metric_col]
                add("1 · Crítica", "Modelo",
                    f"Poner en producción **{best['modelo']}**",
                    f"Es el mejor modelo con {metric_col} = {val}.",
                    "Referencia de desempeño para la decisión de negocio",
                    "Pestaña Modelos → Descargar modelo (.joblib)")

                # Baseline
                base_rows = lb[lb["modelo"].str.contains("Baseline", case=False, na=False)]
                if not base_rows.empty and metric_col in base_rows.columns:
                    base_val = base_rows.iloc[0][metric_col]
                    lift = val - base_val
                    if lift <= 0.02:
                        add("1 · Crítica", "Modelo",
                            "Revisar el planteamiento: el modelo casi no supera al baseline",
                            f"Mejora de solo {lift:.3f} frente al modelo trivial.",
                            "Evita desplegar un modelo sin valor real",
                            "Incorporar nuevas variables explicativas o replantear el objetivo")

            # Sobreajuste
            if "sobreajuste" in lb.columns:
                over = lb.iloc[0].get("sobreajuste")
                if over is not None and over == over and over > 0.10:
                    add("2 · Alta", "Modelo",
                        "Reducir el sobreajuste del modelo ganador",
                        f"La diferencia entre entrenamiento y prueba es {over:.3f}.",
                        "Mejor generalización con datos nuevos",
                        "Limitar profundidad, subir min_samples_leaf, aumentar "
                        "regularización o recolectar más datos")

            # Desempeño bajo
            if metric_col and best[metric_col] == best[metric_col]:
                v = float(best[metric_col])
                umbral = 0.70 if clf else 0.50
                if v < umbral:
                    add("1 · Crítica", "Modelo",
                        "Enriquecer las variables predictoras",
                        f"{metric_col} = {v:.3f}, por debajo de un nivel utilizable.",
                        "Es la palanca con mayor retorno cuando el desempeño es bajo",
                        "Crear variables derivadas (ratios, agregaciones por cliente/periodo, "
                        "componentes de fecha) en la pestaña ETL")
                elif v > 0.98 and clf:
                    add("1 · Crítica", "Modelo",
                        "Verificar fuga de información (data leakage)",
                        f"{metric_col} = {v:.3f} es sospechosamente alto.",
                        "Evita un modelo que falla en producción",
                        "Revisar si alguna variable contiene información del futuro "
                        "o deriva directamente del objetivo")

            # Costo/beneficio de la complejidad
            if len(lb) > 2 and metric_col:
                simples = lb[lb["modelo"].isin(
                    ["Regresión Logística", "Regresión Lineal", "Ridge",
                     "Árbol de Decisión"])]
                if not simples.empty:
                    dif = float(best[metric_col]) - float(simples.iloc[0][metric_col])
                    if dif < 0.02:
                        add("3 · Media", "Modelo",
                            f"Considerar el modelo simple **{simples.iloc[0]['modelo']}**",
                            f"Solo es {dif:.3f} peor que el ganador pero es interpretable "
                            "y más barato de mantener.",
                            "Menor costo operativo y mayor aceptación del negocio",
                            "Comparar en la pestaña Modelos")

            if not training.config.get("tune"):
                add("3 · Media", "Modelo", "Optimizar hiperparámetros del modelo ganador",
                    "El entrenamiento se hizo con parámetros por defecto.",
                    "Típicamente aporta entre 1 y 5 puntos adicionales",
                    "Pestaña Modelos → activar «Optimizar hiperparámetros»")

            if clf and training.class_names and len(training.class_names) == 2:
                add("2 · Alta", "Negocio",
                    "Calibrar el umbral de decisión según el costo del error",
                    "El umbral por defecto (0.5) rara vez es el óptimo económico.",
                    "Maximiza el beneficio neto de las decisiones automatizadas",
                    "Pestaña Modelos → Análisis de umbral")

    # ---------- Variables ----------
    if importance is not None and not importance.empty:
        top = importance.head(5)["variable"].tolist()
        add("2 · Alta", "Negocio",
            f"Concentrar la gestión en: {', '.join(top)}",
            "Son las variables que más explican el resultado.",
            "Enfoca recursos donde el modelo detecta mayor influencia",
            "Pestaña Variables clave → revisar dirección del efecto (SHAP / PDP)")

        acum = importance.copy()
        pocas = acum[acum["acumulado_%"] <= 80]
        if len(pocas) < len(acum) * 0.5 and len(pocas) >= 1:
            add("3 · Media", "Modelo",
                f"Simplificar el modelo a {len(pocas)} variables",
                f"Esas {len(pocas)} explican el 80% de la importancia total.",
                "Modelo más rápido, barato de alimentar y más robusto",
                "Reentrenar seleccionando solo esas variables")

        irrelevantes = importance[importance["importancia_%"] < 0.5]["variable"].tolist()
        if len(irrelevantes) >= 3:
            add("4 · Baja", "Datos",
                f"Evaluar eliminar {len(irrelevantes)} variables sin aporte",
                f"Aportan menos del 0.5% cada una: {', '.join(irrelevantes[:6])}…",
                "Reduce ruido y costo de captura de datos",
                "Pestaña ETL → eliminar columnas")

    # ---------- Clusters ----------
    if cluster_info:
        n_cl = cluster_info.get("n_clusters", 0)
        sil = cluster_info.get("silueta")
        if n_cl:
            add("2 · Alta", "Negocio",
                f"Diseñar estrategias diferenciadas para los {n_cl} segmentos detectados",
                (f"La segmentación tiene silueta {sil}" if sil else
                 "Se identificaron segmentos con perfiles distintos") + ".",
                "Personalización que suele superar a una estrategia única",
                "Pestaña Clusters → revisar el perfil de cada segmento")
        if sil is not None and sil < 0.25:
            add("3 · Media", "Clusters",
                "Revisar la segmentación: los grupos se solapan",
                f"La silueta ({sil}) indica separación débil.",
                "Segmentos más accionables",
                "Probar otro algoritmo (HDBSCAN/Gaussian Mixture), otro k, "
                "o reducir el número de variables de entrada")
        ruido = cluster_info.get("%_ruido")
        if ruido and ruido > 20:
            add("3 · Media", "Clusters",
                f"Analizar el {ruido}% de registros marcados como ruido",
                "DBSCAN/HDBSCAN los considera atípicos.",
                "Pueden ser fraudes, errores de captura u oportunidades nicho",
                "Exportar esos registros y revisarlos manualmente")

    if not acciones:
        add("4 · Baja", "General", "Sin acciones críticas pendientes",
            "El diagnóstico no encontró problemas relevantes.",
            "—", "Continuar al despliegue")

    out = pd.DataFrame(acciones).drop_duplicates(subset=["accion"])
    return out.sort_values("prioridad").reset_index(drop=True)


def executive_summary(diag: dict, score: dict, training=None,
                      importance: pd.DataFrame | None = None,
                      cluster_info: dict | None = None,
                      target: str | None = None) -> str:
    """Resumen ejecutivo en un párrafo, listo para presentar."""
    p = [f"Se analizaron **{diag['filas']:,} registros** con **{diag['columnas']} variables**. "
         f"La calidad de los datos es **{score['nivel']}** ({score['score_total']}/100)."]

    if diag["%_celdas_nulas"] > 5 or diag["filas_duplicadas"] > 0:
        p.append(f"Se identificaron problemas de completitud ({diag['%_celdas_nulas']}% de "
                 f"celdas vacías) y {diag['filas_duplicadas']:,} duplicados que deben "
                 "corregirse antes de decidir sobre los resultados.")

    if training is not None and not training.leaderboard.empty:
        lb = training.leaderboard[training.leaderboard["estado"] == "ok"]
        if not lb.empty:
            best = lb.iloc[0]
            clf = is_classification(training.task)
            if clf:
                acc = best.get("Accuracy")
                auc = best.get("ROC-AUC")
                txt = (f"El mejor modelo es **{best['modelo']}**, "
                       f"con accuracy de {acc:.1%}" if acc == acc else
                       f"El mejor modelo es **{best['modelo']}**")
                if auc == auc and auc is not None:
                    txt += f" y ROC-AUC de {auc:.3f}"
                txt += (f", evaluado sobre {training.config['n_test']:,} registros "
                        "no vistos en el entrenamiento.")
            else:
                r2 = best.get("R²")
                rmse = best.get("RMSE")
                txt = (f"El mejor modelo es **{best['modelo']}**, que explica el "
                       f"{r2:.1%} de la variabilidad de {training.target}"
                       if r2 == r2 else f"El mejor modelo es **{best['modelo']}**")
                if rmse == rmse and rmse is not None:
                    txt += f" con un error típico (RMSE) de {_fmt(rmse)}"
                txt += "."
            p.append(txt)

    if importance is not None and not importance.empty:
        top = importance.head(3)
        p.append("Los factores determinantes son " + ", ".join(
            f"**{r['variable']}** ({r['importancia_%']:.1f}%)" for _, r in top.iterrows()) +
            ", que concentran " + f"{top['importancia_%'].sum():.0f}% de la capacidad explicativa.")

    if cluster_info and cluster_info.get("n_clusters"):
        algo = cluster_info.get("algoritmo")
        sil = cluster_info.get("silueta")
        frase = (f"La segmentación no supervisada identificó **{cluster_info['n_clusters']} "
                 "grupos** con comportamientos diferenciados")
        if algo:
            frase += f" (algoritmo {algo}"
            frase += f", silueta {sil})" if sil is not None else ")"
        p.append(frase + ", base para estrategias específicas por segmento.")

    return " ".join(p)


# =========================================================================== #
# 3. SIMULACIÓN WHAT-IF Y OPTIMIZACIÓN PRESCRIPTIVA
# =========================================================================== #
def predict_row(pipeline, row: dict, features: list[str], label_encoder=None,
                task: str = "regresion") -> dict:
    """Predicción para un registro individual construido por el usuario."""
    X = pd.DataFrame([{f: row.get(f) for f in features}])
    out: dict[str, Any] = {}
    pred = pipeline.predict(X)[0]
    if is_classification(task) and label_encoder is not None:
        out["prediccion"] = str(label_encoder.inverse_transform([int(pred)])[0])
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X)[0]
                out["probabilidades"] = {
                    str(c): round(float(p), 4)
                    for c, p in zip(label_encoder.classes_, proba)}
                out["confianza"] = round(float(np.max(proba)), 4)
            except Exception:
                pass
    else:
        out["prediccion"] = float(pred)
    return out


def sensitivity_analysis(pipeline, base_row: dict, feature: str,
                         values: list, features: list[str],
                         label_encoder=None, task: str = "regresion",
                         positive_class_idx: int = 1) -> pd.DataFrame:
    """Cómo cambia la predicción al mover una sola variable."""
    rows = []
    for v in values:
        r = dict(base_row)
        r[feature] = v
        X = pd.DataFrame([{f: r.get(f) for f in features}])
        try:
            if is_classification(task) and hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(X)[0]
                idx = min(positive_class_idx, len(proba) - 1)
                rows.append({feature: v, "probabilidad": round(float(proba[idx]), 4),
                             "clase_predicha": str(
                                 label_encoder.inverse_transform([int(np.argmax(proba))])[0])
                             if label_encoder is not None else int(np.argmax(proba))})
            else:
                rows.append({feature: v, "prediccion": float(pipeline.predict(X)[0])})
        except Exception:
            continue
    return pd.DataFrame(rows)


def prescriptive_optimizer(pipeline, base_row: dict, levers: dict[str, list],
                           features: list[str], task: str = "regresion",
                           objective: str = "maximizar",
                           positive_class_idx: int = 1,
                           max_combos: int = 4000,
                           label_encoder=None) -> pd.DataFrame:
    """
    Busca la mejor combinación de variables accionables (palancas).

    `levers` = {variable: [valores a probar]}.
    Devuelve el ranking de escenarios con su resultado esperado.
    """
    if not levers:
        return pd.DataFrame()

    keys = list(levers.keys())
    grids = [levers[k] for k in keys]
    combos = list(itertools.product(*grids))[:max_combos]

    rows_in = []
    for combo in combos:
        r = dict(base_row)
        r.update(dict(zip(keys, combo)))
        rows_in.append({f: r.get(f) for f in features})
    X = pd.DataFrame(rows_in)

    try:
        if is_classification(task) and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)
            idx = min(positive_class_idx, proba.shape[1] - 1)
            score = proba[:, idx]
            col = "probabilidad_objetivo"
        else:
            score = pipeline.predict(X)
            col = "resultado_esperado"
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

    out = pd.DataFrame(combos, columns=keys)
    out[col] = np.round(score, 5)

    # Escenario base
    Xb = pd.DataFrame([{f: base_row.get(f) for f in features}])
    try:
        if is_classification(task) and hasattr(pipeline, "predict_proba"):
            base_val = float(pipeline.predict_proba(Xb)[0][
                min(positive_class_idx, pipeline.predict_proba(Xb).shape[1] - 1)])
        else:
            base_val = float(pipeline.predict(Xb)[0])
    except Exception:
        base_val = np.nan

    out["vs_escenario_actual"] = (out[col] - base_val).round(5)
    out["mejora_%"] = ((out[col] - base_val) / (abs(base_val) or 1) * 100).round(2)
    out = out.sort_values(col, ascending=(objective == "minimizar")).reset_index(drop=True)
    out.insert(0, "escenario", range(1, len(out) + 1))
    out.attrs["valor_base"] = base_val
    return out


def actionable_recommendations(opt: pd.DataFrame, levers: dict[str, list],
                               col: str | None = None, top: int = 3) -> list[str]:
    """Traduce el resultado del optimizador a frases accionables."""
    if opt is None or opt.empty or "error" in opt.columns:
        return []
    col = col or ("probabilidad_objetivo" if "probabilidad_objetivo" in opt.columns
                  else "resultado_esperado")
    base = opt.attrs.get("valor_base", np.nan)
    frases = []
    for _, r in opt.head(top).iterrows():
        cambios = ", ".join(f"**{k} = {r[k]}**" for k in levers.keys())
        delta = r.get("vs_escenario_actual")
        frase = f"Ajustando {cambios} → resultado esperado {_fmt(r[col], 4)}"
        if delta == delta and base == base:
            signo = "+" if delta >= 0 else ""
            frase += f" ({signo}{_fmt(delta, 4)} frente al escenario actual de {_fmt(base, 4)})"
        frases.append(frase + ".")
    return frases


def segment_actions(profile: pd.DataFrame, sizes: pd.DataFrame,
                    target_cross: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recomendación prescriptiva por segmento."""
    from .clustering import cluster_signatures
    sig = cluster_signatures(profile, top_n=3)
    rows = []
    for _, r in sizes.iterrows():
        cl = int(r["cluster"])
        share = r["%_del_total"]
        if cl == -1:
            accion = ("Revisar manualmente: son registros atípicos. Pueden indicar "
                      "errores de captura, fraude o nichos poco frecuentes.")
            prioridad = "Media"
        elif share >= 30:
            accion = ("Segmento masivo: estandarizar y automatizar la atención; "
                      "cualquier mejora aquí tiene impacto a gran escala.")
            prioridad = "Alta"
        elif share <= 5:
            accion = ("Segmento pequeño: evaluar si justifica una estrategia propia "
                      "o si conviene fusionarlo con otro grupo.")
            prioridad = "Baja"
        else:
            accion = ("Segmento intermedio: diseñar una oferta o tratamiento específico "
                      "basado en sus rasgos distintivos.")
            prioridad = "Media"
        rows.append({
            "cluster": cl, "tamaño": f"{int(r['n_registros']):,} ({share}%)",
            "rasgos_distintivos": " | ".join(sig.get(cl, [])) or "—",
            "prioridad": prioridad, "accion_recomendada": accion,
        })
    return pd.DataFrame(rows)
