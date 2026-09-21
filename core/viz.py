"""
Sistema visual del tablero: paleta validada, temas y constructores de gráficos.

Reglas aplicadas:
- Paleta categórica de orden fijo (nunca se recicla ni se genera un color nuevo).
- Secuencial = un solo tono azul (magnitud). Divergente = azul↔rojo con gris
  neutro al centro (polaridad: correlaciones, desviaciones).
- Nunca dos ejes Y en el mismo gráfico: se usan gráficos separados.
- Leyenda siempre que haya 2+ series; identidad reforzada con símbolo además
  del color en los diagramas de dispersión.
- Toda vista de color va acompañada de su tabla de datos en la interfaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

DIVERGING = [[0.0, "#0d366b"], [0.15, "#256abf"], [0.35, "#86b6ef"],
             [0.5, "#f0efec"], [0.65, "#f0a0a0"], [0.85, "#e34948"],
             [1.0, "#8f1f1f"]]

SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_3 = "#88867e"
GRID = "#e8e7e3"

STATUS = {"bueno": "#008300", "advertencia": "#eda100",
          "serio": "#eb6834", "critico": "#e34948", "neutro": "#88867e"}

SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x",
           "star", "hexagon", "triangle-down", "pentagon", "bowtie", "hourglass"]

FONT = dict(family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
            size=13, color=TEXT_1)


def _headroom_x(fig: go.Figure, valores, factor: float = 1.18) -> go.Figure:
    """Deja aire a la derecha para que las etiquetas de las barras no se corten."""
    v = np.asarray([x for x in np.asarray(valores, dtype=float) if np.isfinite(x)])
    if v.size == 0:
        return fig
    hi, lo = float(v.max()), float(min(0.0, float(v.min())))
    if hi <= lo:
        return fig
    fig.update_xaxes(range=[lo * factor if lo < 0 else 0, hi * factor])
    return fig


def _headroom_y(fig: go.Figure, valores, factor: float = 1.18) -> go.Figure:
    v = np.asarray([x for x in np.asarray(valores, dtype=float) if np.isfinite(x)])
    if v.size == 0:
        return fig
    hi = float(v.max())
    if hi <= 0:
        return fig
    fig.update_yaxes(range=[0, hi * factor])
    return fig


def color_for(i: int) -> str:
    """Color categórico por posición fija; a partir del 8º se repite el ciclo
    únicamente acompañado de símbolo distinto (codificación secundaria)."""
    return CAT[i % len(CAT)]


def symbol_for(i: int) -> str:
    return SYMBOLS[i % len(SYMBOLS)]


def base_layout(fig: go.Figure, title: str = "", height: int = 380,
                showlegend: bool | None = None, xtitle: str = "",
                ytitle: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_1), x=0, xanchor="left",
                   pad=dict(b=10)) if title else None,
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=FONT, height=height,
        margin=dict(l=10, r=14, t=46 if title else 18, b=10),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=GRID,
                        font=dict(size=12, color=TEXT_1)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(size=12, color=TEXT_2), bgcolor="rgba(0,0,0,0)"),
    )
    if showlegend is not None:
        # Plotly rechaza numpy.bool_: forzamos bool nativo de Python
        fig.update_layout(showlegend=bool(showlegend))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID,
                     ticks="outside", tickcolor=GRID, ticklen=4,
                     tickfont=dict(size=11, color=TEXT_2),
                     title=dict(text=xtitle, font=dict(size=12, color=TEXT_2)))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
                     linecolor="rgba(0,0,0,0)", ticks="",
                     tickfont=dict(size=11, color=TEXT_2),
                     title=dict(text=ytitle, font=dict(size=12, color=TEXT_2)))
    return fig


# --------------------------------------------------------------------------- #
# Calidad
# --------------------------------------------------------------------------- #
def gauge_score(score: float, titulo: str = "Calidad de los datos") -> go.Figure:
    if score >= 90:
        col = STATUS["bueno"]
    elif score >= 75:
        col = "#1baf7a"
    elif score >= 60:
        col = STATUS["advertencia"]
    elif score >= 40:
        col = STATUS["serio"]
    else:
        col = STATUS["critico"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(font=dict(size=34, color=TEXT_1), valueformat=".0f"),
        domain=dict(x=[0.06, 0.94], y=[0.0, 1.0]),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=GRID,
                      tickfont=dict(size=10, color=TEXT_3)),
            bar=dict(color=col, thickness=0.7),
            bgcolor="#f0efec", borderwidth=0,
            steps=[dict(range=[0, 40], color="#faf3f3"),
                   dict(range=[40, 75], color="#fbf7ee"),
                   dict(range=[75, 100], color="#f0f7f3")],
        ),
    ))
    fig.update_layout(paper_bgcolor=SURFACE, font=FONT, height=210,
                      margin=dict(l=16, r=16, t=36, b=6),
                      title=dict(text=titulo, x=0, xanchor="left",
                                 font=dict(size=14, color=TEXT_2)))
    return fig


def bar_dimensions(score: dict) -> go.Figure:
    dims = ["completitud", "unicidad", "consistencia", "validez"]
    labels = ["Completitud", "Unicidad", "Consistencia", "Validez"]
    vals = [score[d] for d in dims]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=[CAT[0]] * 4, line=dict(width=0)),
        text=[f"{v:.0f}" for v in vals], textposition="outside",
        textfont=dict(size=12, color=TEXT_2),
        hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Dimensiones de calidad", height=250, showlegend=False)
    fig.update_xaxes(range=[0, 112], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def bar_missing(profile: pd.DataFrame, top: int = 25) -> go.Figure:
    d = profile[["columna", "%_nulos", "%_vacios"]].copy()
    d["total"] = d["%_nulos"] + d["%_vacios"]
    d = d[d["total"] > 0].sort_values("total", ascending=True).tail(top)
    if d.empty:
        return _empty("No hay valores faltantes en el conjunto")
    fig = go.Figure()
    fig.add_bar(y=d["columna"], x=d["%_nulos"], orientation="h", name="Nulos",
                marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
                hovertemplate="%{y}<br>Nulos: %{x:.2f}%<extra></extra>")
    if d["%_vacios"].sum() > 0:
        fig.add_bar(y=d["columna"], x=d["%_vacios"], orientation="h",
                    name="Vacíos de texto",
                    marker=dict(color=CAT[1], line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>Vacíos: %{x:.2f}%<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_traces(marker_cornerradius=4)
    return base_layout(fig, "Datos faltantes por columna (%)",
                       height=max(260, 22 * len(d) + 90),
                       showlegend=d["%_vacios"].sum() > 0, xtitle="% de registros")


def heatmap_missing(df: pd.DataFrame, max_rows: int = 400) -> go.Figure:
    step = max(1, len(df) // max_rows)
    m = df.iloc[::step].isna().astype(int)
    if m.to_numpy().sum() == 0:
        return _empty("No hay valores faltantes que mostrar")
    fig = go.Figure(go.Heatmap(
        z=m.T.to_numpy(), y=list(m.columns), colorscale=[[0, "#f5f4f1"], [1, CAT[0]]],
        showscale=False,
        hovertemplate="%{y}<br>fila ~%{x}<br>%{z}<extra></extra>"))
    fig = base_layout(fig, "Mapa de faltantes (1 = dato ausente)",
                      height=max(260, 17 * m.shape[1] + 90),
                      xtitle="registros (muestreados)")
    fig.update_yaxes(showgrid=False)
    return fig


def bar_alerts(profile: pd.DataFrame) -> go.Figure:
    todas = []
    for a in profile["alerta"]:
        if a and a != "OK":
            todas += [x.strip() for x in a.split("|")]
    if not todas:
        return _empty("Sin alertas de calidad")
    vc = pd.Series(todas).value_counts().sort_values()
    fig = go.Figure(go.Bar(
        x=vc.to_numpy(), y=list(vc.index), orientation="h",
        marker=dict(color=CAT[1]), text=vc.to_numpy(), textposition="outside",
        textfont=dict(size=12, color=TEXT_2),
        hovertemplate="%{y}: %{x} columnas<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Alertas detectadas", height=max(220, 30 * len(vc) + 80),
                      showlegend=False, xtitle="nº de columnas")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, vc.to_numpy())


# --------------------------------------------------------------------------- #
# Distribuciones
# --------------------------------------------------------------------------- #
def histogram(df: pd.DataFrame, col: str, color_by: str | None = None,
              nbins: int = 40) -> go.Figure:
    fig = go.Figure()
    if color_by and color_by in df.columns and df[color_by].nunique() <= 8:
        for i, (g, sub) in enumerate(df.groupby(df[color_by].astype(str))):
            fig.add_histogram(x=sub[col], name=str(g), nbinsx=nbins,
                              marker=dict(color=color_for(i),
                                          line=dict(color=SURFACE, width=1)),
                              opacity=0.85,
                              hovertemplate=f"{g}<br>%{{x}}: %{{y}}<extra></extra>")
        fig.update_layout(barmode="overlay")
        show_leg = True
    else:
        fig.add_histogram(x=df[col], nbinsx=nbins,
                          marker=dict(color=CAT[0], line=dict(color=SURFACE, width=1)),
                          hovertemplate="%{x}: %{y} registros<extra></extra>",
                          name=col)
        show_leg = False
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if not s.empty:
        fig.add_vline(x=float(s.mean()), line=dict(color=TEXT_2, width=2, dash="dot"),
                      annotation_text="media",
                      annotation_font=dict(size=11, color=TEXT_2))
    return base_layout(fig, f"Distribución de {col}", showlegend=show_leg,
                       xtitle=col, ytitle="frecuencia")


def box_by_group(df: pd.DataFrame, value: str, group: str | None = None) -> go.Figure:
    fig = go.Figure()
    if group and group in df.columns:
        cats = [str(c) for c in df[group].astype(str).value_counts().head(8).index]
        for i, g in enumerate(cats):
            sub = df[df[group].astype(str) == g]
            fig.add_box(y=sub[value], name=g, marker=dict(color=color_for(i)),
                        line=dict(width=2), boxmean=True, fillcolor="rgba(0,0,0,0)")
        show_leg = len(cats) > 1
        title = f"{value} por {group}"
    else:
        fig.add_box(y=df[value], name=value, marker=dict(color=CAT[0]),
                    line=dict(width=2), boxmean=True, fillcolor="rgba(0,0,0,0)")
        show_leg = False
        title = f"Dispersión de {value}"
    return base_layout(fig, title, showlegend=show_leg, ytitle=value)


def bar_categories(vc: pd.DataFrame, col_cat: str = "categoria",
                   col_val: str = "frecuencia", titulo: str = "") -> go.Figure:
    d = vc.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[col_val], y=d[col_cat].astype(str), orientation="h",
        marker=dict(color=CAT[0]), text=d[col_val], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, titulo, height=max(240, 24 * len(d) + 90), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, d[col_val])


def pareto(vc: pd.DataFrame) -> go.Figure:
    """Barras de frecuencia (el acumulado va en su propio gráfico: sin doble eje)."""
    fig = go.Figure(go.Bar(
        x=vc["categoria"].astype(str), y=vc["frecuencia"],
        marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
        hovertemplate="%{x}: %{y}<extra></extra>", name="frecuencia"))
    fig.update_traces(marker_cornerradius=4)
    return base_layout(fig, "Frecuencia por categoría", showlegend=False,
                       ytitle="registros")


def line_cumulative(vc: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=vc["categoria"].astype(str), y=vc["%_acumulado"], mode="lines+markers",
        line=dict(color=CAT[1], width=2), marker=dict(size=8, color=CAT[1]),
        hovertemplate="%{x}: %{y:.1f}% acumulado<extra></extra>", name="acumulado"))
    fig.add_hline(y=80, line=dict(color=TEXT_3, width=1, dash="dot"),
                  annotation_text="80%", annotation_font=dict(size=11, color=TEXT_2))
    return base_layout(fig, "Porcentaje acumulado", height=280, showlegend=False,
                       ytitle="% acumulado")


# --------------------------------------------------------------------------- #
# Correlaciones
# --------------------------------------------------------------------------- #
def heatmap_corr(corr: pd.DataFrame, titulo: str = "Matriz de correlación",
                 zmin: float = -1, zmax: float = 1) -> go.Figure:
    if corr.empty:
        return _empty("Se necesitan al menos dos variables numéricas")
    n = len(corr)
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index),
        colorscale=DIVERGING, zmid=0, zmin=zmin, zmax=zmax,
        text=corr.round(2).to_numpy(),
        texttemplate="%{text}" if n <= 14 else None,
        textfont=dict(size=10),
        colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10, color=TEXT_2),
                      len=0.85),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>"))
    fig = base_layout(fig, titulo, height=max(340, 28 * n + 130))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return fig


def bar_target_relation(rel: pd.DataFrame, top: int = 18) -> go.Figure:
    if rel is None or rel.empty:
        return _empty("Sin relaciones calculables")
    d = rel.head(top).iloc[::-1]
    colores = [CAT[0] if v >= 0 else CAT[7] for v in d["valor"]]
    fig = go.Figure(go.Bar(
        x=d["valor"].abs(), y=d["variable"], orientation="h",
        marker=dict(color=colores),
        customdata=np.stack([d["metrica"], d["valor"], d["fuerza"]], axis=-1),
        text=[f"{v:.3f}" for v in d["valor"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{y}<br>%{customdata[0]} = %{customdata[1]:.3f}"
                      "<br>%{customdata[2]}<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Fuerza de relación con la variable objetivo",
                      height=max(280, 24 * len(d) + 100), showlegend=False,
                      xtitle="magnitud de la asociación (0–1)")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, d["valor"].abs())


def scatter_xy(df: pd.DataFrame, x: str, y: str, color_by: str | None = None,
               trend: bool = True) -> go.Figure:
    fig = go.Figure()
    if color_by and color_by in df.columns and df[color_by].nunique() <= 8:
        for i, (g, sub) in enumerate(df.groupby(df[color_by].astype(str))):
            fig.add_scatter(x=sub[x], y=sub[y], mode="markers", name=str(g),
                            marker=dict(color=color_for(i), size=7, opacity=0.7,
                                        symbol=symbol_for(i),
                                        line=dict(color=SURFACE, width=1)),
                            hovertemplate=f"{g}<br>{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>")
        show_leg = True
    else:
        fig.add_scatter(x=df[x], y=df[y], mode="markers", name=y,
                        marker=dict(color=CAT[0], size=7, opacity=0.65,
                                    line=dict(color=SURFACE, width=1)),
                        hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y}}<extra></extra>")
        show_leg = False
    if trend:
        d = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) > 2:
            try:
                b, a = np.polyfit(d[x], d[y], 1)
                xs = np.linspace(d[x].min(), d[x].max(), 60)
                fig.add_scatter(x=xs, y=a + b * xs, mode="lines", name="tendencia",
                                line=dict(color=TEXT_2, width=2, dash="dash"),
                                hoverinfo="skip")
                show_leg = True
            except Exception:
                pass
    return base_layout(fig, f"{y} vs {x}", showlegend=show_leg, xtitle=x, ytitle=y)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def bar_leaderboard(lb: pd.DataFrame, metric: str) -> go.Figure:
    d = lb[lb["estado"] == "ok"].copy()
    if metric not in d.columns or d.empty:
        return _empty("Métrica no disponible")
    d = d.sort_values(metric, ascending=True)
    colores = [CAT[2] if i == len(d) - 1 else CAT[0] for i in range(len(d))]
    fig = go.Figure(go.Bar(
        x=d[metric], y=d["modelo"], orientation="h",
        marker=dict(color=colores),
        text=[f"{v:.4f}" for v in d[metric]], textposition="outside",
        textfont=dict(size=12, color=TEXT_2),
        hovertemplate="%{y}<br>" + metric + ": %{x:.4f}<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, f"Comparación de modelos · {metric}",
                      height=max(260, 34 * len(d) + 100), showlegend=False,
                      xtitle=metric)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, d[metric])


def grouped_metrics(lb: pd.DataFrame, metrics: list[str]) -> go.Figure:
    d = lb[lb["estado"] == "ok"]
    metrics = [m for m in metrics if m in d.columns][:8]
    if d.empty or not metrics:
        return _empty("Sin métricas para comparar")
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_bar(x=d["modelo"], y=d[m], name=m,
                    marker=dict(color=color_for(i), line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{x}<br>" + m + ": %{y:.4f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_traces(marker_cornerradius=3)
    fig = base_layout(fig, "Métricas por modelo", height=420, showlegend=True)
    fig.update_xaxes(tickangle=-25)
    return fig


def radar_models(lb: pd.DataFrame, metrics: list[str], top: int = 5) -> go.Figure:
    d = lb[lb["estado"] == "ok"].head(top)
    metrics = [m for m in metrics if m in d.columns]
    if d.empty or len(metrics) < 3:
        return _empty("Se necesitan al menos 3 métricas comparables")
    norm = d[metrics].copy()
    for m in metrics:
        lo, hi = norm[m].min(), norm[m].max()
        norm[m] = (norm[m] - lo) / (hi - lo) if hi > lo else 1.0
    fig = go.Figure()
    for i, (_, row) in enumerate(d.iterrows()):
        vals = norm.iloc[i][metrics].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metrics + [metrics[0]], name=row["modelo"],
            line=dict(color=color_for(i), width=2), fill="none",
            marker=dict(size=8, symbol=symbol_for(i)),
            hovertemplate="%{theta}: %{r:.2f} (normalizado)<extra>" + row["modelo"] + "</extra>"))
    fig.update_layout(
        polar=dict(bgcolor=SURFACE,
                   radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID,
                                   tickfont=dict(size=10, color=TEXT_3)),
                   angularaxis=dict(gridcolor=GRID,
                                    tickfont=dict(size=11, color=TEXT_2))),
        paper_bgcolor=SURFACE, font=FONT, height=430,
        margin=dict(l=60, r=60, t=60, b=40),
        title=dict(text="Perfil comparativo (métricas normalizadas 0–1)", x=0,
                   xanchor="left", font=dict(size=15, color=TEXT_1)),
        legend=dict(orientation="h", y=-0.08, x=0, font=dict(size=12, color=TEXT_2)))
    return fig


def heatmap_confusion(cm: np.ndarray, labels: list[str],
                      normalizar: bool = False) -> go.Figure:
    z = cm.astype(float)
    if normalizar:
        z = z / np.clip(z.sum(axis=1, keepdims=True), 1, None) * 100
        texto = [[f"{v:.1f}%" for v in fila] for fila in z]
        titulo = "Matriz de confusión (% por clase real)"
    else:
        texto = [[f"{int(v):,}" for v in fila] for fila in z]
        titulo = "Matriz de confusión (conteos)"
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"Pred: {l}" for l in labels], y=[f"Real: {l}" for l in labels],
        colorscale=SEQ_BLUE, text=texto, texttemplate="%{text}",
        textfont=dict(size=14),
        colorbar=dict(thickness=10, outlinewidth=0,
                      tickfont=dict(size=10, color=TEXT_2), len=0.85),
        hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"))
    fig = base_layout(fig, titulo, height=max(320, 60 * len(labels) + 140))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def lines_roc(curvas: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar",
                    line=dict(color=TEXT_3, width=1, dash="dot"), hoverinfo="skip")
    for i, c in enumerate(curvas):
        fig.add_scatter(x=c["fpr"], y=c["tpr"], mode="lines",
                        name=f"{c['clase']} (AUC {c['auc']:.3f})",
                        line=dict(color=color_for(i), width=2),
                        hovertemplate="FPR %{x:.3f} · TPR %{y:.3f}<extra></extra>")
    return base_layout(fig, "Curva ROC", xtitle="tasa de falsos positivos",
                       ytitle="tasa de verdaderos positivos")


def lines_pr(curvas: list[dict], baseline: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, c in enumerate(curvas):
        fig.add_scatter(x=c["recall"], y=c["precision"], mode="lines",
                        name=f"{c['clase']} (AP {c['ap']:.3f})",
                        line=dict(color=color_for(i), width=2),
                        hovertemplate="Recall %{x:.3f} · Precisión %{y:.3f}<extra></extra>")
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=TEXT_3, width=1, dash="dot"),
                      annotation_text="prevalencia",
                      annotation_font=dict(size=11, color=TEXT_2))
    return base_layout(fig, "Curva Precisión–Recall", xtitle="recall", ytitle="precisión")


def lines_threshold(th: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(["accuracy", "precision", "recall", "f1"]):
        fig.add_scatter(x=th["umbral"], y=th[col], mode="lines", name=col.capitalize(),
                        line=dict(color=color_for(i), width=2),
                        hovertemplate="umbral %{x:.2f}<br>" + col + ": %{y:.3f}<extra></extra>")
    best = th.loc[th["f1"].idxmax()]
    fig.add_vline(x=float(best["umbral"]), line=dict(color=TEXT_2, width=2, dash="dash"),
                  annotation_text=f"F1 máx @ {best['umbral']:.2f}",
                  annotation_font=dict(size=11, color=TEXT_2))
    return base_layout(fig, "Métricas según el umbral de decisión",
                       xtitle="umbral de probabilidad", ytitle="valor de la métrica")


def scatter_pred_vs_real(res: pd.DataFrame) -> go.Figure:
    lo = float(min(res["real"].min(), res["predicho"].min()))
    hi = float(max(res["real"].max(), res["predicho"].max()))
    fig = go.Figure()
    fig.add_scatter(x=res["real"], y=res["predicho"], mode="markers", name="predicciones",
                    marker=dict(color=CAT[0], size=7, opacity=0.6,
                                line=dict(color=SURFACE, width=1)),
                    hovertemplate="real %{x:,.2f}<br>predicho %{y:,.2f}<extra></extra>")
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="predicción perfecta",
                    line=dict(color=TEXT_2, width=2, dash="dash"), hoverinfo="skip")
    return base_layout(fig, "Valores reales vs. predichos", xtitle="real",
                       ytitle="predicho")


def scatter_residuals(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res["predicho"], y=res["residuo"], mode="markers", name="residuo",
                    marker=dict(color=CAT[0], size=7, opacity=0.6,
                                line=dict(color=SURFACE, width=1)),
                    hovertemplate="predicho %{x:,.2f}<br>residuo %{y:,.2f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=TEXT_2, width=2, dash="dash"))
    return base_layout(fig, "Residuos vs. valores predichos", showlegend=False,
                       xtitle="predicho", ytitle="residuo (real − predicho)")


def hist_residuals(res: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=res["residuo"], nbinsx=45,
        marker=dict(color=CAT[0], line=dict(color=SURFACE, width=1)),
        hovertemplate="residuo %{x}<br>%{y} casos<extra></extra>", name="residuos"))
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=2, dash="dash"))
    return base_layout(fig, "Distribución de los residuos", showlegend=False,
                       xtitle="residuo", ytitle="frecuencia")


def lines_learning_curve(lc: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=lc["n_muestras"], y=lc["entrenamiento"], mode="lines+markers",
                    name="Entrenamiento", line=dict(color=CAT[0], width=2),
                    marker=dict(size=8, symbol="circle"),
                    hovertemplate="n=%{x:,}<br>train: %{y:.3f}<extra></extra>")
    fig.add_scatter(x=lc["n_muestras"], y=lc["validacion"], mode="lines+markers",
                    name="Validación", line=dict(color=CAT[1], width=2),
                    marker=dict(size=8, symbol="square"),
                    hovertemplate="n=%{x:,}<br>validación: %{y:.3f}<extra></extra>")
    return base_layout(fig, "Curva de aprendizaje", xtitle="nº de muestras de entrenamiento",
                       ytitle="desempeño")


# --------------------------------------------------------------------------- #
# Importancia
# --------------------------------------------------------------------------- #
def bar_importance(imp: pd.DataFrame, top: int = 20, titulo: str = "",
                   col: str = "importancia_%") -> go.Figure:
    if imp is None or imp.empty:
        return _empty("Este modelo no expone importancias")
    d = imp.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[col], y=d["variable"], orientation="h",
        marker=dict(color=CAT[0]),
        text=[f"{v:.1f}%" if col.endswith("%") else f"{v:.4f}" for v in d[col]],
        textposition="outside", textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, titulo or "Importancia de las variables",
                      height=max(280, 24 * len(d) + 100), showlegend=False,
                      xtitle="importancia relativa")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, d[col])


def bar_coefficients(coef: pd.DataFrame, top: int = 18) -> go.Figure:
    if coef is None or coef.empty:
        return _empty("El modelo no es lineal: no hay coeficientes")
    d = coef.head(top).iloc[::-1]
    colores = [CAT[0] if v > 0 else CAT[7] for v in d["coeficiente"]]
    fig = go.Figure(go.Bar(
        x=d["coeficiente"], y=d["feature"], orientation="h",
        marker=dict(color=colores),
        text=[f"{v:+.3f}" for v in d["coeficiente"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.update_traces(marker_cornerradius=4)
    fig.add_vline(x=0, line=dict(color=TEXT_2, width=1))
    fig = base_layout(fig, "Coeficientes con signo (dirección del efecto)",
                      height=max(280, 24 * len(d) + 100), showlegend=False,
                      xtitle="← disminuye   |   aumenta →")
    lim = float(d["coeficiente"].abs().max()) * 1.30 or 1.0
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[-lim, lim])
    fig.update_yaxes(showgrid=False)
    return fig


def shap_beeswarm(sv: np.ndarray, Xt: pd.DataFrame, names: list[str],
                  top: int = 15) -> go.Figure:
    orden = np.argsort(np.abs(sv).mean(axis=0))[-top:]
    fig = go.Figure()
    for j in orden:
        vals = Xt.iloc[:, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(vals, 2), np.nanpercentile(vals, 98)
        norm = np.clip((vals - lo) / ((hi - lo) or 1), 0, 1)
        jitter = np.random.default_rng(0).normal(0, 0.10, len(vals))
        fig.add_scatter(
            x=sv[:, j], y=np.full(len(vals), names[j], dtype=object),
            mode="markers",
            marker=dict(color=norm, colorscale=SEQ_BLUE, size=6, opacity=0.7,
                        line=dict(width=0),
                        colorbar=dict(title=dict(text="valor de la variable",
                                                 font=dict(size=10, color=TEXT_2),
                                                 side="right"),
                                      thickness=10, outlinewidth=0, len=0.8,
                                      tickvals=[0, 1], ticktext=["bajo", "alto"],
                                      tickfont=dict(size=10, color=TEXT_2))),
            customdata=vals, showlegend=False,
            hovertemplate="%{y}<br>SHAP %{x:.4f}<br>valor %{customdata:.3f}<extra></extra>")
        fig.data[-1].y = np.array([names[j]] * len(vals), dtype=object)
        fig.data[-1].update(y=[names[j]] * len(vals))
    fig.add_vline(x=0, line=dict(color=TEXT_3, width=1))
    return base_layout(fig, "SHAP: efecto de cada variable sobre cada predicción",
                       height=max(320, 30 * len(orden) + 120), showlegend=False,
                       xtitle="valor SHAP (← empuja hacia abajo | empuja hacia arriba →)")


def line_pdp(pdp: pd.DataFrame, feature: str) -> go.Figure:
    if pdp is None or pdp.empty:
        return _empty("No se pudo calcular la dependencia parcial")
    fig = go.Figure(go.Scatter(
        x=pdp[feature], y=pdp["prediccion_promedio"], mode="lines+markers",
        line=dict(color=CAT[0], width=2), marker=dict(size=7),
        hovertemplate=f"{feature}: %{{x}}<br>predicción: %{{y:.4f}}<extra></extra>",
        name=feature))
    return base_layout(fig, f"Efecto marginal de {feature} (dependencia parcial)",
                       showlegend=False, xtitle=feature, ytitle="predicción promedio")


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #
def line_elbow(ks: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ks["k"], y=ks["inercia"], mode="lines+markers",
        line=dict(color=CAT[0], width=2), marker=dict(size=9),
        hovertemplate="k=%{x}<br>inercia %{y:,.0f}<extra></extra>", name="inercia"))
    codo = ks[ks["recomendado"].str.contains("codo", na=False)]
    if not codo.empty:
        k = int(codo.iloc[0]["k"])
        fig.add_vline(x=k, line=dict(color=CAT[1], width=2, dash="dash"),
                      annotation_text=f"codo: k={k}",
                      annotation_font=dict(size=11, color=CAT[1]))
    return base_layout(fig, "Método del codo (inercia)", height=320, showlegend=False,
                       xtitle="número de clusters (k)", ytitle="inercia")


def line_silhouette_sweep(ks: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ks["k"], y=ks["silueta"], mode="lines+markers",
        line=dict(color=CAT[2], width=2), marker=dict(size=9, symbol="square"),
        hovertemplate="k=%{x}<br>silueta %{y:.3f}<extra></extra>", name="silueta"))
    best = ks.loc[ks["silueta"].idxmax()]
    fig.add_vline(x=int(best["k"]), line=dict(color=CAT[1], width=2, dash="dash"),
                  annotation_text=f"mejor: k={int(best['k'])}",
                  annotation_font=dict(size=11, color=CAT[1]))
    return base_layout(fig, "Coeficiente de silueta por k", height=320, showlegend=False,
                       xtitle="número de clusters (k)", ytitle="silueta (−1 a 1)")


def line_index_sweep(ks: pd.DataFrame, col: str, titulo: str,
                     mejor: str = "max") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ks["k"], y=ks[col], mode="lines+markers",
        line=dict(color=CAT[6], width=2), marker=dict(size=9, symbol="diamond"),
        hovertemplate="k=%{x}<br>%{y:.3f}<extra></extra>", name=col))
    idx = ks[col].idxmax() if mejor == "max" else ks[col].idxmin()
    fig.add_vline(x=int(ks.loc[idx, "k"]), line=dict(color=CAT[1], width=2, dash="dash"),
                  annotation_text=f"mejor: k={int(ks.loc[idx,'k'])}",
                  annotation_font=dict(size=11, color=CAT[1]))
    return base_layout(fig, titulo, height=320, showlegend=False,
                       xtitle="número de clusters (k)", ytitle=col)


def line_kdistance(curve: pd.DataFrame, eps: float | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=curve["punto"], y=curve["distancia"], mode="lines",
        line=dict(color=CAT[0], width=2),
        hovertemplate="punto %{x}<br>distancia %{y:.3f}<extra></extra>", name="k-distancia"))
    if eps:
        fig.add_hline(y=eps, line=dict(color=CAT[1], width=2, dash="dash"),
                      annotation_text=f"eps sugerido = {eps:.3f}",
                      annotation_font=dict(size=11, color=CAT[1]))
    return base_layout(fig, "Curva k-distancia (el codo sugiere el eps de DBSCAN)",
                       height=320, showlegend=False,
                       xtitle="puntos ordenados", ytitle="distancia al k-ésimo vecino")


def scatter_clusters(emb: pd.DataFrame, labels: np.ndarray,
                     nombres: dict[int, str] | None = None,
                     titulo: str = "Mapa de segmentos",
                     hover_df: pd.DataFrame | None = None) -> go.Figure:
    """Dispersión 2D con color + símbolo por cluster (identidad nunca solo por color)."""
    fig = go.Figure()
    etiquetas = sorted(set(int(l) for l in labels))
    for i, lab in enumerate(etiquetas):
        m = labels == lab
        es_ruido = lab == -1
        nombre = ("Ruido / atípicos" if es_ruido
                  else (nombres or {}).get(lab, f"Cluster {lab}"))
        fig.add_scatter(
            x=emb.iloc[:, 0][m], y=emb.iloc[:, 1][m], mode="markers", name=nombre,
            marker=dict(color=TEXT_3 if es_ruido else color_for(i if not es_ruido else 0),
                        size=6 if es_ruido else 8,
                        symbol="x" if es_ruido else symbol_for(i),
                        opacity=0.45 if es_ruido else 0.8,
                        line=dict(color=SURFACE, width=1)),
            hovertemplate=f"{nombre}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>")
        # etiqueta directa en el centroide, con fondo para que se lea sobre los puntos
        if not es_ruido and m.sum() > 0:
            fig.add_annotation(
                x=float(emb.iloc[:, 0][m].mean()), y=float(emb.iloc[:, 1][m].mean()),
                text=f"<b>{lab}</b>", showarrow=False,
                font=dict(size=14, color=TEXT_1),
                bgcolor="rgba(252,252,251,0.88)", bordercolor=GRID, borderwidth=1,
                borderpad=3)
    return base_layout(fig, titulo, height=520,
                       xtitle=emb.columns[0], ytitle=emb.columns[1])


def scatter3d_clusters(emb: pd.DataFrame, labels: np.ndarray,
                       nombres: dict[int, str] | None = None) -> go.Figure:
    fig = go.Figure()
    for i, lab in enumerate(sorted(set(int(l) for l in labels))):
        m = labels == lab
        es_ruido = lab == -1
        nombre = ("Ruido / atípicos" if es_ruido
                  else (nombres or {}).get(lab, f"Cluster {lab}"))
        fig.add_trace(go.Scatter3d(
            x=emb.iloc[:, 0][m], y=emb.iloc[:, 1][m], z=emb.iloc[:, 2][m],
            mode="markers", name=nombre,
            marker=dict(color=TEXT_3 if es_ruido else color_for(i), size=3.5,
                        symbol="x" if es_ruido else "circle",
                        opacity=0.45 if es_ruido else 0.8),
            hovertemplate=f"{nombre}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=SURFACE, font=FONT, height=600,
        margin=dict(l=0, r=0, t=46, b=0),
        title=dict(text="Mapa de segmentos en 3D", x=0, xanchor="left",
                   font=dict(size=15, color=TEXT_1)),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=12, color=TEXT_2)),
        scene=dict(xaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, title=emb.columns[0]),
                   yaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, title=emb.columns[1]),
                   zaxis=dict(backgroundcolor=SURFACE, gridcolor=GRID, title=emb.columns[2])))
    return fig


def bar_cluster_sizes(sizes: pd.DataFrame) -> go.Figure:
    d = sizes.copy()
    colores = [TEXT_3 if c == -1 else color_for(i) for i, c in enumerate(d["cluster"])]
    fig = go.Figure(go.Bar(
        x=d["etiqueta"], y=d["n_registros"], marker=dict(color=colores),
        text=[f"{n:,}<br>{p}%" for n, p in zip(d["n_registros"], d["%_del_total"])],
        textposition="outside", textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{x}<br>%{y:,} registros<extra></extra>", name="tamaño"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Tamaño de cada segmento", height=340, showlegend=False,
                      ytitle="registros")
    return _headroom_y(fig, d["n_registros"], 1.22)


def heatmap_cluster_profile(profile: pd.DataFrame) -> go.Figure:
    num = profile[profile["tipo"] == "numérica"]
    if num.empty:
        return _empty("Sin variables numéricas para perfilar")
    piv = num.pivot_table(index="variable", columns="cluster", values="z_diferencia")
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[f"Cluster {c}" if c != -1 else "Ruido" for c in piv.columns],
        y=list(piv.index), colorscale=DIVERGING, zmid=0,
        text=piv.round(2).to_numpy(), texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title=dict(text="desvío<br>(z)", font=dict(size=10, color=TEXT_2)),
                      thickness=10, outlinewidth=0, len=0.85,
                      tickfont=dict(size=10, color=TEXT_2)),
        hovertemplate="%{y} · %{x}<br>z = %{z:.2f}<extra></extra>"))
    fig = base_layout(fig, "Perfil de segmentos (desviación respecto al promedio global)",
                      height=max(320, 26 * len(piv) + 140))
    fig.update_yaxes(showgrid=False)
    return fig


def silhouette_plot(sil: pd.DataFrame) -> go.Figure:
    if sil is None or sil.empty:
        return _empty("Silueta no disponible")
    fig = go.Figure()
    y0 = 0
    for i, (cl, sub) in enumerate(sil.groupby("cluster")):
        vals = np.sort(sub["silueta"].to_numpy())
        ys = np.arange(y0, y0 + len(vals))
        fig.add_bar(x=vals, y=ys, orientation="h", name=f"Cluster {cl}",
                    marker=dict(color=color_for(i), line=dict(width=0)),
                    hovertemplate=f"Cluster {cl}<br>silueta %{{x:.3f}}<extra></extra>")
        y0 += len(vals) + 12
    media = float(sil["silueta"].mean())
    fig.add_vline(x=media, line=dict(color=TEXT_2, width=2, dash="dash"),
                  annotation_text=f"media {media:.3f}",
                  annotation_font=dict(size=11, color=TEXT_2))
    fig = base_layout(fig, "Silueta por registro (valores negativos = mal asignado)",
                      height=480, xtitle="coeficiente de silueta")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_layout(bargap=0)
    return fig


def dendrogram(Z, max_leaves: int = 40) -> go.Figure:
    try:
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    except Exception:
        return _empty("SciPy no disponible para el dendrograma")
    d = scipy_dendro(Z, no_plot=True, truncate_mode="lastp", p=max_leaves)
    fig = go.Figure()
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=CAT[0], width=1.5),
                        showlegend=False, hoverinfo="skip")
    fig = base_layout(fig, "Dendrograma (agrupamiento jerárquico)", height=420,
                      showlegend=False, xtitle="grupos de registros",
                      ytitle="distancia de fusión")
    fig.update_xaxes(showticklabels=False)
    return fig


def bar_target_by_cluster(tab: pd.DataFrame, col: str) -> go.Figure:
    if tab is None or tab.empty or col not in tab.columns:
        return _empty("Sin cruce disponible")
    d = tab.copy()
    fig = go.Figure(go.Bar(
        x=[f"Cluster {c}" if c != -1 else "Ruido" for c in d["cluster"]], y=d[col],
        marker=dict(color=[TEXT_3 if c == -1 else color_for(i)
                           for i, c in enumerate(d["cluster"])]),
        text=[f"{v:,.2f}" for v in d[col]], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{x}<br>" + col + ": %{y:,.3f}<extra></extra>", name=col))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, f"{col} por segmento", height=340, showlegend=False,
                      ytitle=col)
    return _headroom_y(fig, d[col], 1.22)


# --------------------------------------------------------------------------- #
# Prescriptivo
# --------------------------------------------------------------------------- #
def line_sensitivity(sens: pd.DataFrame, feature: str, ycol: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=sens[feature], y=sens[ycol], mode="lines+markers",
        line=dict(color=CAT[0], width=2), marker=dict(size=9),
        hovertemplate=f"{feature}: %{{x}}<br>{ycol}: %{{y:.4f}}<extra></extra>",
        name=ycol))
    return base_layout(fig, f"Sensibilidad del resultado a {feature}",
                       xtitle=feature, ytitle=ycol)


def bar_scenarios(opt: pd.DataFrame, ycol: str, levers: list[str],
                  top: int = 12) -> go.Figure:
    d = opt.head(top).iloc[::-1].copy()
    etiquetas = d[levers].astype(str).agg(" · ".join, axis=1)
    fig = go.Figure(go.Bar(
        x=d[ycol], y=etiquetas, orientation="h",
        marker=dict(color=[CAT[2] if i == len(d) - 1 else CAT[0] for i in range(len(d))]),
        text=[f"{v:.4f}" for v in d[ycol]], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{y}<br>" + ycol + ": %{x:.4f}<extra></extra>", name="escenario"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Mejores escenarios simulados",
                      height=max(300, 28 * len(d) + 110), showlegend=False, xtitle=ycol)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return _headroom_x(fig, d[ycol])


def bar_probabilities(probs: dict) -> go.Figure:
    d = pd.Series(probs).sort_values()
    fig = go.Figure(go.Bar(
        x=d.to_numpy(), y=[str(i) for i in d.index], orientation="h",
        marker=dict(color=[CAT[2] if i == len(d) - 1 else CAT[0] for i in range(len(d))]),
        text=[f"{v:.1%}" for v in d], textposition="outside",
        textfont=dict(size=12, color=TEXT_2),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>", name="probabilidad"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Probabilidad por clase", height=max(200, 40 * len(d) + 90),
                      showlegend=False)
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def line_timeseries(ts: pd.DataFrame, date_col: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=ts[date_col], y=ts["total"], mode="lines+markers",
        line=dict(color=CAT[0], width=2), marker=dict(size=6),
        hovertemplate="%{x|%b %Y}<br>total %{y:,.2f}<extra></extra>", name="total"))
    return base_layout(fig, "Evolución en el tiempo", showlegend=False, ytitle="total")


# --------------------------------------------------------------------------- #
def _empty(mensaje: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=mensaje, showarrow=False,
                       font=dict(size=13, color=TEXT_3), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=200,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# --------------------------------------------------------------------------- #
# Clustering — vistas adicionales
# --------------------------------------------------------------------------- #
def dendrogram_cut(Z, corte: float, k: int, max_leaves: int = 60) -> go.Figure:
    """Dendrograma con las ramas coloreadas por grupo y la línea de corte en k."""
    try:
        from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    except Exception:
        return _empty("SciPy no disponible para el dendrograma")
    d = scipy_dendro(Z, no_plot=True, truncate_mode="lastp", p=max_leaves,
                     color_threshold=corte, above_threshold_color="gray")
    # scipy nombra los colores C1, C2...: los mapeamos a la paleta fija
    mapa, orden = {}, 0
    fig = go.Figure()
    for xs, ys, c in zip(d["icoord"], d["dcoord"], d["color_list"]):
        if c == "gray":
            col = TEXT_3
        else:
            if c not in mapa:
                mapa[c] = color_for(orden)
                orden += 1
            col = mapa[c]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=1.8),
                        showlegend=False,
                        hovertemplate="distancia de fusión %{y:.2f}<extra></extra>")
    fig.add_hline(y=corte, line=dict(color=CAT[7], width=2, dash="dash"),
                  annotation_text=f"corte → {k} grupos",
                  annotation_position="top right",
                  annotation_font=dict(size=12, color=CAT[7]))
    fig = base_layout(fig, "Árbol jerárquico (dendrograma)", height=460,
                      showlegend=False, xtitle="registros agrupados (hojas truncadas)",
                      ytitle="distancia de fusión")
    fig.update_xaxes(showticklabels=False)
    return fig


def bar_scree(var: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=var["componente"], y=var["varianza_%"],
        marker=dict(color=CAT[0], line=dict(color=SURFACE, width=2)),
        text=[f"{v:.1f}%" for v in var["varianza_%"]], textposition="outside",
        textfont=dict(size=11, color=TEXT_2),
        hovertemplate="%{x}: %{y:.2f}% de la varianza<extra></extra>", name="varianza"))
    fig.update_traces(marker_cornerradius=4)
    fig = base_layout(fig, "Varianza explicada por componente", height=320,
                      showlegend=False, ytitle="% de varianza")
    return _headroom_y(fig, var["varianza_%"], 1.2)


def line_cumvar(var: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=var["componente"], y=var["acumulada_%"], mode="lines+markers",
        line=dict(color=CAT[1], width=2), marker=dict(size=8),
        hovertemplate="%{x}: %{y:.1f}% acumulado<extra></extra>", name="acumulada"))
    for umbral in (80, 90):
        fig.add_hline(y=umbral, line=dict(color=TEXT_3, width=1, dash="dot"),
                      annotation_text=f"{umbral}%",
                      annotation_font=dict(size=11, color=TEXT_2))
    fig = base_layout(fig, "Varianza acumulada", height=320, showlegend=False,
                      ytitle="% acumulado")
    fig.update_yaxes(range=[0, 105])
    return fig


def heatmap_loadings(load: pd.DataFrame, n_comp: int = 3, top: int = 20) -> go.Figure:
    cols = [c for c in load.columns if c.startswith("PC")][:n_comp]
    d = load.head(top).set_index("variable")[cols]
    lim = float(np.nanmax(np.abs(d.to_numpy()))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=d.to_numpy(), x=cols, y=list(d.index), colorscale=DIVERGING, zmid=0,
        zmin=-lim, zmax=lim, text=d.round(2).to_numpy(), texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(thickness=10, outlinewidth=0, len=0.85,
                      tickfont=dict(size=10, color=TEXT_2)),
        hovertemplate="%{y} → %{x}<br>carga %{z:.3f}<extra></extra>"))
    fig = base_layout(fig, "Qué variables forman cada componente (cargas)",
                      height=max(320, 24 * len(d) + 120))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def scatter_clusters_fast(emb: pd.DataFrame, labels: np.ndarray, titulo: str,
                          height: int = 430) -> go.Figure:
    """Dispersión con WebGL (rápida para miles de puntos), color + símbolo por grupo."""
    fig = go.Figure()
    etiquetas = sorted(set(int(l) for l in labels))
    i_color = 0
    for lab in etiquetas:
        m = labels == lab
        ruido = lab == -1
        nombre = "Ruido" if ruido else f"Cluster {lab}"
        if ruido:
            col, sym, op, size = TEXT_3, "x", 0.45, 5
        else:
            col, sym, op, size = color_for(i_color), symbol_for(i_color), 0.78, 6
            i_color += 1
        fig.add_trace(go.Scattergl(
            x=emb.iloc[:, 0].to_numpy()[m], y=emb.iloc[:, 1].to_numpy()[m],
            mode="markers", name=f"{nombre} ({int(m.sum()):,})",
            marker=dict(color=col, symbol=sym, size=size, opacity=op,
                        line=dict(color=SURFACE, width=0.5)),
            hovertemplate=f"{nombre}<br>%{{x:.2f}}, %{{y:.2f}}<extra></extra>"))
        if not ruido and m.sum() > 0:
            fig.add_annotation(
                x=float(emb.iloc[:, 0].to_numpy()[m].mean()),
                y=float(emb.iloc[:, 1].to_numpy()[m].mean()),
                text=f"<b>{lab}</b>", showarrow=False, font=dict(size=13, color=TEXT_1),
                bgcolor="rgba(252,252,251,0.88)", bordercolor=GRID, borderwidth=1,
                borderpad=3)
    fig = base_layout(fig, titulo, height=height,
                      xtitle=emb.columns[0], ytitle=emb.columns[1])
    fig.update_layout(legend=dict(y=-0.18, yanchor="top", orientation="h"))
    fig.update_layout(margin=dict(b=70))
    return fig


def scatter_plain(emb: pd.DataFrame, titulo: str, height: int = 430) -> go.Figure:
    """Proyección sin etiquetas: muestra la forma natural de los datos."""
    fig = go.Figure(go.Scattergl(
        x=emb.iloc[:, 0], y=emb.iloc[:, 1], mode="markers", name="registros",
        marker=dict(color=CAT[0], size=5, opacity=0.55, line=dict(width=0)),
        hovertemplate="%{x:.2f}, %{y:.2f}<extra></extra>"))
    return base_layout(fig, titulo, height=height, showlegend=False,
                       xtitle=emb.columns[0], ytitle=emb.columns[1])


def heatmap_dbscan_grid(grid: pd.DataFrame, valor: str = "silueta",
                        rec: dict | None = None) -> go.Figure:
    """Mapa eps × min_samples: silueta (secuencial) con el punto recomendado."""
    if grid is None or grid.empty:
        return _empty("Sin resultados de la rejilla")
    piv = grid.pivot_table(index="min_samples", columns="eps", values=valor)
    grupos = grid.pivot_table(index="min_samples", columns="eps", values="clusters")
    ruido = grid.pivot_table(index="min_samples", columns="eps", values="%_ruido")
    custom = np.dstack([grupos.to_numpy(), ruido.to_numpy()])
    titulo = ("Silueta (sin ruido) por combinación" if valor == "silueta"
              else "% de ruido por combinación")
    fig = go.Figure(go.Heatmap(
        z=piv.to_numpy(), x=[f"{e:.2f}" for e in piv.columns],
        y=[str(m) for m in piv.index], colorscale=SEQ_BLUE,
        customdata=custom, hoverongaps=False,
        colorbar=dict(thickness=10, outlinewidth=0, len=0.85,
                      tickfont=dict(size=10, color=TEXT_2)),
        hovertemplate="eps %{x} · min_samples %{y}<br>" + valor +
                      " %{z:.3f}<br>grupos %{customdata[0]:.0f} · ruido %{customdata[1]:.1f}%"
                      "<extra></extra>"))
    if rec:
        fig.add_scatter(x=[f"{rec['eps']:.2f}"], y=[str(rec["min_samples"])],
                        mode="markers+text", text=["recomendado"], textposition="top center",
                        textfont=dict(size=11, color=CAT[7]),
                        marker=dict(symbol="star", size=16, color=CAT[7],
                                    line=dict(color=SURFACE, width=1.5)),
                        showlegend=False, hoverinfo="skip")
    fig = base_layout(fig, titulo, height=320, xtitle="eps", ytitle="min_samples")
    fig.update_yaxes(showgrid=False, type="category")
    fig.update_xaxes(type="category", tickangle=-45, nticks=18)
    return fig


def heatmap_crosstab(tabla: pd.DataFrame, titulo: str) -> go.Figure:
    """Tabla cruzada cluster × clase real como mapa de calor secuencial."""
    z = tabla.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"real {c}" for c in tabla.columns],
        y=[("ruido" if int(i) == -1 else f"cluster {i}") for i in tabla.index],
        colorscale=SEQ_BLUE, text=z, texttemplate="%{text}", textfont=dict(size=13),
        showscale=False, hovertemplate="%{y} · %{x}: %{z}<extra></extra>"))
    fig = base_layout(fig, titulo, height=max(220, 44 * len(tabla) + 110))
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig
