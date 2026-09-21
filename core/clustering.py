"""
Segmentación no supervisada: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN,
OPTICS, Agglomerative, Birch, Mean Shift, Spectral y Gaussian Mixture.

Incluye selección de k (codo + silueta + Calinski + Davies-Bouldin),
proyección 2D/3D (PCA, t-SNE, UMAP), perfilado de segmentos y narrativa.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, OPTICS, AgglomerativeClustering,
    Birch, MeanShift, SpectralClustering, estimate_bandwidth,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from .etl import build_preprocessor
from .schema import is_numeric

warnings.filterwarnings("ignore")

try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

RANDOM_STATE = 42


# =========================================================================== #
# Catálogo
# =========================================================================== #
ALGORITHMS = {
    "KMeans": {
        "necesita_k": True,
        "descripcion": "Particiona en k grupos esféricos minimizando la inercia. "
                       "Rápido y el más usado; exige definir k.",
    },
    "MiniBatch KMeans": {
        "necesita_k": True,
        "descripcion": "Variante de KMeans por lotes: ideal para datasets grandes.",
    },
    "Jerárquico (Ward)": {
        "necesita_k": True,
        "descripcion": "Aglomerativo: une los grupos más cercanos paso a paso. "
                       "Permite ver el dendrograma.",
    },
    "Gaussian Mixture": {
        "necesita_k": True,
        "descripcion": "Modelo probabilístico: cada punto tiene probabilidad de "
                       "pertenecer a cada grupo. Admite grupos elípticos.",
    },
    "Birch": {
        "necesita_k": True,
        "descripcion": "Construye un árbol de características; muy eficiente en memoria.",
    },
    "Spectral": {
        "necesita_k": True,
        "descripcion": "Usa el grafo de similitud: detecta grupos no convexos.",
    },
    "DBSCAN": {
        "necesita_k": False,
        "descripcion": "Basado en densidad: descubre el nº de grupos y marca "
                       "outliers como ruido (-1). Requiere eps y min_samples.",
    },
    "HDBSCAN": {
        "necesita_k": False,
        "descripcion": "DBSCAN jerárquico: no necesita eps, maneja densidades "
                       "variables. Recomendado cuando DBSCAN es difícil de calibrar.",
    },
    "OPTICS": {
        "necesita_k": False,
        "descripcion": "Variante de DBSCAN robusta a densidades distintas.",
    },
    "Mean Shift": {
        "necesita_k": False,
        "descripcion": "Busca modas de densidad; estima el nº de grupos solo.",
    },
}

DENSITY_ALGOS = {"DBSCAN", "HDBSCAN", "OPTICS", "Mean Shift"}


@dataclass
class ClusterResult:
    algoritmo: str
    labels: np.ndarray
    n_clusters: int
    n_ruido: int
    metricas: dict = field(default_factory=dict)
    modelo: Any = None
    parametros: dict = field(default_factory=dict)
    embedding: pd.DataFrame | None = None
    perfil: pd.DataFrame | None = None
    tamanos: pd.DataFrame | None = None
    error: str | None = None


# =========================================================================== #
# Preparación de la matriz
# =========================================================================== #
def prepare_matrix(df: pd.DataFrame, features: list[str],
                   scaler: str = "standard", encoder: str = "onehot",
                   max_rows: int | None = None,
                   random_state: int = RANDOM_STATE) -> tuple[np.ndarray, pd.DataFrame, Any]:
    """Escala/codifica las variables y devuelve (matriz, df_usado, preprocesador)."""
    X = df[features].copy()
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)
    pre, _ = build_preprocessor(X, scaler=scaler, encoder=encoder)
    M = pre.fit_transform(X)
    M = np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return M, X, pre


# =========================================================================== #
# Selección de k
# =========================================================================== #
def k_selection(M: np.ndarray, k_min: int = 2, k_max: int = 10,
                algorithm: str = "KMeans",
                random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Barrido de k con inercia (codo), silueta, Calinski-Harabasz y Davies-Bouldin."""
    rows = []
    k_max = int(min(k_max, max(2, len(M) - 1)))
    sample_idx = _sample_idx(len(M))
    for k in range(int(k_min), k_max + 1):
        try:
            model = _build_k_model(algorithm, k, random_state)
            labels = model.fit_predict(M)
            if len(set(labels)) < 2:
                continue
            inertia = float(getattr(model, "inertia_", np.nan))
            if not np.isfinite(inertia):
                inertia = _pseudo_inertia(M, labels)
            rows.append({
                "k": k,
                "inercia": round(inertia, 2),
                "silueta": round(float(silhouette_score(M[sample_idx], labels[sample_idx])), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(M, labels)), 2),
                "davies_bouldin": round(float(davies_bouldin_score(M, labels)), 4),
            })
        except Exception:
            continue
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["delta_inercia_%"] = (-out["inercia"].pct_change() * 100).round(2)
    out["recomendado"] = ""
    best_sil = out.loc[out["silueta"].idxmax(), "k"]
    best_db = out.loc[out["davies_bouldin"].idxmin(), "k"]
    best_ch = out.loc[out["calinski_harabasz"].idxmax(), "k"]
    elbow = _elbow_k(out)
    for k, tag in [(best_sil, "mejor silueta"), (best_db, "mejor Davies-Bouldin"),
                   (best_ch, "mejor Calinski"), (elbow, "codo")]:
        if k is None:
            continue
        m = out["k"] == k
        out.loc[m, "recomendado"] = (out.loc[m, "recomendado"] + " · " + tag).str.strip(" ·")
    return out


def _build_k_model(algorithm: str, k: int, random_state: int):
    if algorithm == "MiniBatch KMeans":
        return MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5)
    if algorithm == "Jerárquico (Ward)":
        return AgglomerativeClustering(n_clusters=k, linkage="ward")
    if algorithm == "Gaussian Mixture":
        return GaussianMixture(n_components=k, random_state=random_state)
    if algorithm == "Birch":
        return Birch(n_clusters=k)
    if algorithm == "Spectral":
        return SpectralClustering(n_clusters=k, random_state=random_state,
                                  assign_labels="kmeans", n_neighbors=10)
    return KMeans(n_clusters=k, random_state=random_state, n_init=10)


def _pseudo_inertia(M: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for lab in set(labels):
        pts = M[labels == lab]
        if len(pts):
            total += float(((pts - pts.mean(axis=0)) ** 2).sum())
    return total


def _elbow_k(df: pd.DataFrame) -> int | None:
    """
    Codo = el k donde la caída de la inercia se frena más bruscamente
    (mayor cambio de pendiente). No depende del rango de k probado, a diferencia
    de los métodos de distancia a la recta.
    """
    if len(df) < 3:
        return None
    x = df["k"].to_numpy(dtype=float)
    y = df["inercia"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        return None
    caidas = y[:-1] - y[1:]                     # cuánto baja al pasar de k a k+1
    frenazo = caidas[:-1] - caidas[1:]          # cuánto se reduce esa caída
    return int(x[1:-1][int(np.argmax(frenazo))])


def _sample_idx(n: int, max_n: int = 5000, seed: int = RANDOM_STATE) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, max_n, replace=False)


# =========================================================================== #
# Ayuda para DBSCAN: curva k-distancia
# =========================================================================== #
def kdistance_curve(M: np.ndarray, min_samples: int = 5) -> pd.DataFrame:
    """Curva de distancias ordenadas: el 'codo' sugiere el eps de DBSCAN."""
    k = min(min_samples, max(2, len(M) - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(M)
    d, _ = nn.kneighbors(M)
    dist = np.sort(d[:, -1])
    return pd.DataFrame({"punto": np.arange(len(dist)), "distancia": dist})


def suggest_eps(M: np.ndarray, min_samples: int = 5) -> float:
    curve = kdistance_curve(M, min_samples)
    y = curve["distancia"].to_numpy()
    if len(y) < 3 or y.max() == y.min():
        return float(np.median(y)) if len(y) else 0.5
    x = np.arange(len(y), dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    idx = int(np.argmax(yn - xn))
    return float(round(y[idx], 4))


# =========================================================================== #
# Ejecución
# =========================================================================== #
def run_clustering(M: np.ndarray, algorithm: str = "KMeans", n_clusters: int = 3,
                   eps: float | None = None, min_samples: int = 5,
                   linkage: str = "ward", min_cluster_size: int = 15,
                   covariance_type: str = "full", quantile: float = 0.2,
                   random_state: int = RANDOM_STATE) -> ClusterResult:
    params: dict[str, Any] = {}
    try:
        if algorithm in ("KMeans", "MiniBatch KMeans", "Birch", "Spectral",
                         "Gaussian Mixture", "Jerárquico (Ward)"):
            if algorithm == "Jerárquico (Ward)":
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                params = {"n_clusters": n_clusters, "linkage": linkage}
            elif algorithm == "Gaussian Mixture":
                model = GaussianMixture(n_components=n_clusters,
                                        covariance_type=covariance_type,
                                        random_state=random_state)
                params = {"n_components": n_clusters, "covariance_type": covariance_type}
            else:
                model = _build_k_model(algorithm, n_clusters, random_state)
                params = {"n_clusters": n_clusters}
            labels = model.fit_predict(M)

        elif algorithm == "DBSCAN":
            e = eps if eps and eps > 0 else suggest_eps(M, min_samples)
            model = DBSCAN(eps=e, min_samples=min_samples, n_jobs=-1)
            labels = model.fit_predict(M)
            params = {"eps": round(float(e), 4), "min_samples": min_samples}

        elif algorithm == "HDBSCAN":
            if not HAS_HDBSCAN:
                raise RuntimeError("HDBSCAN no disponible en esta versión de scikit-learn.")
            model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
            labels = model.fit_predict(M)
            params = {"min_cluster_size": min_cluster_size, "min_samples": min_samples}

        elif algorithm == "OPTICS":
            model = OPTICS(min_samples=min_samples, n_jobs=-1)
            labels = model.fit_predict(M)
            params = {"min_samples": min_samples}

        elif algorithm == "Mean Shift":
            bw = estimate_bandwidth(M, quantile=quantile,
                                    n_samples=min(len(M), 2000), random_state=random_state)
            model = MeanShift(bandwidth=bw if bw and bw > 0 else None, n_jobs=-1)
            labels = model.fit_predict(M)
            params = {"bandwidth": round(float(bw), 4) if bw else "auto"}
        else:
            raise ValueError(f"Algoritmo desconocido: {algorithm}")

        labels = np.asarray(labels)
        uniq = set(labels.tolist())
        n_noise = int((labels == -1).sum())
        n_cl = len(uniq - {-1})

        metrics = cluster_metrics(M, labels)
        return ClusterResult(algoritmo=algorithm, labels=labels, n_clusters=n_cl,
                             n_ruido=n_noise, metricas=metrics, modelo=model,
                             parametros=params)
    except Exception as e:
        return ClusterResult(algoritmo=algorithm, labels=np.array([]), n_clusters=0,
                             n_ruido=0, error=f"{type(e).__name__}: {e}",
                             parametros=params)


def cluster_metrics(M: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels != -1
    lab = labels[mask]
    sub = M[mask]
    out: dict[str, Any] = {
        "n_clusters": int(len(set(lab.tolist()))),
        "n_ruido": int((labels == -1).sum()),
        "%_ruido": round(float((labels == -1).mean() * 100), 2),
    }
    if len(set(lab.tolist())) < 2 or len(sub) < 3:
        out.update({"silueta": None, "calinski_harabasz": None, "davies_bouldin": None})
        return out
    idx = _sample_idx(len(sub))
    try:
        out["silueta"] = round(float(silhouette_score(sub[idx], lab[idx])), 4)
    except Exception:
        out["silueta"] = None
    try:
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(sub, lab)), 2)
    except Exception:
        out["calinski_harabasz"] = None
    try:
        out["davies_bouldin"] = round(float(davies_bouldin_score(sub, lab)), 4)
    except Exception:
        out["davies_bouldin"] = None
    return out


def compare_algorithms(M: np.ndarray, algorithms: list[str], n_clusters: int = 3,
                       **kwargs) -> tuple[pd.DataFrame, dict[str, ClusterResult]]:
    """Corre varios algoritmos y devuelve una tabla comparativa."""
    results: dict[str, ClusterResult] = {}
    rows = []
    for algo in algorithms:
        r = run_clustering(M, algorithm=algo, n_clusters=n_clusters, **kwargs)
        results[algo] = r
        if r.error:
            rows.append({"algoritmo": algo, "estado": "error", "detalle": r.error})
            continue
        rows.append({
            "algoritmo": algo, "estado": "ok",
            "n_clusters": r.n_clusters,
            "n_ruido": r.n_ruido,
            "%_ruido": r.metricas.get("%_ruido"),
            "silueta": r.metricas.get("silueta"),
            "calinski_harabasz": r.metricas.get("calinski_harabasz"),
            "davies_bouldin": r.metricas.get("davies_bouldin"),
            "parametros": str(r.parametros),
        })
    tabla = pd.DataFrame(rows)
    if "silueta" in tabla.columns:
        tabla = tabla.sort_values("silueta", ascending=False, na_position="last")
        tabla = tabla.reset_index(drop=True)
    return tabla, results


# =========================================================================== #
# Proyección para visualizar
# =========================================================================== #
def project(M: np.ndarray, method: str = "PCA", n_components: int = 2,
            random_state: int = RANDOM_STATE,
            perplexity: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Reduce a 2D/3D para graficar los clusters."""
    info: dict[str, Any] = {"metodo": method}
    pedidas = int(n_components)
    # PCA/UMAP no pueden devolver más componentes que variables o registros
    n_components = int(max(1, min(pedidas, M.shape[1], len(M))))
    if method == "t-SNE" and len(M) < 8:
        method = "PCA"          # t-SNE necesita más puntos que la perplejidad
        info["aviso"] = "Muy pocos registros para t-SNE: se muestra PCA."
    if method == "t-SNE":
        per = float(min(max(2.0, min(perplexity, (len(M) - 1) / 3)), len(M) - 1))
        red = TSNE(n_components=min(n_components, 3), random_state=random_state,
                   perplexity=per, init="pca", learning_rate="auto")
        emb = red.fit_transform(M)
        info["perplexity"] = per
    elif method == "UMAP" and HAS_UMAP:
        red = umap.UMAP(n_components=n_components, random_state=random_state)
        emb = red.fit_transform(M)
    else:
        red = PCA(n_components=n_components, random_state=random_state)
        emb = red.fit_transform(M)
        info["varianza_explicada_%"] = [round(v * 100, 2)
                                        for v in red.explained_variance_ratio_]
        info["varianza_acumulada_%"] = round(
            float(red.explained_variance_ratio_.sum() * 100), 2)
        info["loadings"] = red.components_
        info["metodo"] = "PCA"
    emb = np.asarray(emb, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
    # Con una sola variable hay una sola dimensión: se rellena con ceros para
    # que los mapas 2D/3D sigan funcionando.
    if emb.shape[1] < pedidas:
        emb = np.hstack([emb, np.zeros((len(emb), pedidas - emb.shape[1]))])
        info["aviso"] = info.get("aviso") or ("Solo hay una dimensión: el segundo "
                                              "eje es constante.")
    cols = [f"Dim {i+1}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, columns=cols), info


def pca_loadings(M: np.ndarray, feature_names: list[str],
                 n_components: int = 2) -> pd.DataFrame:
    """Contribución de cada variable a los componentes principales."""
    n = min(n_components, M.shape[1])
    p = PCA(n_components=n, random_state=RANDOM_STATE).fit(M)
    df = pd.DataFrame(p.components_.T,
                      columns=[f"PC{i+1}" for i in range(n)],
                      index=feature_names[:M.shape[1]])
    df["peso_total"] = df.abs().sum(axis=1)
    return df.sort_values("peso_total", ascending=False).round(4).reset_index(
        names="variable")


# =========================================================================== #
# Perfilado de segmentos
# =========================================================================== #
def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    s = pd.Series(labels).value_counts().sort_index()
    df = pd.DataFrame({"cluster": s.index, "n_registros": s.values})
    df["%_del_total"] = (df["n_registros"] / df["n_registros"].sum() * 100).round(2)
    df["etiqueta"] = df["cluster"].map(lambda c: "Ruido / outliers" if c == -1
                                       else f"Cluster {c}")
    return df


def profile_clusters(df_orig: pd.DataFrame, labels: np.ndarray,
                     features: list[str]) -> pd.DataFrame:
    """Medias/modas por cluster y su desviación respecto al total."""
    d = df_orig.copy()
    d["_cluster"] = labels
    rows = []
    num = [c for c in features if is_numeric(d[c])]
    cat = [c for c in features if c not in num]

    for c in num:
        g = d.groupby("_cluster")[c].agg(["mean", "median", "std", "count"])
        total_mean = d[c].mean()
        total_std = d[c].std(ddof=0) or 1
        for cl, r in g.iterrows():
            rows.append({
                "cluster": cl, "variable": c, "tipo": "numérica",
                "valor_cluster": round(float(r["mean"]), 4),
                "mediana_cluster": round(float(r["median"]), 4),
                "valor_global": round(float(total_mean), 4),
                "diferencia_%": round(float((r["mean"] - total_mean) /
                                            (abs(total_mean) or 1) * 100), 2),
                "z_diferencia": round(float((r["mean"] - total_mean) / total_std), 3),
                "n": int(r["count"]),
            })

    for c in cat:
        for cl, sub in d.groupby("_cluster"):
            vc = sub[c].astype(str).value_counts(normalize=True)
            if vc.empty:
                continue
            top = vc.index[0]
            share_global = (d[c].astype(str) == top).mean()
            rows.append({
                "cluster": cl, "variable": c, "tipo": "categórica",
                "valor_cluster": f"{top} ({vc.iloc[0]*100:.1f}%)",
                "mediana_cluster": None,
                "valor_global": f"{top} ({share_global*100:.1f}%) global",
                "diferencia_%": round(float((vc.iloc[0] - share_global) * 100), 2),
                "z_diferencia": round(float((vc.iloc[0] - share_global) /
                                            (np.sqrt(share_global * (1 - share_global)) or 1)), 3),
                "n": int(len(sub)),
            })

    return pd.DataFrame(rows)


def cluster_signatures(profile: pd.DataFrame, top_n: int = 4) -> dict[int, list[str]]:
    """Las variables que más distinguen a cada cluster."""
    sig: dict[int, list[str]] = {}
    if profile.empty:
        return sig
    for cl, sub in profile.groupby("cluster"):
        s = sub.reindex(sub["z_diferencia"].abs().sort_values(ascending=False).index)
        frases = []
        for _, r in s.head(top_n).iterrows():
            if r["tipo"] == "numérica":
                direc = "por encima" if r["z_diferencia"] > 0 else "por debajo"
                frases.append(
                    f"{r['variable']}: {r['valor_cluster']:,} ({direc} del promedio "
                    f"global en {abs(r['diferencia_%']):.0f}%)")
            else:
                frases.append(f"{r['variable']}: predomina {r['valor_cluster']}")
        sig[int(cl)] = frases
    return sig


def name_clusters(profile: pd.DataFrame, sizes: pd.DataFrame) -> pd.DataFrame:
    """Nombre descriptivo automático para cada segmento."""
    sig = cluster_signatures(profile, top_n=2)
    rows = []
    for _, r in sizes.iterrows():
        cl = int(r["cluster"])
        if cl == -1:
            nombre = "Atípicos / ruido"
        else:
            frases = sig.get(cl, [])
            partes = [f.split(":")[0] for f in frases[:2]]
            nombre = ("Segmento " + str(cl) +
                      (f" · alto contraste en {' y '.join(partes)}" if partes else ""))
        rows.append({
            "cluster": cl, "nombre_sugerido": nombre,
            "n_registros": int(r["n_registros"]), "%_del_total": r["%_del_total"],
            "caracteristicas": " | ".join(sig.get(cl, [])) or "—",
        })
    return pd.DataFrame(rows)


def silhouette_by_point(M: np.ndarray, labels: np.ndarray) -> pd.DataFrame | None:
    """Silueta individual: permite ver clusters mal definidos."""
    mask = labels != -1
    lab, sub = labels[mask], M[mask]
    if len(set(lab.tolist())) < 2:
        return None
    idx = _sample_idx(len(sub), 4000)
    vals = silhouette_samples(sub[idx], lab[idx])
    return pd.DataFrame({"cluster": lab[idx], "silueta": vals}).sort_values(
        ["cluster", "silueta"]).reset_index(drop=True)


def target_by_cluster(df_orig: pd.DataFrame, labels: np.ndarray,
                      target: str) -> pd.DataFrame | None:
    """Cruce del segmento con una variable de negocio (tasa o promedio)."""
    if target not in df_orig.columns:
        return None
    d = df_orig.copy()
    d["_cluster"] = labels
    if is_numeric(d[target]):
        g = d.groupby("_cluster")[target].agg(
            promedio="mean", mediana="median", minimo="min", maximo="max", n="count")
        g["vs_global_%"] = ((g["promedio"] - d[target].mean()) /
                            (abs(d[target].mean()) or 1) * 100)
        return g.round(3).reset_index(names="cluster")
    tab = pd.crosstab(d["_cluster"], d[target].astype(str), normalize="index") * 100
    tab = tab.round(2)
    tab["n"] = d.groupby("_cluster").size()
    return tab.reset_index(names="cluster")


def dendrogram_data(M: np.ndarray, max_rows: int = 800,
                    method: str = "ward") -> tuple[np.ndarray, np.ndarray] | None:
    """Matriz de enlace para el dendrograma (submuestreada)."""
    try:
        from scipy.cluster.hierarchy import linkage
    except Exception:
        return None
    idx = _sample_idx(len(M), max_rows)
    Z = linkage(M[idx], method=method)
    return Z, idx


def assign_new(result: ClusterResult, M_new: np.ndarray) -> np.ndarray | None:
    """Asigna nuevos registros al segmento más cercano (si el modelo lo permite)."""
    model = result.modelo
    if model is None:
        return None
    if hasattr(model, "predict"):
        try:
            return np.asarray(model.predict(M_new))
        except Exception:
            pass
    return None


# =========================================================================== #
# Selección de variables clave
# =========================================================================== #
LABEL_NAME_RE = re.compile(
    r"^(cultivar|clase|class|target|label|labels|etiqueta|objetivo|y|grupo|group|"
    r"cluster|segmento|segment|categoria|category|especie|species|variedad|"
    r"variety|outcome|resultado|diagnostico|diagnosis)(_?\d+)?$", re.IGNORECASE)


def _nombre_normalizado(c) -> str:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", t).strip("_").lower()


def is_label_like(df: pd.DataFrame, c) -> bool:
    """
    ¿La columna parece una etiqueta real (clase, cultivar, target...)?
    Criterio: nombre de etiqueta y pocos valores distintos. Una columna así no
    debe usarse para agrupar: sería darle la respuesta al algoritmo.
    """
    s = df[c]
    nu = s.nunique(dropna=True)
    if nu < 2 or nu > 20:
        return False
    return bool(LABEL_NAME_RE.search(_nombre_normalizado(c)))


def detect_reference_column(df: pd.DataFrame) -> str | None:
    """Primera columna con pinta de etiqueta real, para validar los clusters."""
    for c in df.columns:
        if is_label_like(df, c):
            return c
    return None


def suggest_features(df: pd.DataFrame, roles: dict, exclude: list[str] | None = None,
                     max_feats: int = 15, corr_thr: float = 0.85,
                     include_categorical: bool = False,
                     drop_redundant: bool = False) -> tuple[list[str], pd.DataFrame]:
    """
    Sugiere variables para segmentar y explica por qué entra o sale cada una.

    Descarta identificadores, constantes, casi constantes, fechas, texto libre,
    columnas con pinta de etiqueta real (cultivar, clase, target…) y códigos
    categóricos numéricos. Las variables muy correlacionadas se señalan; solo se
    quitan si `drop_redundant=True`.
    """
    exclude = set(exclude or [])
    filas, candidatas = [], []

    for c in df.columns:
        r = roles.get(c)
        rol = r.role if r is not None else "?"
        s = df[c]
        nu = s.nunique(dropna=True)
        motivo = None
        if c in exclude:
            motivo = "excluida: es la variable objetivo o la de referencia"
        elif rol == "id":
            motivo = "identificador: agruparía por código, no por comportamiento"
        elif rol == "constante":
            motivo = "constante: no tiene variación"
        elif is_label_like(df, c):
            motivo = ("parece la etiqueta real (clase/grupo): úsala como referencia "
                      "para validar, nunca para agrupar")
        elif rol == "fecha":
            motivo = "fecha: extrae año/mes en ETL si la quieres usar"
        elif rol == "texto":
            motivo = "texto libre o alta cardinalidad"
        elif rol in ("categorica", "booleana") and not is_numeric(s) and not include_categorical:
            motivo = "categórica: actívalas si quieres incluirlas (se codifican one-hot)"
        else:
            top_share = s.value_counts(normalize=True, dropna=True)
            if len(top_share) and top_share.iloc[0] > 0.95:
                motivo = f"casi constante ({top_share.iloc[0]*100:.0f}% es un solo valor)"
            elif s.isna().mean() > 0.5:
                motivo = f"{s.isna().mean()*100:.0f}% de nulos"
        if motivo:
            filas.append({"variable": c, "rol": rol, "sugerida": False, "motivo": motivo})
        else:
            candidatas.append(c)

    num = [c for c in candidatas if is_numeric(df[c])]
    cat = [c for c in candidatas if c not in num]

    def cv(c):
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        m = v.mean()
        if len(v) < 2:
            return 0.0
        return float(abs(v.std() / m)) if m and np.isfinite(m) else float(v.std() or 0)

    # se conserva el orden original del archivo (como harías a mano);
    # la variabilidad solo decide qué queda fuera si hay más de max_feats
    orden_cv = sorted(num, key=cv, reverse=True)
    permitidas = set(orden_cv[:max_feats]) if len(orden_cv) > max_feats else set(orden_cv)

    corr = df[num].corr().abs() if len(num) > 1 else pd.DataFrame()
    elegidas: list[str] = []
    for c in num:
        par = None
        for e in elegidas:
            if not corr.empty and corr.loc[c, e] > corr_thr:
                par = (e, corr.loc[c, e])
                break
        if c not in permitidas:
            filas.append({"variable": c, "rol": roles[c].role, "sugerida": False,
                          "motivo": f"fuera del top {max_feats} por variabilidad"})
        elif par and drop_redundant:
            filas.append({"variable": c, "rol": roles[c].role, "sugerida": False,
                          "motivo": f"redundante con {par[0]} (r = {par[1]:.2f})"})
        else:
            elegidas.append(c)
            nota = (f"numérica · ojo: muy correlacionada con {par[0]} (r = {par[1]:.2f})"
                    if par else f"numérica, variabilidad relativa {cv(c):.2f}")
            filas.append({"variable": c, "rol": roles[c].role, "sugerida": True,
                          "motivo": nota})

    for c in cat:
        if include_categorical:
            elegidas.append(c)
            filas.append({"variable": c, "rol": roles[c].role, "sugerida": True,
                          "motivo": f"categórica con {df[c].nunique()} niveles"})
        else:
            filas.append({"variable": c, "rol": roles[c].role, "sugerida": False,
                          "motivo": "categórica (no incluida)"})

    tabla = pd.DataFrame(filas)
    orden = {c: i for i, c in enumerate(df.columns)}
    tabla["_o"] = tabla["variable"].map(orden)
    tabla = (tabla.sort_values(["sugerida", "_o"], ascending=[False, True])
             .drop(columns="_o").reset_index(drop=True))
    return elegidas, tabla


# =========================================================================== #
# Validación externa contra una etiqueta real
# =========================================================================== #
def external_validation(labels: np.ndarray, reference) -> dict:
    """
    Compara los clusters contra una etiqueta real que NO se usó para agrupar.

    - ARI (Adjusted Rand Index): 1 = coincidencia perfecta, 0 = azar.
    - NMI: información mutua normalizada (0 a 1).
    - Tasa de acierto: mejor emparejamiento cluster → clase (método húngaro);
      el ruido de DBSCAN cuenta como una categoría propia.
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ref = pd.Series(reference).astype(str).to_numpy()
    lab = np.asarray(labels)
    tabla = pd.crosstab(pd.Series(lab, name="cluster"), pd.Series(ref, name="referencia"))
    filas, cols = linear_sum_assignment(-tabla.to_numpy())
    aciertos = int(tabla.to_numpy()[filas, cols].sum())
    emparejamiento = {int(tabla.index[f]): str(tabla.columns[c]) for f, c in zip(filas, cols)}
    return {
        "ari": float(adjusted_rand_score(ref, lab)),
        "nmi": float(normalized_mutual_info_score(ref, lab)),
        "acierto": aciertos / max(len(lab), 1),
        "tabla": tabla,
        "emparejamiento": emparejamiento,
    }


# =========================================================================== #
# DBSCAN: búsqueda en rejilla de eps × min_samples
# =========================================================================== #
def dbscan_grid(M: np.ndarray, eps_values=None, min_samples_values=(3, 4, 5, 6, 8, 10),
                max_rows: int = 3000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Prueba combinaciones de eps y min_samples y mide grupos, ruido y silueta
    (la silueta se calcula sin los puntos de ruido).
    """
    idx = _sample_idx(len(M), max_rows, random_state)
    sub = M[idx]
    if eps_values is None:
        kd = kdistance_curve(sub, 5)["distancia"].to_numpy()
        lo = float(np.percentile(kd, 2) * 0.7)
        hi = float(np.percentile(kd, 99) * 1.6)
        if not np.isfinite(lo) or hi <= lo:
            lo, hi = 0.1, 2.0
        eps_values = np.round(np.linspace(lo, hi, 36), 3)
    filas = []
    for eps in eps_values:
        for ms in min_samples_values:
            if ms >= len(sub):
                continue
            lab = DBSCAN(eps=float(eps), min_samples=int(ms), n_jobs=-1).fit_predict(sub)
            n_cl = len(set(lab.tolist()) - {-1})
            ruido = int((lab == -1).sum())
            sil, sil_con = None, None
            mask = lab != -1
            if n_cl >= 2 and mask.sum() > n_cl:
                try:
                    sil = float(silhouette_score(sub[mask], lab[mask]))
                except Exception:
                    sil = None
            # variante "ruido como un grupo más" (la que se calcula a veces en clase)
            if len(set(lab.tolist())) >= 2 and len(set(lab.tolist())) < len(sub):
                try:
                    sil_con = float(silhouette_score(sub, lab))
                except Exception:
                    sil_con = None
            filas.append({"eps": round(float(eps), 3), "min_samples": int(ms),
                          "clusters": n_cl, "ruido": ruido,
                          "%_ruido": round(ruido / len(sub) * 100, 1),
                          "silueta": round(sil, 4) if sil is not None else None,
                          "silueta_con_ruido": round(sil_con, 4) if sil_con is not None else None})
    return pd.DataFrame(filas)


def recommend_dbscan(grid: pd.DataFrame, max_noise_pct: float = 20.0,
                     min_noise_pct: float = 0.5) -> dict | None:
    """
    Mejor silueta entre las combinaciones con al menos 2 grupos y un ruido
    razonable (ni cero —DBSCAN no estaría detectando atípicos— ni excesivo).
    """
    if grid is None or grid.empty:
        return None
    ok = grid[(grid["clusters"] >= 2) & grid["silueta"].notna()]
    rango = ok[(ok["%_ruido"] <= max_noise_pct) & (ok["%_ruido"] >= min_noise_pct)]
    base = rango if not rango.empty else ok[ok["%_ruido"] <= max_noise_pct]
    if base.empty:
        base = ok
    if base.empty:
        return None
    mejor = base.sort_values(["silueta", "%_ruido"], ascending=[False, True]).iloc[0]
    return {"eps": float(mejor["eps"]), "min_samples": int(mejor["min_samples"]),
            "clusters": int(mejor["clusters"]), "%_ruido": float(mejor["%_ruido"]),
            "silueta": float(mejor["silueta"])}


def dendrogram_jump_k(Z) -> dict:
    """Criterio del salto más grande entre alturas consecutivas de fusión."""
    alturas = np.asarray(Z)[:, 2]
    if len(alturas) < 2:
        return {"k": None}
    saltos = np.diff(alturas)
    i = int(np.argmax(saltos))
    n = len(alturas) + 1
    return {"k": int(n - (i + 1)), "altura_antes": float(alturas[i]),
            "altura_despues": float(alturas[i + 1]), "salto": float(saltos[i])}


def hopkins(M: np.ndarray, sample_frac: float = 0.1, max_m: int = 300,
            random_state: int = RANDOM_STATE) -> float:
    """
    Estadístico de Hopkins: tendencia de los datos a formar grupos.
    ~0.5 = distribución aleatoria (no hay clusters reales); > 0.7 = hay estructura.
    """
    rng = np.random.default_rng(random_state)
    n, d = M.shape
    if n < 20:
        return float("nan")
    m = int(min(max_m, max(10, n * sample_frac)))
    nn = NearestNeighbors(n_neighbors=2).fit(M)
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(M[idx], n_neighbors=2)[0][:, 1]
    lo, hi = M.min(axis=0), M.max(axis=0)
    U = rng.uniform(lo, hi, size=(m, d))
    u = nn.kneighbors(U, n_neighbors=1)[0][:, 0]
    return float(u.sum() / (u.sum() + w.sum()))


def hopkins_label(h: float) -> str:
    if not np.isfinite(h):
        return "no calculable"
    if h >= 0.75:
        return "estructura de grupos clara"
    if h >= 0.6:
        return "estructura de grupos moderada"
    return "los datos parecen casi aleatorios: los clusters pueden ser artificiales"


# =========================================================================== #
# Jerárquico sobre muestra + asignación del resto
# =========================================================================== #
def hierarchical_fit(M: np.ndarray, n_clusters: int = 3, linkage_method: str = "ward",
                     max_rows: int = 4000, random_state: int = RANDOM_STATE) -> dict:
    """
    Ajusta el aglomerativo sobre una muestra (su costo es cuadrático) y asigna el
    resto de registros al centroide más cercano. Devuelve etiquetas completas,
    la matriz de enlace de la muestra y la altura de corte para el dendrograma.
    """
    from scipy.cluster.hierarchy import linkage as sc_linkage, fcluster
    from sklearn.neighbors import NearestCentroid

    idx = _sample_idx(len(M), max_rows, random_state)
    sub = M[idx]
    metric = "euclidean"
    Z = sc_linkage(sub, method=linkage_method, metric=metric)
    lab_sub = fcluster(Z, t=n_clusters, criterion="maxclust") - 1

    if len(idx) < len(M) and len(set(lab_sub)) > 1:
        nc = NearestCentroid().fit(sub, lab_sub)
        labels = nc.predict(M)
        labels[idx] = lab_sub
    else:
        labels = np.full(len(M), -1)
        labels[idx] = lab_sub

    # altura de corte: entre la fusión k-ésima y la (k-1)-ésima desde arriba
    alturas = np.sort(Z[:, 2])
    if n_clusters >= 2 and len(alturas) >= n_clusters:
        corte = float((alturas[-(n_clusters - 1)] + alturas[-n_clusters]) / 2)
    else:
        corte = float(alturas[-1]) if len(alturas) else 0.0

    return {"labels": np.asarray(labels, dtype=int), "Z": Z, "idx_muestra": idx,
            "corte": corte, "metricas": cluster_metrics(M, np.asarray(labels, dtype=int)),
            "n_muestra": len(idx)}


def pca_variance(M: np.ndarray, max_comp: int = 20) -> pd.DataFrame:
    """Varianza explicada por componente (para el gráfico de sedimentación)."""
    n = int(min(max_comp, M.shape[1], M.shape[0]))
    p = PCA(n_components=n, random_state=RANDOM_STATE).fit(M)
    r = p.explained_variance_ratio_ * 100
    return pd.DataFrame({"componente": [f"PC{i+1}" for i in range(n)],
                         "varianza_%": np.round(r, 2),
                         "acumulada_%": np.round(np.cumsum(r), 2)})


def centroids_table(df_orig: pd.DataFrame, labels: np.ndarray,
                    features: list[str]) -> pd.DataFrame:
    """Promedio de cada variable por cluster, en sus unidades originales."""
    d = df_orig[features].copy()
    num = [c for c in features if is_numeric(d[c])]
    if not num:
        return pd.DataFrame()
    d["cluster"] = labels
    g = d.groupby("cluster")[num].mean().round(3)
    g.loc["GLOBAL"] = d[num].mean().round(3)
    g.insert(0, "n", list(d.groupby("cluster").size()) + [len(d)])
    return g.reset_index()
