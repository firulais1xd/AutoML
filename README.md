# AutoML Studio

Plataforma de análisis de datos de punta a punta: **subes un CSV o Excel** y obtienes
diagnóstico de calidad, ETL reproducible, análisis exploratorio, comparación de modelos
supervisados, variables clave, segmentación no supervisada y un plan de acción
descriptivo y prescriptivo.

---

## Instalación

Necesitas Python 3.10 o superior.

```bash
# 1. Descomprime el proyecto y entra en la carpeta
cd automl_studio

# 2. (Recomendado) crea un entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows:  .venv\Scripts\activate

# 3. Instala las dependencias
pip install -r requirements.txt

# 4. Lanza la aplicación
streamlit run app.py
```

Se abre en `http://localhost:8501`.

> **Atajo:** en Linux/macOS puedes usar `bash run.sh`; en Windows, `run.bat`.
> Ambos crean el entorno, instalan lo necesario y arrancan la app.

---

## Publicar en Streamlit Community Cloud

La nube de Streamlit lee el código desde **GitHub**, así que primero hay que subirlo ahí.

### 1. Sube el proyecto a GitHub

Con la app de GitHub Desktop o desde la web de GitHub: crea un repositorio nuevo
(puede ser privado) y sube **toda la carpeta `automl_studio`**. Desde la terminal sería:

```bash
cd automl_studio
git init
git add .
git commit -m "AutoML Studio"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/automl-studio.git
git push -u origin main
```

### 2. Despliega

1. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu
   cuenta de GitHub.
2. **Create app** → **Deploy a public app from a repo**.
3. Rellena:
   - **Repository:** `TU_USUARIO/automl-studio`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Deploy**. La primera construcción tarda entre 5 y 10 minutos instalando las
   librerías; queda publicada en una URL del tipo
   `https://tu-app.streamlit.app`.

No hay que configurar nada más: `requirements.txt` y `.python-version` ya vienen
preparados en el proyecto.

### Qué cambia respecto a tu computador

- **CatBoost y UMAP no se instalan en la nube** (consumen demasiada memoria para el
  plan gratuito). La app los detecta ausentes y simplemente no los ofrece; XGBoost y
  LightGBM cubren lo mismo. Para tenerlos en local, `requirements-full.txt` los incluye
  y los lanzadores `run.sh` / `run.bat` lo usan automáticamente.
- El plan gratuito comparte memoria limitada. Con archivos de más de ~200 000 filas o
  entrenando 10 modelos con optimización de hiperparámetros a la vez, la app puede
  reiniciarse sola. Si te pasa: entrena menos modelos por tanda, desactiva la
  optimización de hiperparámetros, o usa *Muestrear filas* en la pestaña ETL.
- **Los datos que suban los usuarios son visibles para quien tenga el enlace** si la
  app es pública. Para información sensible, marca la app como privada en los ajustes
  de Streamlit Cloud (*Settings → Sharing*) o úsala solo en local.
- Cada vez que hagas `git push`, la app se actualiza sola.

> **Alternativa sin GitHub:** si solo la vas a usar tú, quédate con la ejecución local
> del apartado anterior. Es más rápida, sin límites de memoria y los datos nunca salen
> de tu equipo.

---

## Cómo se usa

1. **Sube tu archivo** en la barra lateral (CSV, XLSX, XLS, TXT, TSV, Parquet o JSON).
   El separador, la codificación, el separador decimal y **la fila del encabezado**
   se detectan solos (funciona con exportes de ERP que traen líneas de título antes
   de la tabla, `1.234,56`, `$ 1.200`, `45%` y fechas `dd/mm/aaaa`). Puedes forzar
   cualquiera de esos valores en *Opciones de lectura*. Si no tienes un archivo a
   mano, marca **Usar datos de ejemplo**.
2. Revisa **Calidad** y aplica la **limpieza automática** en la pestaña ETL.
3. En **Modelos**, elige la variable a predecir y entrena. La app detecta sola si es
   clasificación o regresión.
4. Explora **Variables clave**, **Clusters** e **Insights**.
5. Descarga el modelo, los datos limpios y el reporte en **Exportar**.

---

## Las nueve pestañas

### 📁 Datos
Vista previa, diccionario de variables con el **rol detectado** de cada columna
(numérica, categórica, fecha, booleana, identificador, constante, texto libre) y
observaciones por columna.

### 🔍 Calidad
- **Score 0–100** en cuatro dimensiones: completitud, unicidad, consistencia y validez.
- Perfil por columna: nulos, vacíos de texto, únicos, ceros, negativos, outliers
  (IQR y z-score), asimetría, curtosis, moda, longitudes, espacios sobrantes e
  inconsistencias de mayúsculas.
- **Hallazgos priorizados** (Alta / Media / Baja) con la acción concreta a tomar.
- Mapa de faltantes y **correlación entre patrones de ausencia** (revela faltantes
  sistemáticos, no aleatorios).
- Duplicados exactos, duplicados por llave de negocio y columnas idénticas.

### 🧪 ETL
Receta de transformaciones encadenadas que **siempre parte del archivo original**, así
que puedes eliminar cualquier paso sin perder nada.

18 operaciones disponibles: eliminar/renombrar columnas, normalizar nombres, eliminar
duplicados y filas vacías, convertir tipos, limpiar texto, convertir vacíos en nulos,
imputar (media / mediana / moda / cero / KNN / interpolación / ffill / constante, con
bandera opcional de dato faltante), eliminar filas con nulos, tratar outliers
(winsorizar / anular / eliminar, por IQR, z-score o percentil), filtrar filas, crear
columnas calculadas, extraer componentes de fecha, agrupar categorías infrecuentes,
transformar variables (log, raíz, Yeo-Johnson, z-score, min-max), discretizar y muestrear.

Un botón sugiere la **limpieza automática** a partir del diagnóstico, y la receta se
exporta a **JSON** o a **código Python** reproducible.

### 📊 Análisis
Estadística descriptiva completa, distribuciones, pruebas de normalidad, correlaciones
(Pearson, Spearman, Kendall), asociación entre categóricas (**Cramér's V**), fuerza de
relación de cada variable con el objetivo con su prueba estadística adecuada
(Pearson, Eta/ANOVA o Cramér's V/chi-cuadrado con p-valor), análisis bivariado y
agregación temporal.

### 🤖 Modelos
Detecta automáticamente si el problema es de clasificación (binaria o multiclase) o
de regresión, y entrena en paralelo sobre el mismo split:

| Clasificación | Regresión |
|---|---|
| Regresión Logística, Ridge Classifier | Lineal, Ridge, Lasso, Elastic Net |
| Árbol de Decisión, Random Forest, Extra Trees | Árbol, Random Forest, Extra Trees |
| Gradient Boosting, HistGradientBoosting, AdaBoost | Gradient Boosting, HistGB, AdaBoost |
| **XGBoost, LightGBM, CatBoost** | **XGBoost, LightGBM, CatBoost** |
| SVM, KNN, Naive Bayes, Red Neuronal (MLP) | SVM, KNN, Red Neuronal (MLP) |
| Baseline (clase mayoritaria) | Baseline (media) |

**Métricas de clasificación:** accuracy, accuracy balanceada, precisión, recall, F1,
F1 ponderado, precisión/recall macro, ROC-AUC, PR-AUC, log loss, MCC y kappa de Cohen.

**Métricas de regresión:** R², R² ajustado, RMSE, MAE, MAPE, sMAPE, error absoluto
mediano, varianza explicada y RMSE normalizado.

Además: validación cruzada con media y desviación, **indicador de sobreajuste**
(entrenamiento − prueba), tiempo de entrenamiento, optimización de hiperparámetros por
búsqueda aleatoria, matriz de confusión, reporte por clase, curvas ROC y
Precisión-Recall, **análisis del umbral de decisión**, gráficos de residuos, real vs.
predicho y curva de aprendizaje.

El **Baseline siempre se incluye**: si tu modelo no lo supera con holgura, no aporta
valor, y la app te lo dice.

### ⭐ Variables clave
Cuatro lecturas independientes de la importancia, todas agregadas a las **columnas
originales** (las columnas one-hot de una misma variable se suman):

- **Nativa** del modelo (`feature_importances_` o coeficientes).
- **Por permutación**: cuánto empeora el modelo al desordenar cada variable. Es la
  medida más fiable del aporte real.
- **SHAP**: importancia media y *beeswarm* con el efecto de cada variable en cada
  predicción individual.
- **Ranking de consenso** que combina los métodos anteriores.

Más **coeficientes con signo** (dirección del efecto) y **dependencia parcial (PDP)**:
cómo cambia la predicción al mover una sola variable.

### 🧩 Clusters
Eliges tus **variables clave**, pulsas **Analizar clusters** una vez y recorres siete
secciones. Los parámetros de cada algoritmo se ajustan en vivo, sin recalcular todo.

**Selección de variables clave**
- Botones rápidos: **✨ Sugeridas** (numéricas no redundantes y con más variabilidad;
  excluye identificadores, constantes y la variable objetivo), **🔢 Todas las
  numéricas**, **⭐ Top del modelo** (las 8 más importantes del mejor modelo entrenado)
  y **🧹 Limpiar**. Luego agregas o quitas las que tú consideres clave.
- Tabla que explica **por qué se sugiere o no cada variable**.

**Las siete secciones**

1. **Variables clave** — estadístico de **Hopkins** (¿los datos forman grupos de verdad
   o son aleatorios?), correlación entre las variables elegidas y alerta de pares
   redundantes, que pesarían doble en la distancia.
2. **Codo y silueta** — gráfico del **método del codo** y gráfico de **silueta por k**
   lado a lado, más Calinski-Harabasz y Davies-Bouldin, con el k recomendado marcado.
3. **K-Means** — eliges k; ves silueta, Davies-Bouldin e inercia, el mapa en **PCA y
   t-SNE** lado a lado, tamaño de cada grupo, promedio de cada variable por grupo en
   sus unidades originales y silueta por registro.
4. **DBSCAN** — `eps` sugerido automáticamente con la **curva k-distancia**, y
   `min_samples`; muestra cuántos grupos encontró y cuánto quedó como ruido, con
   consejos si todo cae en ruido o en un solo grupo.
5. **Árbol jerárquico** — **dendrograma** con las ramas coloreadas por grupo y la
   línea de corte; eliges el método de enlace (Ward, completo, promedio, simple) y el
   número de grupos.
6. **PCA y t-SNE** — varianza explicada por componente y acumulada, **cargas** (qué
   variables forman cada eje), la forma natural de los datos sin agrupar, y una
   **cuadrícula comparativa: K-Means, DBSCAN y Jerárquico, cada uno en PCA y en
   t-SNE**. La perplejidad de t-SNE es ajustable.
7. **Comparar y perfilar** — tabla comparativa (puedes sumar Gaussian Mixture,
   HDBSCAN, OPTICS, Birch, Mean Shift o Spectral), nombre y rasgos de cada segmento,
   mapa de calor de desviaciones, cruce con cualquier variable de negocio, acciones
   recomendadas por segmento y descarga de los datos con su etiqueta de grupo.

Todos los mapas usan **la misma muestra de puntos** para que las vistas sean
comparables, y cada grupo lleva color *y símbolo* propios.

### 💡 Insights
- **Resumen ejecutivo** en un párrafo, listo para presentar.
- **Descriptivo**: narrativa automática sobre volumen, calidad, distribuciones,
  variables categóricas, relaciones y variable objetivo.
- **Prescriptivo**: plan de acción priorizado donde cada acción dice *qué hacer*,
  *por qué*, *qué impacto esperar* y *cómo hacerlo*. Detecta fuga de información,
  sobreajuste, modelos que no superan al baseline y variables prescindibles.
- **Simulador what-if**: construye un caso y observa la predicción y su sensibilidad
  al mover una variable.
- **Optimizador de decisiones**: eliges las variables que *sí puedes controlar*
  (precio, plan, descuento, canal…), la app prueba todas las combinaciones y ordena
  los escenarios por resultado esperado frente al escenario actual.

### 📤 Exportar
Datos transformados, perfil de calidad, estadística descriptiva, tabla comparativa de
modelos, predicciones del conjunto de prueba, el **modelo entrenado en `.joblib`** con
el código de uso listo para copiar, y un **reporte ejecutivo completo en Markdown**.

---

## Estructura del proyecto

```
automl_studio/
├── app.py                    Interfaz Streamlit (9 pestañas)
├── requirements.txt
├── run.sh / run.bat          Lanzadores
├── core/
│   ├── io_utils.py           Lectura con detección de separador y codificación
│   ├── schema.py             Inferencia de roles semánticos de cada columna
│   ├── quality.py            Diagnóstico de calidad y score
│   ├── etl.py                Operaciones, receta y pipeline de preprocesamiento
│   ├── eda.py                Estadística, correlaciones y relación con el objetivo
│   ├── models.py             Catálogo de modelos, entrenamiento y métricas
│   ├── importance.py         Importancia nativa, permutación, SHAP y PDP
│   ├── clustering.py         Diez algoritmos, selección de k y perfilado
│   ├── insights.py           Motor descriptivo, prescriptivo y optimizador
│   └── viz.py                Sistema visual y constructores de gráficos
└── sample_data/
    ├── generar_datos.py      Genera los conjuntos de ejemplo
    ├── clientes_churn.csv    Clasificación binaria (con errores a propósito)
    └── precios_inmuebles.xlsx Regresión
```

---

## Usar el núcleo sin la interfaz

Todo el motor funciona como librería, útil para integrarlo en un pipeline:

```python
from core import io_utils, etl, models, importance, clustering

df, meta = io_utils.read_any("mis_datos.xlsx")

# Limpieza sugerida a partir del diagnóstico
receta = etl.auto_recipe(df)
df, bitacora = etl.apply_recipe(df, receta)

# Entrenar y comparar
resultado = models.train_models(
    df, target="abandono",
    selected_models=["Random Forest", "XGBoost", "LightGBM"],
    cv_folds=5,
)
print(resultado.leaderboard)

# Variables clave del mejor modelo
mejor = resultado.results[resultado.best_model]
print(importance.native_importance(mejor.pipeline, resultado.features))

# Segmentación
M, X, pre = clustering.prepare_matrix(df, ["antiguedad_meses", "cargo_mensual"])
print(clustering.k_selection(M, 2, 10))
seg = clustering.run_clustering(M, "KMeans", n_clusters=4)
print(seg.metricas)
```

---

## Notas técnicas

- El preprocesamiento (imputación, escalado, codificación) vive **dentro del pipeline
  de scikit-learn**, así que se ajusta solo con los datos de entrenamiento: no hay
  fuga de información hacia el conjunto de prueba.
- Las columnas detectadas como **identificador** o **constante** se excluyen por
  defecto del modelado, que es la causa más común de un accuracy irrealmente alto.
- Las categóricas de alta cardinalidad usan codificación ordinal en vez de one-hot
  para no reventar la dimensionalidad.
- El modelo exportado incluye el preprocesamiento completo: `modelo.predict(datos_crudos)`
  funciona directamente sobre columnas sin transformar.
- CatBoost, LightGBM, XGBoost, SHAP y UMAP son opcionales: si alguna falta, la app
  simplemente no ofrece ese modelo o esa función, sin fallar.
- Los colores siguen una paleta validada para daltonismo; en los mapas de segmentos la
  identidad de cada grupo se refuerza con un **símbolo distinto** además del color, y
  toda vista de color va acompañada de su tabla de datos.
