@echo off
REM Lanzador de AutoML Studio para Windows.
cd /d "%~dp0"

if not exist ".venv" (
  echo Creando entorno virtual...
  python -m venv .venv
)

call .venv\Scripts\activate.bat

if not exist ".venv\instalado.txt" (
  echo Instalando dependencias, puede tardar unos minutos la primera vez...
  python -m pip install --upgrade pip -q
  if exist "requirements-full.txt" (
    pip install -r requirements-full.txt
  ) else (
    pip install -r requirements.txt
  )
  echo ok> .venv\instalado.txt
)

if not exist "sample_data\clientes_churn.csv" (
  echo Generando datos de ejemplo...
  python sample_data\generar_datos.py
)

echo Abriendo AutoML Studio en http://localhost:8501
streamlit run app.py
pause
