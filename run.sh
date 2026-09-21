#!/usr/bin/env bash
# Lanzador de AutoML Studio para Linux y macOS.
set -e
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

if [ ! -d ".venv" ]; then
  echo "▸ Creando entorno virtual…"
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.instalado" ]; then
  echo "▸ Instalando dependencias (puede tardar unos minutos la primera vez)…"
  pip install --upgrade pip -q
  if [ -f "requirements-full.txt" ]; then
    pip install -r requirements-full.txt
  else
    pip install -r requirements.txt
  fi
  touch .venv/.instalado
fi

if [ ! -f "sample_data/clientes_churn.csv" ]; then
  echo "▸ Generando datos de ejemplo…"
  python sample_data/generar_datos.py
fi

echo "▸ Abriendo AutoML Studio en http://localhost:8501"
streamlit run app.py
