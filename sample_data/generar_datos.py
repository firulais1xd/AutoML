"""Genera datasets de ejemplo (con problemas de calidad a propósito)."""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def churn_telecom(n: int = 3000) -> pd.DataFrame:
    antiguedad = rng.integers(1, 72, n)
    cargo_mensual = np.round(rng.normal(70, 25, n).clip(18, 160), 2)
    soporte = rng.poisson(1.2, n)
    plan = rng.choice(["Básico", "Estándar", "Premium"], n, p=[0.45, 0.35, 0.20])
    contrato = rng.choice(["Mes a mes", "Un año", "Dos años"], n, p=[0.55, 0.28, 0.17])
    pago = rng.choice(["Débito automático", "Tarjeta", "Transferencia", "Efectivo"], n)
    ciudad = rng.choice(["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga"],
                        n, p=[0.40, 0.22, 0.18, 0.12, 0.08])
    fibra = rng.choice([0, 1], n, p=[0.45, 0.55])
    satisfaccion = np.clip(rng.normal(7, 2, n), 1, 10).round(1)

    logit = (
        -1.1
        - 0.045 * antiguedad
        + 0.022 * cargo_mensual
        + 0.38 * soporte
        - 0.30 * satisfaccion
        + np.where(contrato == "Mes a mes", 1.3, np.where(contrato == "Un año", 0.2, -0.9))
        + np.where(pago == "Efectivo", 0.55, 0)
        - 0.25 * fibra
        + rng.normal(0, 0.55, n)
    )
    churn = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

    df = pd.DataFrame({
        "cliente_id": [f"CLI-{i:06d}" for i in range(1, n + 1)],
        "ciudad": ciudad,
        "plan": plan,
        "tipo_contrato": contrato,
        "metodo_pago": pago,
        "antiguedad_meses": antiguedad,
        "cargo_mensual": cargo_mensual,
        "cargo_total": np.round(cargo_mensual * antiguedad * rng.uniform(0.9, 1.1, n), 2),
        "llamadas_soporte": soporte,
        "tiene_fibra": fibra,
        "satisfaccion": satisfaccion,
        "fecha_alta": pd.to_datetime("2019-01-01") +
                      pd.to_timedelta(rng.integers(0, 1800, n), unit="D"),
        "pais": "Colombia",                      # constante a propósito
        "abandono": np.where(churn == 1, "Sí", "No"),
    })

    # Problemas de calidad intencionales
    for col, frac in [("satisfaccion", 0.09), ("cargo_total", 0.05), ("plan", 0.04)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan
    idx = rng.choice(n, 60, replace=False)
    df.loc[idx, "ciudad"] = " bogotá "            # espacios + minúsculas
    df.loc[rng.choice(n, 40, replace=False), "metodo_pago"] = ""
    df = pd.concat([df, df.sample(70, random_state=1)], ignore_index=True)  # duplicados
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def ventas_inmuebles(n: int = 2500) -> pd.DataFrame:
    area = np.round(rng.gamma(6, 15, n) + 35, 1)
    habitaciones = np.clip((area / 38 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    banos = np.clip((habitaciones * 0.7 + rng.normal(0, 0.5, n)).round(), 1, 4).astype(int)
    antiguedad = rng.integers(0, 45, n)
    estrato = rng.choice([2, 3, 4, 5, 6], n, p=[0.15, 0.30, 0.28, 0.17, 0.10])
    zona = rng.choice(["Norte", "Centro", "Sur", "Occidente", "Chapinero"], n)
    parqueadero = rng.choice([0, 1, 2], n, p=[0.35, 0.50, 0.15])
    piso = rng.integers(1, 20, n)

    precio = (
        95 * area
        + 9000 * habitaciones
        + 12000 * banos
        - 1400 * antiguedad
        + 26000 * estrato
        + 15000 * parqueadero
        + 900 * piso
        + np.where(zona == "Chapinero", 60000, np.where(zona == "Norte", 45000, 0))
        + rng.normal(0, 22000, n)
    ).clip(60000, None).round(-3)

    df = pd.DataFrame({
        "inmueble_id": [f"INM-{i:05d}" for i in range(1, n + 1)],
        "zona": zona, "estrato": estrato, "area_m2": area,
        "habitaciones": habitaciones, "banos": banos,
        "antiguedad_anios": antiguedad, "parqueaderos": parqueadero, "piso": piso,
        "tiene_ascensor": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "fecha_publicacion": pd.to_datetime("2023-01-01") +
                             pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
        "precio_millones": np.round(precio / 1000, 2),
    })
    idx = rng.choice(n, int(n * 0.06), replace=False)
    df.loc[idx, "area_m2"] = np.nan
    return df


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    churn_telecom().to_csv(here / "clientes_churn.csv", index=False)
    ventas_inmuebles().to_excel(here / "precios_inmuebles.xlsx", index=False)
    print("Datasets generados en", here)
