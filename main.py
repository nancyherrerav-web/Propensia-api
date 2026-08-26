from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import pandas as pd
import joblib
import json
from datetime import datetime

app = FastAPI(
    title="Propensia API",
    servers=[{"url": "https://propensia-api.onrender.com"}]
)

modelo = joblib.load("modelo_propensia.pkl")
encoders = joblib.load("encoders_propensia.pkl")
columnas = joblib.load("columnas_modelo.pkl")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://usuario:password@host:5432/nombre_db")

def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)

@app.get("/")
def home():
    return {"status": "Propensia API funcionando"}

# 1. CONSULTAR CLIENTE
@app.get("/consultar_cliente/{cliente_id}")
def consultar_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM clientes WHERE cliente_id = %s", conn, params=(cliente_id,))
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return df.to_dict(orient="records")[0]

# 2. CONSULTAR OFERTAS
@app.get("/consultar_ofertas/{cliente_id}")
def consultar_ofertas(cliente_id: str):
    conn = get_conn()
    cliente_df = pd.read_sql("SELECT * FROM clientes WHERE cliente_id = %s", conn, params=(cliente_id,))
    ofertas_df = pd.read_sql("SELECT * FROM ofertas", conn)
    conn.close()
    if cliente_df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    cliente = cliente_df.iloc[0]

    filas = []
    info_ofertas = []
    for _, oferta in ofertas_df.iterrows():
        fila = {
            'antiguedad_meses': cliente['antiguedad_meses'],
            'monto_facturado_prom': cliente['monto_facturado_prom'],
            'consumo_datos_gb_prom': cliente['consumo_datos_gb_prom'],
            'dias_mora_prom': cliente['dias_mora_prom'],
            'meses_moroso': cliente['meses_moroso'],
            'n_reclamos': cliente['n_reclamos'],
            'precio_mensual': oferta['precio_mensual'],
            'gb_incluidos': oferta['gb_incluidos'],
            'ahorro_pct': oferta['ahorro_pct'],
        }
        for col, enc in encoders.items():
            valor = None
            if col == 'tipo_cliente': valor = str(cliente['tipo_cliente'])
            elif col == 'canal': valor = str(cliente['canal_mas_usado'])
            elif col == 'tipo_oferta': valor = str(oferta['tipo_oferta'])
            elif col == 'elegible_mt': valor = str(cliente['elegible_mt'])
            elif col == 'oferta_es_mt': valor = str(oferta['es_movistar_total'])
            fila[col] = enc.transform([valor])[0] if valor in enc.classes_ else 0

        filas.append(fila)
        info_ofertas.append({
            "oferta_id": oferta['oferta_id'],
            "nombre_oferta": oferta['nombre_oferta'],
            "precio_mensual": float(oferta['precio_mensual']),
            "es_movistar_total": bool(oferta['es_movistar_total']),
        })

    X_batch = pd.DataFrame(filas)[columnas]
    probs = modelo.predict_proba(X_batch)[:, 1]

    resultados = []
    for i in range(len(info_ofertas)):
        item = info_ofertas[i]
        item["probabilidad"] = round(float(probs[i]), 3)
        resultados.append(item)

    resultados = sorted(resultados, key=lambda x: -x['probabilidad'])[:5]
    return {"cliente_id": cliente_id, "top_ofertas": resultados}

# 3. REGISTRAR OFERTA
class Registro(BaseModel):
    cliente_id: str
    oferta_recomendada: str
    probabilidad: float
    estado: str
    motivo_rechazo: Optional[str] = None

@app.post("/registrar_oferta")
def registrar_oferta(datos: Registro):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO registros (cliente_id, oferta_recomendada, probabilidad, estado, motivo_rechazo, fecha) VALUES (%s, %s, %s, %s, %s, %s)",
        (datos.cliente_id, datos.oferta_recomendada, datos.probabilidad, datos.estado, datos.motivo_rechazo, str(datetime.now()))
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "registrado"}

# 4. HISTORIAL DE UN CLIENTE
@app.get("/historial_cliente/{cliente_id}")
def historial_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM registros WHERE cliente_id = %s ORDER BY fecha DESC", conn, params=(cliente_id,))
    conn.close()
    return df.to_dict(orient="records")

# 5. AGREGAR O ACTUALIZAR CLIENTE
class Cliente(BaseModel):
    cliente_id: str
    tipo_cliente: str
    antiguedad_meses: int
    monto_facturado_prom: float
    consumo_datos_gb_prom: float
    dias_mora_prom: float
    meses_moroso: int
    n_reclamos: int
    elegible_mt: bool
    es_movistar_total: bool = False
    canal_mas_usado: str

@app.post("/agregar_o_actualizar_cliente")
def agregar_o_actualizar_cliente(datos: Cliente):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM clientes WHERE cliente_id = %s", (datos.cliente_id,))
    cur.execute("""
        INSERT INTO clientes (cliente_id, tipo_cliente, antiguedad_meses, monto_facturado_prom,
        consumo_datos_gb_prom, dias_mora_prom, meses_moroso, n_reclamos, elegible_mt, es_movistar_total, canal_mas_usado)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (datos.cliente_id, datos.tipo_cliente, datos.antiguedad_meses, datos.monto_facturado_prom,
          datos.consumo_datos_gb_prom, datos.dias_mora_prom, datos.meses_moroso, datos.n_reclamos,
          datos.elegible_mt, datos.es_movistar_total, datos.canal_mas_usado))
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "cliente guardado", "cliente_id": datos.cliente_id}

# 6. ESTADÍSTICAS GENERALES
@app.get("/estadisticas_generales")
def estadisticas_generales():
    with open("estadisticas.json", "r", encoding="utf-8") as f:
        return json.load(f)
