from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlite3
import pandas as pd
import joblib
import json
from datetime import datetime

app = FastAPI(title="Propensia API")

modelo = joblib.load("modelo_propensia.pkl")
encoders = joblib.load("encoders_propensia.pkl")
columnas = joblib.load("columnas_modelo.pkl")

DB_PATH = "propensia.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

@app.get("/")
def home():
    return {"status": "Propensia API funcionando"}

# 1. CONSULTAR CLIENTE
@app.get("/consultar_cliente/{cliente_id}")
def consultar_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM clientes WHERE cliente_id = ?", conn, params=(cliente_id,))
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return df.to_dict(orient="records")[0]

# 2. CONSULTAR OFERTAS (ranking del modelo, top 5)
@app.get("/consultar_ofertas/{cliente_id}")
def consultar_ofertas(cliente_id: str):
    conn = get_conn()
    cliente_df = pd.read_sql("SELECT * FROM clientes WHERE cliente_id = ?", conn, params=(cliente_id,))
    ofertas_df = pd.read_sql("SELECT * FROM ofertas", conn)
    conn.close()
    if cliente_df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    cliente = cliente_df.iloc[0]

    resultados = []
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

        fila_df = pd.DataFrame([fila])[columnas]
        prob = modelo.predict_proba(fila_df)[0][1]
        resultados.append({
            "oferta_id": oferta['oferta_id'],
            "nombre_oferta": oferta['nombre_oferta'],
            "precio_mensual": float(oferta['precio_mensual']),
            "es_movistar_total": bool(oferta['es_movistar_total']),
            "probabilidad": round(float(prob), 3)
        })

    resultados = sorted(resultados, key=lambda x: -x['probabilidad'])[:5]
    return {"cliente_id": cliente_id, "top_ofertas": resultados}

# 3. REGISTRAR OFERTA (ofrecida / aceptada / rechazada)
class Registro(BaseModel):
    cliente_id: str
    oferta_recomendada: str
    probabilidad: float
    estado: str  # "ofrecida", "aceptada" o "rechazada"
    motivo_rechazo: Optional[str] = None

@app.post("/registrar_oferta")
def registrar_oferta(datos: Registro):
    conn = get_conn()
    conn.execute(
        "INSERT INTO registros (cliente_id, oferta_recomendada, probabilidad, estado, motivo_rechazo, fecha) VALUES (?, ?, ?, ?, ?, ?)",
        (datos.cliente_id, datos.oferta_recomendada, datos.probabilidad, datos.estado, datos.motivo_rechazo, str(datetime.now()))
    )
    conn.commit()
    conn.close()
    return {"status": "registrado"}

# 4. HISTORIAL DE UN CLIENTE (para ver qué se le ofreció antes)
@app.get("/historial_cliente/{cliente_id}")
def historial_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM registros WHERE cliente_id = ? ORDER BY fecha DESC", conn, params=(cliente_id,))
