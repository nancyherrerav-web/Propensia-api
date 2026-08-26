from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import sqlite3
import pandas as pd
import joblib
import json
from datetime import datetime

from security import (
    validar_entrada,
    filtrar_salida
)

# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================

app = FastAPI(
    title="Propensia API",
    servers=[{"url": "https://propensia-api.onrender.com"}]
)

# Carga defensiva de artefactos de Machine Learning
try:
    modelo = joblib.load("modelo_propensia.pkl")
    encoders = joblib.load("encoders_propensia.pkl")
    columnas = joblib.load("columnas_modelo.pkl")
except Exception as e:
    modelo, encoders, columnas = None, {}, []
    print(f"Advertencia: No se pudieron cargar los modelos ML: {e}")

DB_PATH = "propensia.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


# ============================================================
# MIDDLEWARE DE SEGURIDAD GLOBAL
# ============================================================

@app.middleware("http")
async def audit_security_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response


# ============================================================
# CAMPOS DE SALIDA PERMITIDOS
# ============================================================

CAMPOS_CLIENTE_PERMITIDOS = [
    "cliente_id", "tipo_cliente", "antiguedad_meses",
    "monto_facturado_prom", "consumo_datos_gb_prom",
    "dias_mora_prom", "meses_moroso", "n_reclamos",
    "elegible_mt", "es_movistar_total", "canal_mas_usado"
]

CAMPOS_HISTORIAL_PERMITIDOS = [
    "cliente_id", "oferta_recomendada", "probabilidad",
    "estado", "motivo_rechazo", "fecha"
]


# ============================================================
# MODELOS DE DATOS (PYDANTIC)
# ============================================================

class ConsultaAsesor(BaseModel):
    pregunta: str


class RespuestaDify(BaseModel):
    respuesta: str


class Registro(BaseModel):
    cliente_id: str
    oferta_recomendada: str
    probabilidad: float
    estado: str
    motivo_rechazo: Optional[str] = None


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


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "status": "Propensia API funcionando",
        "seguridad": "filtros de entrada y salida activos"
    }


# ------------------------------------------------------------
# 1. FILTROS DE SEGURIDAD (INTEGRACIÓN DIFY)
# ------------------------------------------------------------

@app.post("/validar_consulta")
def validar_consulta(datos: ConsultaAsesor):
    permitida, motivo = validar_entrada(datos.pregunta)

    if not permitida:
        return {
            "permitida": False,
            "respuesta": motivo
        }

    return {
        "permitida": True,
        "respuesta": "Consulta permitida"
    }


@app.post("/filtrar_respuesta")
def filtrar_respuesta(datos: RespuestaDify):
    permitida, respuesta_final = filtrar_salida(datos.respuesta)

    return {
        "permitida": permitida,
        "respuesta": respuesta_final
    }


# ------------------------------------------------------------
# 2. CONSULTAR CLIENTE
# ------------------------------------------------------------

@app.get("/consultar_cliente/{cliente_id}")
def consultar_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM clientes WHERE cliente_id = ?",
        conn,
        params=(cliente_id,)
    )
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    cliente = df.iloc[0]
    resultado = {}

    for campo in CAMPOS_CLIENTE_PERMITIDOS:
        if campo in df.columns:
            valor = cliente[campo]
            if pd.isna(valor):
                valor = None
            elif hasattr(valor, "item"):
                valor = valor.item()
            resultado[campo] = valor

    return resultado


# ------------------------------------------------------------
# 3. CONSULTAR OFERTAS (RECOMENDACIÓN ML)
# ------------------------------------------------------------

@app.get("/consultar_ofertas/{cliente_id}")
def consultar_ofertas(cliente_id: str):
    if not modelo:
        raise HTTPException(
            status_code=500,
            detail="Modelo predictivo no inicializado en el servidor."
        )

    conn = get_conn()
    cliente_df = pd.read_sql(
        "SELECT * FROM clientes WHERE cliente_id = ?",
        conn,
        params=(cliente_id,)
    )
    ofertas_df = pd.read_sql("SELECT * FROM ofertas", conn)
    conn.close()

    if cliente_df.empty:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    cliente = cliente_df.iloc[0]
    filas = []
    info_ofertas = []

    for _, oferta in ofertas_df.iterrows():
        fila = {
            "antiguedad_meses": cliente["antiguedad_meses"],
            "monto_facturado_prom": cliente["monto_facturado_prom"],
            "consumo_datos_gb_prom": cliente["consumo_datos_gb_prom"],
            "dias_mora_prom": cliente["dias_mora_prom"],
            "meses_moroso": cliente["meses_moroso"],
            "n_reclamos": cliente["n_reclamos"],
            "precio_mensual": oferta["precio_mensual"],
            "gb_incluidos": oferta["gb_incluidos"],
            "ahorro_pct": oferta["ahorro_pct"],
        }

        for col, enc in encoders.items():
            valor = None
            if col == "tipo_cliente":
                valor = str(cliente["tipo_cliente"])
            elif col == "canal":
                valor = str(cliente["canal_mas_usado"])
            elif col == "tipo_oferta":
                valor = str(oferta["tipo_oferta"])
            elif col == "elegible_mt":
                valor = str(cliente["elegible_mt"])
            elif col == "oferta_es_mt":
                valor = str(oferta["es_movistar_total"])

            if valor in enc.classes_:
                fila[col] = enc.transform([valor])[0]
            else:
                fila[col] = 0

        filas.append(fila)
        info_ofertas.append({
            "oferta_id": oferta["oferta_id"],
            "nombre_oferta": oferta["nombre_oferta"],
            "precio_mensual": float(oferta["precio_mensual"]),
            "es_movistar_total": bool(oferta["es_movistar_total"])
        })

    X_batch = pd.DataFrame(filas)[columnas]
    probs = modelo.predict_proba(X_batch)[:, 1]

    resultados = []
    for i in range(len(info_ofertas)):
        item = info_ofertas[i]
        item["probabilidad"] = round(float(probs[i]), 3)
        resultados.append(item)

    resultados = sorted(resultados, key=lambda x: -x["probabilidad"])[:5]

    return {
        "cliente_id": cliente_id,
        "fuente": "modelo_propensia",
        "top_ofertas": resultados
    }


# ------------------------------------------------------------
# 4. REGISTRAR OFERTA
# ------------------------------------------------------------

@app.post("/registrar_oferta")
def registrar_oferta(datos: Registro):
    estados_permitidos = ["ofrecida", "aceptada", "rechazada"]
    if datos.estado not in estados_permitidos:
        raise HTTPException(
            status_code=400,
            detail="Estado inválido. Debe ser: ofrecida, aceptada o rechazada."
        )

    if datos.probabilidad < 0 or datos.probabilidad > 1:
        raise HTTPException(
            status_code=400,
            detail="La probabilidad debe estar entre 0 y 1."
        )

    conn = get_conn()
    cliente = pd.read_sql(
        "SELECT cliente_id FROM clientes WHERE cliente_id = ?",
        conn,
        params=(datos.cliente_id,)
    )

    if cliente.empty:
        conn.close()
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    conn.execute(
        """
        INSERT INTO registros
        (cliente_id, oferta_recomendada, probabilidad, estado, motivo_rechazo, fecha)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datos.cliente_id,
            datos.oferta_recomendada,
            datos.probabilidad,
            datos.estado,
            datos.motivo_rechazo,
            str(datetime.now())
        )
    )
    conn.commit()
    conn.close()

    return {"status": "registrado"}


# ------------------------------------------------------------
# 5. HISTORIAL DE CLIENTE
# ------------------------------------------------------------

@app.get("/historial_cliente/{cliente_id}")
def historial_cliente(cliente_id: str):
    conn = get_conn()
    df = pd.read_sql(
        """
        SELECT * FROM registros
        WHERE cliente_id = ?
        ORDER BY fecha DESC
        """,
        conn,
        params=(cliente_id,)
    )
    conn.close()

    if df.empty:
        return []

    resultado = []
    for _, fila in df.iterrows():
        item = {}
        for campo in CAMPOS_HISTORIAL_PERMITIDOS:
            if campo in df.columns:
                valor = fila[campo]
                if pd.isna(valor):
                    valor = None
                elif hasattr(valor, "item"):
                    valor = valor.item()
                item[campo] = valor
        resultado.append(item)

    return resultado


# ------------------------------------------------------------
# 6. AGREGAR O ACTUALIZAR CLIENTE
# ------------------------------------------------------------

@app.post("/agregar_o_actualizar_cliente")
def agregar_o_actualizar_cliente(datos: Cliente):
    conn = get_conn()
    conn.execute("DELETE FROM clientes WHERE cliente_id = ?", (datos.cliente_id,))
    conn.execute(
        """
        INSERT INTO clientes
        (
            cliente_id, tipo_cliente, antiguedad_meses, monto_facturado_prom,
            consumo_datos_gb_prom, dias_mora_prom, meses_moroso, n_reclamos,
            elegible_mt, es_movistar_total, canal_mas_usado
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datos.cliente_id, datos.tipo_cliente, datos.antiguedad_meses,
            datos.monto_facturado_prom, datos.consumo_datos_gb_prom,
            datos.dias_mora_prom, datos.meses_moroso, datos.n_reclamos,
            datos.elegible_mt, datos.es_movistar_total, datos.canal_mas_usado
        )
    )
    conn.commit()
    conn.close()

    return {
        "status": "cliente guardado",
        "cliente_id": datos.cliente_id
    }


# ------------------------------------------------------------
# 7. ESTADÍSTICAS GENERALES
# ------------------------------------------------------------

@app.get("/estadisticas_generales")
def estadisticas_generales():
    try:
        with open("estadisticas.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="No se encontró estadisticas.json"
        )
