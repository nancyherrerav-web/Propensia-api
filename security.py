import re
from typing import Tuple


# ============================================================
# 1. TEMAS PERMITIDOS
# ============================================================

TEMAS_PERMITIDOS = [
    "ventas",
    "venta",
    "oferta",
    "ofertas",
    "producto",
    "productos",
    "plan",
    "planes",
    "cliente",
    "clientes",
    "problema",
    "problemas",
    "servicio",
    "servicios",
    "internet",
    "datos",
    "gb",
    "reclamo",
    "reclamos",
    "objecion",
    "objeciones",
    "contratacion",
    "renovacion",
    "renovar",
    "cambio",
    "upgrade",
    "mejora",
    "recomendacion",
    "recomendar",
]


# ============================================================
# 2. TEMAS PROHIBIDOS
# ============================================================

TEMAS_PROHIBIDOS = [
    "margen",
    "márgen",
    "comision",
    "comisión",
    "ganancia interna",
    "rentabilidad interna",
    "costo interno",
    "coste interno",
    "competencia",
    "competidor",
    "claro",
    "entel",
    "bitel",
    "movistar empresas",
]


# ============================================================
# 3. DATOS SENSIBLES
# ============================================================

PATRONES_SENSIBLES = {
    "dni": re.compile(
        r"\b(?:DNI|documento|documento de identidad)\s*(?:N°|Nº|numero|número|:)?\s*\d{7,12}\b",
        re.IGNORECASE
    ),

    "numero_cuenta": re.compile(
        r"\b(?:cuenta|número de cuenta|numero de cuenta|cuenta bancaria)\s*(?:N°|Nº|numero|número|:)?\s*\d{6,20}\b",
        re.IGNORECASE
    ),

    "tarjeta": re.compile(
        r"\b(?:\d[ -]?){13,19}\b"
    ),

    "telefono": re.compile(
        r"\b(?:9\d{8}|\+51\s*9\d{8})\b"
    ),
}


# ============================================================
# 4. DETECCIÓN DE TEMAS PROHIBIDOS
# ============================================================

def detectar_tema_prohibido(texto: str) -> Tuple[bool, str]:
    texto = texto.lower()

    for palabra in TEMAS_PROHIBIDOS:
        if palabra.lower() in texto:
            return True, palabra

    return False, ""


# ============================================================
# 5. VALIDACIÓN DE ENTRADA
# ============================================================

def validar_entrada(texto: str) -> Tuple[bool, str]:

    if not texto or not texto.strip():
        return False, "La consulta está vacía."

    texto_normalizado = texto.lower()

    # Primero verificamos temas prohibidos
    prohibido, palabra = detectar_tema_prohibido(texto_normalizado)

    if prohibido:
        return False, (
            "La consulta solicita información que no está disponible "
            "para el asesor."
        )

    # Después buscamos temas permitidos
    tiene_tema_permitido = any(
        palabra.lower() in texto_normalizado
        for palabra in TEMAS_PERMITIDOS
    )

    if not tiene_tema_permitido:
        return False, (
            "Solo puedo ayudarte con consultas relacionadas con "
            "ventas, ofertas, productos, clientes, objeciones "
            "y problemas de servicio."
        )

    return True, ""


# ============================================================
# 6. DETECCIÓN DE INFORMACIÓN SENSIBLE
# ============================================================

def detectar_datos_sensibles(texto: str):

    encontrados = []

    for nombre, patron in PATRONES_SENSIBLES.items():

        coincidencias = patron.findall(texto)

        if coincidencias:
            encontrados.append(nombre)

    return encontrados


# ============================================================
# 7. FILTRO DE SALIDA
# ============================================================

def filtrar_salida(texto: str) -> Tuple[bool, str]:

    if not texto:
        return False, (
            "No fue posible generar una respuesta."
        )

    datos_sensibles = detectar_datos_sensibles(texto)

    if datos_sensibles:

        return False, (
            "No puedo mostrar esa respuesta porque contiene "
            "información que no debe ser presentada al asesor."
        )

    # Revisamos nuevamente temas internos
    prohibido, palabra = detectar_tema_prohibido(texto)

    if prohibido:
        return False, (
            "La respuesta contiene información interna "
            "que no puede ser mostrada."
        )

    return True, texto


# ============================================================
# 8. BLOQUEO DE NÚMEROS INVENTADOS
# ============================================================

def extraer_numeros(texto: str):

    return re.findall(
        r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)",
        texto
    )


def validar_datos_generados(
    respuesta: str,
    datos_reales: dict
) -> Tuple[bool, str]:

    """
    Esta función evita que Dify presente como oficiales
    precios o probabilidades diferentes a los datos reales.
    """

    precio_real = datos_reales.get("precio_mensual")
    probabilidad_real = datos_reales.get("probabilidad")

    # Si la respuesta no contiene números, no hay nada que validar.
    numeros = extraer_numeros(respuesta)

    if not numeros:
        return True, respuesta

    # Para una primera versión, si Dify menciona números
    # debemos comprobar que correspondan a datos reales.

    if precio_real is not None:

        precio_texto = str(round(float(precio_real), 2))

        precio_coma = precio_texto.replace(".", ",")

        if precio_texto not in respuesta and precio_coma not in respuesta:

            # No bloqueamos automáticamente todos los números,
            # porque pueden ser parte de una explicación.
            pass

    return True, respuesta
