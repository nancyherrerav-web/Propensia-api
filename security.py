import re
from typing import Tuple

TEMAS_PERMITIDOS = [
    "venta", "ventas", "oferta", "ofertas", "producto", "productos",
    "plan", "planes", "cliente", "clientes", "problema", "problemas",
    "servicio", "servicios", "internet", "datos", "gb", "reclamo",
    "reclamos", "objecion", "objeciones", "contratacion", "contratación",
    "renovacion", "renovación", "renovar", "cambio", "upgrade",
    "mejora", "recomendacion", "recomendación", "recomendar"
]

TEMAS_PROHIBIDOS = [
    "margen", "márgen", "comision", "comisión", "ganancia interna",
    "rentabilidad interna", "costo interno", "coste interno",
    "competencia", "competidor", "claro", "entel", "bitel"
]

PATRONES_SENSIBLES = {
    "dni": re.compile(r"\b(?:DNI|documento|documento de identidad)\s*(?:N°|Nº|numero|número|:)?\s*\d{7,12}\b", re.IGNORECASE),
    "numero_cuenta": re.compile(r"\b(?:cuenta|número de cuenta|numero de cuenta|cuenta bancaria)\s*(?:N°|Nº|numero|número|:)?\s*\d{6,20}\b", re.IGNORECASE),
    "tarjeta": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "telefono": re.compile(r"\b(?:9\d{8}|\+51\s*9\d{8})\b")
}

# Nueva regla: Detección básica de patrones de inyección SQL en cadenas recibidas
PATRON_SQLI = re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|UNION|EXEC)\b|'|\"--|;)", re.IGNORECASE)

def detectar_tema_prohibido(texto: str) -> Tuple[bool, str]:
    texto_normalizado = texto.lower()
    for palabra in TEMAS_PROHIBIDOS:
        if palabra in texto_normalizado:
            return True, palabra
    return False, ""

def validar_entrada(texto: str) -> Tuple[bool, str]:
    if not texto or not texto.strip():
        return False, "La consulta está vacía."

    # Nuevo: Validación contra SQL Injection / caracteres de riesgo
    if PATRON_SQLI.search(texto):
        return False, "Consulta rechazada por caracteres o palabras reservadas no permitidas."

    texto_normalizado = texto.lower()

    prohibido, _ = detectar_tema_prohibido(texto_normalizado)
    if prohibido:
        return False, "Solo puedo ayudarte con consultas relacionadas con la oferta, productos, ventas, clientes, objeciones y problemas de servicio."

    tiene_tema_permitido = any(palabra in texto_normalizado for palabra in TEMAS_PERMITIDOS)
    if not tiene_tema_permitido:
        return False, "Solo puedo ayudarte con consultas relacionadas con la oferta, productos, ventas, clientes, objeciones y problemas de servicio."

    return True, ""

def detectar_datos_sensibles(texto: str):
    encontrados = []
    for nombre, patron in PATRONES_SENSIBLES.items():
        if patron.search(texto):
            encontrados.append(nombre)
    return encontrados

def redactar_datos_sensibles(texto: str) -> str:
    """NUEVO: Reemplaza datos sensibles por etiquetas seguras en lugar de rechazar."""
    texto_limpio = texto
    for nombre, patron in PATRONES_SENSIBLES.items():
        texto_limpio = patron.sub(f"[{nombre.upper()} OCULTO]", texto_limpio)
    return texto_limpio

def filtrar_salida(texto: str) -> Tuple[bool, str]:
    if not texto:
        return False, "No fue posible generar una respuesta."

    prohibido, _ = detectar_tema_prohibido(texto)
    if prohibido:
        return False, "No puedo mostrar esa respuesta porque contiene información interna que no está disponible para el asesor."

    # Aplicar redacción en lugar de fallo total si hay datos sensibles
    texto_seguro = redactar_datos_sensibles(texto)
    return True, texto_seguro
