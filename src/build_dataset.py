"""
build_dataset.py
================
Construye el conjunto de datos limpio de CounselChat utilizado por las
etapas de recuperación, ranking y evaluación.

Se ejecuta una única vez y genera:

    data/counselchat_limpio.parquet
    data/counselchat_limpio.csv

A partir de ese momento, el resto de módulos trabajan sobre estos archivos
sin necesidad de descargar y procesar nuevamente el dataset original.

Uso:
    python src/build_dataset.py
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

CARPETA_DATOS = (
    Path(__file__).resolve().parent.parent / "data"
)


# -------------------------------------------------------------------------
# CARGA
# -------------------------------------------------------------------------

def cargar_datos() -> pd.DataFrame:
    """
    Descarga CounselChat desde Hugging Face y lo convierte en DataFrame.
    """

    dataset = load_dataset("nbertagnolli/counsel-chat")
    df = dataset["train"].to_pandas()

    print(
        f"Dataset original: "
        f"{df.shape[0]} respuestas, "
        f"{df.shape[1]} columnas"
    )

    return df


# -------------------------------------------------------------------------
# LIMPIEZA
# -------------------------------------------------------------------------

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las decisiones de limpieza derivadas del análisis exploratorio.
    """

    df = df.copy()

    # -----------------------------------------------------
    # Eliminar respuestas sin contenido
    # -----------------------------------------------------

    antes = len(df)

    df = df[
        df["answerText"]
        .fillna("")
        .str.strip()
        != ""
    ].copy()

    print(
        "Respuestas eliminadas por texto vacío: "
        f"{antes - len(df)}"
    )

    # -----------------------------------------------------
    # Construir texto consolidado de la pregunta
    # -----------------------------------------------------

    cuerpo = (
        df["questionText"]
        .fillna("")
        .str.strip()
    )

    titulo = (
        df["questionTitle"]
        .fillna("")
        .str.strip()
    )

    df["pregunta_texto"] = cuerpo.mask(
        cuerpo == "",
        titulo,
    )

    # Comprobación de seguridad.
    sin_pregunta = (
        df["pregunta_texto"]
        .fillna("")
        .str.strip()
        == ""
    ).sum()

    if sin_pregunta > 0:
        raise ValueError(
            f"Existen {sin_pregunta} filas sin texto de pregunta."
        )

    # -----------------------------------------------------
    # Longitud de las respuestas
    # -----------------------------------------------------

    df["long_palabras"] = (
        df["answerText"]
        .str.split()
        .str.len()
    )

    return df


# -------------------------------------------------------------------------
# CONJUNTO DE EVALUACIÓN
# -------------------------------------------------------------------------

def marcar_conjunto_evaluacion(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identifica las preguntas que pueden utilizarse en la evaluación principal.

    Una pregunta se incluye cuando:

    1. Tiene al menos dos respuestas válidas.
    2. Existe una única respuesta con el máximo número de upvotes.

    Los upvotes se utilizan únicamente como señal de referencia observada,
    no como medida directa de calidad terapéutica.
    """

    df = df.copy()

    # Número de respuestas válidas por pregunta.
    conteo = df.groupby("questionID").size()

    preguntas_varias = conteo[
        conteo >= 2
    ].index

    # Solo analizamos preguntas con al menos dos candidatas.
    rankeables = df[
        df["questionID"].isin(preguntas_varias)
    ]

    # Número de respuestas que alcanzan el máximo de upvotes
    # dentro de cada pregunta.
    ganador_unico = (
        rankeables
        .groupby("questionID")["upvotes"]
        .apply(
            lambda votos:
            (votos == votos.max()).sum() == 1
        )
    )

    preguntas_eval = ganador_unico[
        ganador_unico
    ].index

    df["en_evaluacion"] = (
        df["questionID"]
        .isin(preguntas_eval)
    )

    print(
        "Preguntas con al menos dos respuestas: "
        f"{len(preguntas_varias)}"
    )

    print(
        "Preguntas del conjunto principal de evaluación: "
        f"{len(preguntas_eval)}"
    )

    return df


# -------------------------------------------------------------------------
# GUARDADO
# -------------------------------------------------------------------------

def guardar(df: pd.DataFrame) -> None:
    """
    Guarda el conjunto procesado en Parquet y CSV.
    """

    columnas = [
        "questionID",
        "questionTitle",
        "pregunta_texto",
        "answerText",
        "topic",
        "upvotes",
        "views",
        "long_palabras",
        "en_evaluacion",
    ]

    CARPETA_DATOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_parquet = (
        CARPETA_DATOS
        / "counselchat_limpio.parquet"
    )

    ruta_csv = (
        CARPETA_DATOS
        / "counselchat_limpio.csv"
    )

    df[columnas].to_parquet(
        ruta_parquet,
        index=False,
    )

    df[columnas].to_csv(
        ruta_csv,
        index=False,
    )

    print(
        f"\nGuardado: {len(df)} respuestas"
    )

    print(f"  -> {ruta_parquet}")
    print(f"  -> {ruta_csv}")


# -------------------------------------------------------------------------
# EJECUCIÓN
# -------------------------------------------------------------------------

def main() -> None:

    df = cargar_datos()

    # IMPORTANTE:
    # primero se limpia el corpus.
    df = limpiar_datos(df)

    # Después se determina qué preguntas pueden evaluarse.
    df = marcar_conjunto_evaluacion(df)

    guardar(df)


if __name__ == "__main__":
    main()