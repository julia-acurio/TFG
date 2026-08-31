"""
ranking.py
==========
Etapa de RANKING del sistema.

retrieval.py recupera varias preguntas similares a la consulta del usuario.
Este módulo reúne todas las respuestas asociadas a esas preguntas y utiliza
un Cross-Encoder para ordenarlas según su relevancia respecto a la consulta.

Los upvotes NO se utilizan para construir el ranking. Se reservan
exclusivamente para la evaluación.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

MODELO_RANKING = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RUTA_DATOS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "counselchat_limpio.parquet"
)


# -------------------------------------------------------------------------
# CARGA DEL CORPUS
# -------------------------------------------------------------------------

def cargar_corpus() -> pd.DataFrame:
    """Carga el dataset limpio completo."""
    return pd.read_parquet(RUTA_DATOS)


# -------------------------------------------------------------------------
# RESPUESTAS CANDIDATAS
# -------------------------------------------------------------------------

def obtener_respuestas(
    question_ids,
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Obtiene todas las respuestas asociadas a las preguntas recuperadas.
    """

    respuestas = corpus[
        corpus["questionID"].isin(question_ids)
    ].copy()

    respuestas = respuestas[
        respuestas["answerText"]
        .fillna("")
        .str.strip()
        != ""
    ]

    if respuestas.empty:
        raise ValueError(
            "No existen respuestas para las preguntas recuperadas."
        )

    return respuestas.reset_index(drop=True)


# -------------------------------------------------------------------------
# RANKING
# -------------------------------------------------------------------------

def rankear_respuestas(
    consulta: str,
    question_ids,
    corpus: pd.DataFrame,
    modelo: CrossEncoder,
) -> pd.DataFrame:
    """
    Ordena todas las respuestas candidatas mediante un Cross-Encoder.
    """

    if not consulta or not consulta.strip():
        raise ValueError("La consulta no puede estar vacía.")

    respuestas = obtener_respuestas(
        question_ids,
        corpus,
    )

    # El Cross-Encoder lee conjuntamente:
    # [consulta del usuario, respuesta candidata]
    pares = [
        [consulta, respuesta]
        for respuesta in respuestas["answerText"]
    ]

    # Puntuación de relevancia para cada pareja.
    puntuaciones = modelo.predict(
        pares,
        show_progress_bar=False,
    )

    respuestas["score"] = np.asarray(
        puntuaciones
    ).reshape(-1)

    # Ordenamos de mayor a menor puntuación.
    respuestas = respuestas.sort_values(
        "score",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    respuestas.insert(
        0,
        "rank",
        range(1, len(respuestas) + 1),
    )

    return respuestas


def mejor_respuesta(
    consulta: str,
    question_ids,
    corpus: pd.DataFrame,
    modelo: CrossEncoder,
) -> pd.Series:
    """
    Devuelve la respuesta situada en primera posición.
    """

    ranking = rankear_respuestas(
        consulta,
        question_ids,
        corpus,
        modelo,
    )

    return ranking.iloc[0]


# -------------------------------------------------------------------------
# PRUEBA DEL MÓDULO
# -------------------------------------------------------------------------

def main() -> None:
    """
    Prueba técnica sencilla del módulo.
    """

    corpus = cargar_corpus()

    print(
        f"Cargando Cross-Encoder: {MODELO_RANKING}"
    )

    modelo = CrossEncoder(
        MODELO_RANKING
    )

    # Elegimos varias preguntas únicamente para comprobar
    # que el módulo funciona de forma independiente.
    question_ids = (
        corpus["questionID"]
        .drop_duplicates()
        .head(5)
        .tolist()
    )

    consulta = (
        corpus.loc[
            corpus["questionID"] == question_ids[0],
            "pregunta_texto",
        ]
        .iloc[0]
    )

    resultados = rankear_respuestas(
        consulta,
        question_ids,
        corpus,
        modelo,
    )

    print(f"\nConsulta:\n{consulta}\n")
    print("Ranking de respuestas:\n")

    for _, fila in resultados.head(10).iterrows():

        texto = fila["answerText"].replace(
            "\n",
            " ",
        )

        print(
            f"{fila['rank']:>2}. "
            f"score={fila['score']:.3f} "
            f"{texto[:120]}..."
        )


if __name__ == "__main__":
    main()