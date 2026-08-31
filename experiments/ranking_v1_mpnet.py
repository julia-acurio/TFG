"""
ranking_v1_mpnet.py
===================
Primera versión experimental del ranking.

Ordena las respuestas de una única pregunta según su similitud
semántica con la consulta utilizando MPNet.

Los upvotes no se utilizan para construir el ranking.
"""

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer, util


MODELO_SBERT = "all-mpnet-base-v2"

RUTA_DATOS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "counselchat_limpio.parquet"
)


def cargar_corpus() -> pd.DataFrame:
    """Carga el dataset limpio completo."""
    return pd.read_parquet(RUTA_DATOS)


def obtener_respuestas(
    question_id,
    corpus: pd.DataFrame,
) -> pd.DataFrame:

    respuestas = corpus[
        corpus["questionID"] == question_id
    ].copy()

    respuestas = respuestas.dropna(
        subset=["answerText"]
    )

    if respuestas.empty:
        raise ValueError(
            f"No existen respuestas para questionID={question_id!r}"
        )

    return respuestas.reset_index(drop=True)


def rankear_respuestas(
    consulta: str,
    question_id,
    corpus: pd.DataFrame,
    modelo: SentenceTransformer,
) -> pd.DataFrame:

    if not consulta.strip():
        raise ValueError(
            "La consulta no puede estar vacía."
        )

    respuestas = obtener_respuestas(
        question_id,
        corpus,
    )

    emb_consulta = modelo.encode(
        consulta,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    emb_respuestas = modelo.encode(
        respuestas["answerText"].tolist(),
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similitudes = util.cos_sim(
        emb_consulta,
        emb_respuestas,
    )[0]

    respuestas["score"] = (
        similitudes.cpu().numpy()
    )

    respuestas = respuestas.sort_values(
        "score",
        ascending=False,
    ).reset_index(drop=True)

    respuestas.insert(
        0,
        "rank",
        range(1, len(respuestas) + 1),
    )

    return respuestas


def mejor_respuesta(
    consulta: str,
    question_id,
    corpus: pd.DataFrame,
    modelo: SentenceTransformer,
) -> pd.Series:

    ranking = rankear_respuestas(
        consulta,
        question_id,
        corpus,
        modelo,
    )

    return ranking.iloc[0]