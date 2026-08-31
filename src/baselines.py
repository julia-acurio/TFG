"""
baselines.py
============
Métodos de referencia utilizados para contextualizar el rendimiento
del sistema semántico.

Cada función recibe las respuestas candidatas de una pregunta y devuelve
sus posiciones ordenadas de mejor a peor.

Baselines:
  - Aleatorio: establece el rendimiento esperado sin utilizar información.
  - Longitud: ordena por número de palabras.
  - BM25: baseline léxico que ordena según coincidencia con la consulta.
  - Visualizaciones: baseline diagnóstico para estudiar el efecto de la
    exposición detectado durante el EDA.

Los upvotes no se utilizan para construir ningún ranking. Se reservan
exclusivamente como señal de referencia durante la evaluación.
"""

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


def tokenizar(texto: str) -> list[str]:
    """
    Tokenización sencilla para BM25.

    Convierte a minúsculas y conserva únicamente palabras y números.
    """
    return re.findall(r"\b\w+\b", texto.lower())


def ranking_aleatorio(
    candidatas: pd.DataFrame,
    rng: np.random.Generator,
) -> list[int]:
    """
    Ordena las respuestas candidatas al azar.
    """
    indices = np.arange(len(candidatas))

    rng.shuffle(indices)

    return indices.tolist()


def ranking_longitud(
    candidatas: pd.DataFrame,
) -> list[int]:
    """
    Ordena las respuestas de mayor a menor longitud.
    """
    candidatas = candidatas.reset_index(drop=True)

    return (
        candidatas["long_palabras"]
        .sort_values(
            ascending=False,
            kind="stable",
        )
        .index
        .tolist()
    )


def ranking_bm25(
    candidatas: pd.DataFrame,
    consulta: str,
) -> list[int]:
    """
    Ordena las respuestas según su similitud léxica con la consulta
    utilizando BM25.
    """

    if not consulta.strip():
        raise ValueError("La consulta no puede estar vacía.")

    candidatas = candidatas.reset_index(drop=True)

    corpus_tokenizado = [
        tokenizar(texto)
        for texto in candidatas["answerText"]
    ]

    consulta_tokenizada = tokenizar(consulta)

    bm25 = BM25Okapi(corpus_tokenizado)

    puntuaciones = bm25.get_scores(
        consulta_tokenizada
    )

    return np.argsort(
        -puntuaciones,
        kind="stable",
    ).tolist()


def ranking_visualizaciones(
    candidatas: pd.DataFrame,
) -> list[int]:
    """
    Ordena las respuestas de mayor a menor número de visualizaciones.

    Este baseline tiene únicamente una finalidad diagnóstica: permite
    cuantificar el efecto de la exposición sobre la señal de upvotes.
    No se utiliza como modelo de ranking del sistema.
    """

    candidatas = candidatas.reset_index(drop=True)

    return (
        candidatas["views"]
        .sort_values(
            ascending=False,
            kind="stable",
        )
        .index
        .tolist()
    )

def ranking_visualizaciones(candidatas: pd.DataFrame) -> list[int]:
    """
    Ordena las respuestas de mayor a menor número de visualizaciones.

    Se utiliza únicamente como baseline diagnóstico para medir
    el efecto de la exposición sobre los upvotes.
    """
    candidatas = candidatas.reset_index(drop=True)

    return (
        candidatas["views"]
        .sort_values(
            ascending=False,
            kind="stable",
        )
        .index
        .tolist()
    )