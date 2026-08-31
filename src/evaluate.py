"""
evaluate.py
===========
Evaluación de los métodos de ranking de respuestas.

Compara:
  - Cross-Encoder: nuevo sistema de ranking (V2).
  - MPNet: sistema semántico inicial (V1).
  - BM25.
  - Longitud.
  - Aleatorio.
  - Visualizaciones (solo diagnóstico).

Los upvotes se utilizan únicamente para identificar la respuesta
de referencia durante la evaluación.

Métricas:
  - Precision@1
  - MRR

Uso:
    python src/evaluate.py
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder, util

import baselines
import ranking


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

MODELO_MPNET = "all-mpnet-base-v2"

RANDOM_SEED = 27
N_REPETICIONES_AZAR = 100


# -------------------------------------------------------------------------
# CARGA
# -------------------------------------------------------------------------

def cargar_evaluacion():
    """Carga el corpus y selecciona las preguntas de evaluación."""

    corpus = ranking.cargar_corpus()

    df_eval = corpus[
        corpus["en_evaluacion"]
    ].copy()

    print(
        f"Conjunto de evaluación: "
        f"{df_eval['questionID'].nunique()} preguntas, "
        f"{len(df_eval)} respuestas"
    )

    return corpus, df_eval


# -------------------------------------------------------------------------
# MÉTRICAS
# -------------------------------------------------------------------------

def evaluar_metodo(
    df_eval: pd.DataFrame,
    funcion_ranking,
) -> dict:
    """
    Evalúa un método que devuelve los índices de las respuestas
    ordenados de mejor a peor.
    """

    aciertos = 0
    suma_reciprocos = 0.0
    n = 0

    for _, candidatas in df_eval.groupby(
        "questionID",
        sort=False,
    ):

        candidatas = candidatas.reset_index(drop=True)

        # Respuesta de referencia.
        idx_referencia = candidatas[
            "upvotes"
        ].idxmax()

        # Ranking generado sin utilizar upvotes.
        orden = funcion_ranking(candidatas)

        posicion = (
            orden.index(idx_referencia) + 1
        )

        if posicion == 1:
            aciertos += 1

        suma_reciprocos += 1.0 / posicion
        n += 1

    return {
        "Precision@1": aciertos / n,
        "MRR": suma_reciprocos / n,
    }


# -------------------------------------------------------------------------
# MPNET - V1
# -------------------------------------------------------------------------

def crear_ranking_mpnet(
    modelo: SentenceTransformer,
):

    def ranking_mpnet(candidatas):

        candidatas = candidatas.reset_index(
            drop=True
        )

        consulta = candidatas[
            "pregunta_texto"
        ].iloc[0]

        emb_consulta = modelo.encode(
            consulta,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        emb_respuestas = modelo.encode(
            candidatas["answerText"].tolist(),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similitudes = util.cos_sim(
            emb_consulta,
            emb_respuestas,
        )[0].cpu().numpy()

        return np.argsort(
            -similitudes,
            kind="stable",
        ).tolist()

    return ranking_mpnet


# -------------------------------------------------------------------------
# CROSS-ENCODER - V2
# -------------------------------------------------------------------------

def evaluar_crossencoder(
    df_eval: pd.DataFrame,
    corpus: pd.DataFrame,
    modelo: CrossEncoder,
) -> dict:

    aciertos = 0
    suma_reciprocos = 0.0
    n = 0

    for question_id, candidatas in df_eval.groupby(
        "questionID",
        sort=False,
    ):

        consulta = candidatas[
            "pregunta_texto"
        ].iloc[0]

        # Usamos directamente ranking.py.
        resultado = ranking.rankear_respuestas(
            consulta=consulta,
            question_ids=[question_id],
            corpus=corpus,
            modelo=modelo,
        )

        # Única respuesta con máximo de upvotes.
        max_upvotes = candidatas[
            "upvotes"
        ].max()

        posicion = (
            resultado.index[
                resultado["upvotes"]
                == max_upvotes
            ][0]
            + 1
        )

        if posicion == 1:
            aciertos += 1

        suma_reciprocos += 1.0 / posicion
        n += 1

    return {
        "Precision@1": aciertos / n,
        "MRR": suma_reciprocos / n,
    }


# -------------------------------------------------------------------------
# ALEATORIO
# -------------------------------------------------------------------------

def evaluar_aleatorio(
    df_eval: pd.DataFrame,
) -> dict:
    """
    Repite el baseline aleatorio varias veces y devuelve la media.
    """

    resultados = []

    for repeticion in range(
        N_REPETICIONES_AZAR
    ):

        rng = np.random.default_rng(
            RANDOM_SEED + repeticion
        )

        def aleatorio(candidatas):
            return baselines.ranking_aleatorio(
                candidatas,
                rng,
            )

        resultados.append(
            evaluar_metodo(
                df_eval,
                aleatorio,
            )
        )

    return {
        "Precision@1": np.mean([
            r["Precision@1"]
            for r in resultados
        ]),
        "MRR": np.mean([
            r["MRR"]
            for r in resultados
        ]),
    }


# -------------------------------------------------------------------------
# EJECUCIÓN
# -------------------------------------------------------------------------

def main():

    corpus, df_eval = cargar_evaluacion()

    print("\nCargando MPNet...")
    modelo_mpnet = SentenceTransformer(
        MODELO_MPNET
    )

    print("Cargando Cross-Encoder...")
    modelo_cross = CrossEncoder(
        ranking.MODELO_RANKING
    )

    # MPNet V1
    ranking_mpnet = crear_ranking_mpnet(
        modelo_mpnet
    )

    # BM25
    def bm25(candidatas):

        consulta = candidatas[
            "pregunta_texto"
        ].iloc[0]

        return baselines.ranking_bm25(
            candidatas,
            consulta,
        )

    resultados = {}

    print("\nEvaluando Cross-Encoder...")
    resultados["Cross-Encoder V2"] = (
        evaluar_crossencoder(
            df_eval,
            corpus,
            modelo_cross,
        )
    )

    print("Evaluando MPNet...")
    resultados["MPNet V1"] = evaluar_metodo(
        df_eval,
        ranking_mpnet,
    )

    print("Evaluando BM25...")
    resultados["BM25"] = evaluar_metodo(
        df_eval,
        bm25,
    )

    print("Evaluando longitud...")
    resultados["Longitud"] = evaluar_metodo(
        df_eval,
        baselines.ranking_longitud,
    )

    print("Evaluando aleatorio...")
    resultados["Aleatorio (media)"] = (
        evaluar_aleatorio(df_eval)
    )

    print("Evaluando visualizaciones...")
    resultados["Views (diagnóstico)"] = (
        evaluar_metodo(
            df_eval,
            baselines.ranking_visualizaciones,
        )
    )

    print(
        "\nResultados sobre el conjunto de evaluación:\n"
    )

    print(
        f"{'Método':<24}"
        f"{'Precision@1':>13}"
        f"{'MRR':>10}"
    )

    print("-" * 47)

    for nombre, r in resultados.items():

        print(
            f"{nombre:<24}"
            f"{r['Precision@1']:>13.3f}"
            f"{r['MRR']:>10.3f}"
        )


if __name__ == "__main__":
    main()