"""
train_ranker.py
===============
Entrenamiento del ranker V3.

Se fine-tunea un Cross-Encoder sobre CounselChat utilizando un enfoque
listwise: para cada pregunta, el modelo observa conjuntamente todas sus
respuestas candidatas y aprende a situar arriba las respuestas con
preferencia observada.

TRAIN:
    utilizado para aprender los parámetros.

VALIDATION:
    utilizado para seleccionar el mejor checkpoint.

TEST:
    NO se utiliza en este archivo.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    losses,
)

from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderRerankingEvaluator,
)


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

RANDOM_SEED = 27

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_TRAIN = (
    BASE_DIR / "data" / "ranking" / "train.parquet"
)

RUTA_VALIDATION = (
    BASE_DIR / "data" / "ranking" / "validation.parquet"
)

CARPETA_MODELOS = (
    BASE_DIR / "models"
)

MODELO_BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"

RUTA_MODELO_FINAL = (
    CARPETA_MODELOS / "ranker_v3"
)

RUTA_CHECKPOINTS = (
    CARPETA_MODELOS / "ranker_v3_checkpoints"
)


# -------------------------------------------------------------------------
# REPRODUCIBILIDAD
# -------------------------------------------------------------------------

def fijar_semillas() -> None:
    """Fija las semillas utilizadas durante el entrenamiento."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# -------------------------------------------------------------------------
# CARGA
# -------------------------------------------------------------------------

def cargar_datos():
    """Carga únicamente TRAIN y VALIDATION."""

    train = pd.read_parquet(RUTA_TRAIN)
    validation = pd.read_parquet(RUTA_VALIDATION)

    # Comprobación adicional contra fuga de información.
    ids_train = set(train["questionID"])
    ids_validation = set(validation["questionID"])

    if not ids_train.isdisjoint(ids_validation):
        raise ValueError(
            "ERROR: existen questionID compartidos entre train y validation."
        )

    print(
        f"Train: "
        f"{train['questionID'].nunique()} preguntas, "
        f"{len(train)} respuestas"
    )

    print(
        f"Validation: "
        f"{validation['questionID'].nunique()} preguntas, "
        f"{len(validation)} respuestas"
    )

    return train, validation


# -------------------------------------------------------------------------
# FORMATO LISTWISE
# -------------------------------------------------------------------------

def convertir_listwise(
    df: pd.DataFrame,
) -> Dataset:
    """
    Convierte el DataFrame en una fila por pregunta.

    Cada ejemplo tendrá:

        query  -> texto de la pregunta
        docs   -> lista de respuestas candidatas
        labels -> lista de etiquetas 0/1
    """

    queries = []
    docs = []
    labels = []

    for _, grupo in df.groupby(
        "questionID",
        sort=False,
    ):

        grupo = grupo.reset_index(drop=True)

        queries.append(
            grupo["pregunta_texto"].iloc[0]
        )

        docs.append(
            grupo["answerText"].tolist()
        )

        labels.append(
            grupo["etiqueta_preferencia"]
            .astype(float)
            .tolist()
        )

    return Dataset.from_dict(
        {
            "query": queries,
            "docs": docs,
            "labels": labels,
        }
    )


# -------------------------------------------------------------------------
# EVALUADOR DE VALIDACIÓN
# -------------------------------------------------------------------------

def crear_evaluador(
    validation: pd.DataFrame,
):
    """
    Construye el evaluador de ranking para validation.
    """

    samples = []

    max_candidatas = 0

    for _, grupo in validation.groupby(
        "questionID",
        sort=False,
    ):

        positivas = grupo.loc[
            grupo["etiqueta_preferencia"] == 1,
            "answerText",
        ].tolist()

        negativas = grupo.loc[
            grupo["etiqueta_preferencia"] == 0,
            "answerText",
        ].tolist()

        samples.append(
            {
                "query": grupo[
                    "pregunta_texto"
                ].iloc[0],
                "positive": positivas,
                "negative": negativas,
            }
        )

        max_candidatas = max(
            max_candidatas,
            len(grupo),
        )

    evaluator = CrossEncoderRerankingEvaluator(
        samples=samples,
        at_k=max_candidatas,
        name="validation",
        batch_size=16,
        show_progress_bar=True,
    )

    return evaluator, max_candidatas


# -------------------------------------------------------------------------
# P@1 Y MRR
# -------------------------------------------------------------------------

def evaluar_metricas(
    modelo,
    df: pd.DataFrame,
) -> dict:
    """
    Calcula Precision@1 y MRR sobre un conjunto de preguntas.

    Si existe empate en la referencia, cualquiera de las respuestas
    con etiqueta 1 se considera correcta.
    """

    aciertos = 0
    suma_rr = 0.0
    n = 0

    for _, grupo in df.groupby(
        "questionID",
        sort=False,
    ):

        grupo = grupo.reset_index(drop=True)

        consulta = grupo[
            "pregunta_texto"
        ].iloc[0]

        pares = [
            [consulta, respuesta]
            for respuesta in grupo["answerText"]
        ]

        scores = np.asarray(
            modelo.predict(
                pares,
                show_progress_bar=False,
            )
        ).reshape(-1)

        orden = np.argsort(
            -scores,
            kind="stable",
        )

        etiquetas = (
            grupo["etiqueta_preferencia"]
            .to_numpy()
        )

        # Precision@1
        if etiquetas[orden[0]] == 1:
            aciertos += 1

        # Primera respuesta relevante.
        for posicion, indice in enumerate(
            orden,
            start=1,
        ):
            if etiquetas[indice] == 1:
                suma_rr += 1.0 / posicion
                break

        n += 1

    return {
        "Precision@1": aciertos / n,
        "MRR": suma_rr / n,
    }


# -------------------------------------------------------------------------
# ENTRENAMIENTO
# -------------------------------------------------------------------------

def main():

    fijar_semillas()

    CARPETA_MODELOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df, validation_df = cargar_datos()

    train_dataset = convertir_listwise(
        train_df
    )

    validation_dataset = convertir_listwise(
        validation_df
    )

    print(
        f"\nModelo base: {MODELO_BASE}"
    )

    print(
        "Dispositivo:",
        "GPU" if torch.cuda.is_available()
        else "CPU",
    )

    modelo = CrossEncoder(
        MODELO_BASE,
        num_labels=1,
    )

    # -----------------------------------------------------
    # VALIDACIÓN ANTES DE ENTRENAR
    # -----------------------------------------------------

    print(
        "\nRendimiento del modelo ANTES del fine-tuning:"
    )

    metricas_antes = evaluar_metricas(
        modelo,
        validation_df,
    )

    print(
        f"Precision@1 = "
        f"{metricas_antes['Precision@1']:.3f}"
    )

    print(
        f"MRR = "
        f"{metricas_antes['MRR']:.3f}"
    )

    # -----------------------------------------------------
    # LAMBDA LOSS
    # -----------------------------------------------------

    loss = losses.LambdaLoss(
        modelo
    )

    evaluator, max_candidatas = crear_evaluador(
        validation_df
    )

    # Seleccionamos el mejor checkpoint mediante MRR
    # calculado sobre todas las candidatas de validation.
    metrica_mejor_modelo = (
        f"validation_mrr@{max_candidatas}"
    )

    # -----------------------------------------------------
    # CONFIGURACIÓN
    # -----------------------------------------------------

    args = CrossEncoderTrainingArguments(
        output_dir=str(RUTA_CHECKPOINTS),

        num_train_epochs=3,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=8,

        learning_rate=1e-5,
        warmup_steps=0.1,

        eval_strategy="epoch",
        save_strategy="epoch",

        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model=metrica_mejor_modelo,
        greater_is_better=True,

        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,

        logging_steps=10,

        report_to="none",
    )

    trainer = CrossEncoderTrainer(
        model=modelo,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    # -----------------------------------------------------
    # ENTRENAMIENTO
    # -----------------------------------------------------

    print(
        "\nComenzando entrenamiento V3...\n"
    )

    trainer.train()

    mejor_modelo = trainer.model

    # -----------------------------------------------------
    # VALIDACIÓN FINAL
    # -----------------------------------------------------

    print(
        "\nRendimiento DESPUÉS del fine-tuning:"
    )

    metricas_despues = evaluar_metricas(
        mejor_modelo,
        validation_df,
    )

    print(
        f"Precision@1 = "
        f"{metricas_despues['Precision@1']:.3f}"
    )

    print(
        f"MRR = "
        f"{metricas_despues['MRR']:.3f}"
    )

    # -----------------------------------------------------
    # GUARDADO
    # -----------------------------------------------------

    mejor_modelo.save_pretrained(
        str(RUTA_MODELO_FINAL)
    )

    resultados = {
        "modelo_base": MODELO_BASE,
        "n_train_preguntas":
            int(train_df["questionID"].nunique()),
        "n_validation_preguntas":
            int(validation_df["questionID"].nunique()),
        "antes": metricas_antes,
        "despues": metricas_despues,
    }

    with open(
        RUTA_MODELO_FINAL
        / "resultados_validation.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            resultados,
            f,
            indent=4,
        )

    print(
        f"\nModelo guardado en:\n"
        f"{RUTA_MODELO_FINAL}"
    )

    print(
        "\nIMPORTANTE: test.parquet NO se ha utilizado."
    )


if __name__ == "__main__":
    main()