"""
prepare_training.py
===================
Prepara los datos para entrenar y evaluar el ranker V3.

Objetivo:
  1. Seleccionar preguntas que contienen información útil de preferencia.
  2. Crear una etiqueta binaria de preferencia observada.
  3. Separar las preguntas en TRAIN / VALIDATION / TEST.
  4. Garantizar que una misma pregunta nunca aparece en más de un conjunto.
  5. Guardar los splits para que todos los experimentos utilicen exactamente
     la misma partición.

IMPORTANTE:
Los upvotes se utilizan únicamente para construir la etiqueta de preferencia.
No se utilizan como característica de entrada del modelo.

Uso:
    python src/prepare_training.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

RANDOM_SEED = 27

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_DATOS = (
    BASE_DIR
    / "data"
    / "counselchat_limpio.parquet"
)

CARPETA_SALIDA = (
    BASE_DIR
    / "data"
    / "ranking"
)


# -------------------------------------------------------------------------
# CARGA
# -------------------------------------------------------------------------

def cargar_datos() -> pd.DataFrame:
    """Carga el dataset limpio generado por build_dataset.py."""

    df = pd.read_parquet(RUTA_DATOS)

    columnas_necesarias = {
        "questionID",
        "pregunta_texto",
        "answerText",
        "topic",
        "upvotes",
    }

    faltantes = columnas_necesarias - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias: {faltantes}"
        )

    return df.copy()


# -------------------------------------------------------------------------
# SELECCIÓN DE PREGUNTAS ÚTILES
# -------------------------------------------------------------------------

def seleccionar_preguntas_utiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Conserva únicamente preguntas que permiten aprender una preferencia.

    Una pregunta es útil cuando:
      - tiene al menos dos respuestas;
      - no todas las respuestas tienen los mismos upvotes.

    Por ejemplo:

        respuesta A -> 3 upvotes
        respuesta B -> 0 upvotes

    contiene señal.

    En cambio:

        respuesta A -> 0
        respuesta B -> 0

    no permite saber cuál fue preferida.
    """

    df = df.copy()

    grupos = df.groupby("questionID")

    n_respuestas = grupos.size()

    minimo_upvotes = grupos["upvotes"].min()
    maximo_upvotes = grupos["upvotes"].max()

    preguntas_utiles = n_respuestas[
        (n_respuestas >= 2)
        & (maximo_upvotes > minimo_upvotes)
    ].index

    df = df[
        df["questionID"].isin(preguntas_utiles)
    ].copy()

    print(
        f"Preguntas con señal de preferencia: "
        f"{df['questionID'].nunique()}"
    )

    return df


# -------------------------------------------------------------------------
# ETIQUETAS
# -------------------------------------------------------------------------

def crear_etiquetas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Crea la etiqueta que aprenderá el modelo.

    etiqueta_preferencia = 1
        si la respuesta tiene el máximo de upvotes de su pregunta.

    etiqueta_preferencia = 0
        para el resto.

    Si varias respuestas empatan en el máximo, todas reciben etiqueta 1.
    """

    df = df.copy()

    maximos = df.groupby(
        "questionID"
    )["upvotes"].transform("max")

    df["etiqueta_preferencia"] = (
        df["upvotes"] == maximos
    ).astype(int)

    # Número de respuestas que comparten el máximo.
    n_maximos = df.groupby(
        "questionID"
    )["etiqueta_preferencia"].transform("sum")

    df["empate_maximo"] = n_maximos > 1

    # Número total de respuestas candidatas de la pregunta.
    df["n_respuestas"] = df.groupby(
        "questionID"
    )["questionID"].transform("size")

    return df


# -------------------------------------------------------------------------
# SPLIT POR PREGUNTAS
# -------------------------------------------------------------------------

def crear_splits(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Divide las preguntas en:

        70 % train
        15 % validation
        15 % test

    La separación se realiza siempre por questionID.

    Para obtener conjuntos comparables, se estratifica considerando:
      - si existe empate en la respuesta más votada;
      - el número de respuestas candidatas de la pregunta.
    """

    preguntas = (
        df.groupby("questionID")
        .agg(
            empate_maximo=("empate_maximo", "first"),
            n_respuestas=("n_respuestas", "first"),
            topic=("topic", "first"),
        )
        .reset_index()
    )

    # Agrupamos las preguntas según su número de respuestas.
    preguntas["grupo_respuestas"] = pd.cut(
        preguntas["n_respuestas"],
        bins=[1, 2, 5, float("inf")],
        labels=["2", "3-5", ">5"],
    )

    # Combinamos dificultad aproximada + existencia de empate.
    preguntas["estrato"] = (
        preguntas["grupo_respuestas"].astype(str)
        + "_empate_"
        + preguntas["empate_maximo"].astype(str)
    )

    # Comprobación antes de estratificar.
    conteo_estratos = preguntas["estrato"].value_counts()

    if (conteo_estratos < 4).any():
        raise ValueError(
            "Algún estrato contiene demasiado pocas preguntas "
            "para realizar una partición estratificada fiable:\n"
            f"{conteo_estratos}"
        )

    # -----------------------------------------------------
    # 70 % TRAIN / 30 % RESTO
    # -----------------------------------------------------

    train, resto = train_test_split(
        preguntas,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=preguntas["estrato"],
    )

    # -----------------------------------------------------
    # 15 % VALIDATION / 15 % TEST
    # -----------------------------------------------------

    validacion, test = train_test_split(
        resto,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=resto["estrato"],
    )

    train = train.copy()
    validacion = validacion.copy()
    test = test.copy()

    train["split"] = "train"
    validacion["split"] = "validation"
    test["split"] = "test"

    manifest = pd.concat(
        [train, validacion, test],
        ignore_index=True,
    )

    return manifest

# -------------------------------------------------------------------------
# COMPROBACIONES DE SEGURIDAD
# -------------------------------------------------------------------------

def comprobar_splits(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """
    Comprueba automáticamente que no exista fuga de información.
    """

    train_ids = set(
        manifest.loc[
            manifest["split"] == "train",
            "questionID",
        ]
    )

    validation_ids = set(
        manifest.loc[
            manifest["split"] == "validation",
            "questionID",
        ]
    )

    test_ids = set(
        manifest.loc[
            manifest["split"] == "test",
            "questionID",
        ]
    )

    # Ninguna pregunta puede estar en dos conjuntos.
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    # Todas las preguntas útiles deben aparecer exactamente una vez.
    todos_ids = (
        train_ids
        | validation_ids
        | test_ids
    )

    ids_originales = set(
        df["questionID"].unique()
    )

    assert todos_ids == ids_originales

    # Cada pregunta debe tener al menos una respuesta positiva
    # y una negativa.
    resumen = df.groupby(
        "questionID"
    )["etiqueta_preferencia"].agg(
        ["min", "max"]
    )

    assert (resumen["min"] == 0).all()
    assert (resumen["max"] == 1).all()

    print(
        "\nComprobación de fuga de datos: CORRECTA"
    )


# -------------------------------------------------------------------------
# GUARDADO
# -------------------------------------------------------------------------

def guardar_splits(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    """Guarda los conjuntos que utilizará V3."""

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Incorporamos a cada respuesta el split de su pregunta.
    df = df.merge(
        manifest[
            ["questionID", "split"]
        ],
        on="questionID",
        how="left",
        validate="many_to_one",
    )

    # IMPORTANTE:
    # upvotes NO se guardan en los archivos usados para entrenamiento,
    # para reducir el riesgo de utilizarlos accidentalmente como entrada.
    columnas_modelo = [
        "questionID",
        "pregunta_texto",
        "answerText",
        "topic",
        "etiqueta_preferencia",
        "empate_maximo",
        "n_respuestas",
    ]

    for nombre in [
        "train",
        "validation",
        "test",
    ]:

        datos = df[
            df["split"] == nombre
        ][columnas_modelo].copy()

        ruta = (
            CARPETA_SALIDA
            / f"{nombre}.parquet"
        )

        datos.to_parquet(
            ruta,
            index=False,
        )

    # Manifest de preguntas:
    # permite saber exactamente qué pregunta pertenece a cada split.
    manifest.to_csv(
        CARPETA_SALIDA
        / "split_manifest.csv",
        index=False,
    )


# -------------------------------------------------------------------------
# RESUMEN
# -------------------------------------------------------------------------

def mostrar_resumen(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:

    print("\nDistribución de preguntas:\n")

    for nombre in [
        "train",
        "validation",
        "test",
    ]:

        ids = manifest.loc[
            manifest["split"] == nombre,
            "questionID",
        ]

        datos = df[
            df["questionID"].isin(ids)
        ]

        print(
            f"{nombre:<11} "
            f"{len(ids):>4} preguntas | "
            f"{len(datos):>4} respuestas"
        )

    print(
        "\nPreguntas con ganador único: "
        f"{(~manifest['empate_maximo']).sum()}"
    )

    print(
        "Preguntas con empate en el máximo: "
        f"{manifest['empate_maximo'].sum()}"
    )


# -------------------------------------------------------------------------
# EJECUCIÓN
# -------------------------------------------------------------------------

def main() -> None:

    print("Cargando dataset...")

    df = cargar_datos()

    df = seleccionar_preguntas_utiles(
        df
    )

    df = crear_etiquetas(
        df
    )

    manifest = crear_splits(
        df
    )

    comprobar_splits(
        df,
        manifest,
    )

    mostrar_resumen(
        df,
        manifest,
    )

    guardar_splits(
        df,
        manifest,
    )

    print(
        f"\nSplits guardados en:\n"
        f"{CARPETA_SALIDA}"
    )


if __name__ == "__main__":
    main()