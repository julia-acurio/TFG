"""
retrieval.py
============
Etapa de RECUPERACIÓN del sistema.

Dada una consulta en inglés (p. ej. "I have no motivation to study"),
busca entre las preguntas de CounselChat aquellas con mayor similitud
semántica mediante embeddings densos de Sentence-BERT (SBERT).

Proceso:
  1. Se genera un embedding para cada pregunta del corpus.
  2. Estos embeddings se almacenan para evitar recalcularlos en cada ejecución.
  3. La consulta del usuario se transforma en otro embedding.
  4. Se calcula la similitud coseno entre la consulta y todas las preguntas.
  5. Se devuelven las top-k preguntas más similares.

Uso:
    python src/retrieval.py

También puede especificarse otro modelo:
    python src/retrieval.py --model all-mpnet-base-v2
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


# -------------------------------------------------------------------------
# CONFIGURACIÓN
# -------------------------------------------------------------------------

MODELO_SBERT = "all-mpnet-base-v2"

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_DATOS = BASE_DIR / "data" / "counselchat_limpio.parquet"

RUTA_EMBEDDINGS = BASE_DIR / "data" / "embeddings"


# -------------------------------------------------------------------------
# CARGA DEL CORPUS
# -------------------------------------------------------------------------

def cargar_preguntas() -> pd.DataFrame:
    """
    Carga el dataset limpio y obtiene una única fila por pregunta.

    El dataset contiene una fila por respuesta, por lo que una misma pregunta
    puede aparecer varias veces. Para la etapa de recuperación solo se necesita
    una representación de cada pregunta.
    """
    df = pd.read_parquet(RUTA_DATOS)

    columnas_necesarias = {"questionID", "pregunta_texto", "topic"}

    faltantes = columnas_necesarias - set(df.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias en el dataset: {faltantes}"
        )

    preguntas = (
        df.drop_duplicates(subset="questionID")
        [["questionID", "pregunta_texto", "topic"]]
        .dropna(subset=["pregunta_texto"])
        .reset_index(drop=True)
    )

    print(f"Preguntas únicas en el corpus: {len(preguntas)}")

    return preguntas


# -------------------------------------------------------------------------
# ÍNDICE SEMÁNTICO
# -------------------------------------------------------------------------

def ruta_indice(nombre_modelo: str) -> Path:
    """
    Construye una ruta diferente para los embeddings de cada modelo SBERT.
    """
    nombre_seguro = nombre_modelo.replace("/", "_")
    return RUTA_EMBEDDINGS / f"{nombre_seguro}.npy"


def construir_indice(
    preguntas: pd.DataFrame,
    modelo: SentenceTransformer,
) -> np.ndarray:
    """
    Convierte las preguntas del corpus en embeddings densos.
    """
    print("Generando embeddings del corpus...")

    embeddings = modelo.encode(
        preguntas["pregunta_texto"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def cargar_o_construir_indice(
    preguntas: pd.DataFrame,
    modelo: SentenceTransformer,
    nombre_modelo: str,
) -> np.ndarray:
    """
    Carga los embeddings desde disco si ya existen.

    Si todavía no han sido calculados, los genera con SBERT y los almacena
    para reutilizarlos en futuras ejecuciones.
    """
    RUTA_EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    ruta = ruta_indice(nombre_modelo)

    if ruta.exists():

        embeddings = np.load(ruta)

        # Comprobación básica para evitar utilizar un índice incompatible.
        if len(embeddings) == len(preguntas):
            print(f"Cargando embeddings existentes: {ruta.name}")
            return embeddings

        print(
            "El número de embeddings guardados no coincide con el corpus. "
            "Se reconstruirá el índice."
        )

    embeddings = construir_indice(preguntas, modelo)

    np.save(ruta, embeddings)

    print(f"Embeddings guardados en: {ruta}")

    return embeddings


# -------------------------------------------------------------------------
# RECUPERACIÓN
# -------------------------------------------------------------------------

def recuperar(
    consulta: str,
    preguntas: pd.DataFrame,
    embeddings: np.ndarray,
    modelo: SentenceTransformer,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recupera las top-k preguntas semánticamente más similares a la consulta.

    La similitud se calcula mediante similitud coseno entre el embedding de
    la consulta y los embeddings de las preguntas del corpus.
    """

    if not consulta or not consulta.strip():
        raise ValueError("La consulta no puede estar vacía.")

    if top_k <= 0:
        raise ValueError("top_k debe ser mayor que 0.")

    # Evita solicitar más resultados de los disponibles.
    top_k = min(top_k, len(preguntas))

    # Embedding de la nueva consulta.
    emb_consulta = modelo.encode(
        consulta,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Similitud con todas las preguntas del corpus.
    similitudes = util.cos_sim(
        emb_consulta,
        embeddings,
    )[0]

    # Índices correspondientes a las preguntas más similares.
    mejores = similitudes.argsort(descending=True)[:top_k]

    resultado = preguntas.iloc[mejores.tolist()].copy()

    resultado["similitud"] = (
        similitudes[mejores]
        .cpu()
        .numpy()
    )

    # Posición explícita dentro del ranking recuperado.
    resultado.insert(
        0,
        "rank",
        range(1, len(resultado) + 1),
    )

    return resultado.reset_index(drop=True)


# -------------------------------------------------------------------------
# EJECUCIÓN DE PRUEBA
# -------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Recuperación semántica de preguntas de CounselChat."
    )

    parser.add_argument(
        "--model",
        default=MODELO_SBERT,
        help="Modelo Sentence-BERT utilizado para generar los embeddings.",
    )

    parser.add_argument(
        "--query",
        default="I have no motivation to study",
        help="Consulta en inglés que se desea buscar.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Número de preguntas similares que se recuperan.",
    )

    args = parser.parse_args()

    print(f"Modelo: {args.model}")

    modelo = SentenceTransformer(args.model)

    preguntas = cargar_preguntas()

    embeddings = cargar_o_construir_indice(
        preguntas,
        modelo,
        args.model,
    )

    print(f"\nConsulta: {args.query!r}\n")

    resultados = recuperar(
        consulta=args.query,
        preguntas=preguntas,
        embeddings=embeddings,
        modelo=modelo,
        top_k=args.top_k,
    )

    print("Preguntas más similares del corpus:\n")

    for _, fila in resultados.iterrows():

        texto = fila["pregunta_texto"].replace("\n", " ")

        print(
            f"{fila['rank']:>2}. "
            f"[{fila['similitud']:.3f}] "
            f"({fila['topic']}) "
            f"{texto[:100]}"
        )


if __name__ == "__main__":
    main()