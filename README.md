# Ranking de respuestas en conversaciones de apoyo en salud mental

Este repositorio contiene e código correspondiente alTrabajo Fin de Grado centrado en el diseño y evaluación de un sistema de recuperación y ranking de respuestas de apoyo en salud mental utilizando el dataset CounselChat.

## Objetivo

Dada una consulta de un usuario, el sistema:

1. Recupera preguntas semánticamente similares de CounselChat.
2. Obtiene las respuestas asociadas.
3. Ordena las respuestas candidatas mediante distintos métodos de ranking.
4. Evalúa el rendimiento utilizando señales de preferencia observadas en el dataset.

Los `upvotes` se utilizan como señal de referencia para la evaluación, pero no se interpretan como una medida directa de calidad terapéutica.

## Estructura del proyecto


```text
TFG/
├── data/
│   └── ranking/
│       └── split_manifest.csv
│
├── experiments/
│   ├── ranking_v1_mpnet.py
│   ├── ranking_v2_crossencoder.py
│   └── evaluate_v2_comparacion.py
│
├── notebooks/
│   ├── 01_EDA_CounselChat.ipynb
│   └── archive/
│
├── src/
│   ├── baselines.py
│   ├── build_dataset.py
│   ├── evaluate.py
│   ├── evaluate_validation.py
│   ├── main.py
│   ├── prepare_training.py
│   ├── ranking.py
│   ├── retrieval.py
│   └── train_ranker.py
│
└── README.md
```
## Análisis exploratorio

El análisis exploratorio principal se encuentra en:

`notebooks/01_EDA_CounselChat.ipynb`

En él se estudian, entre otros aspectos:

- estructura y calidad del dataset;
- número de respuestas disponibles por pregunta;
- distribución de `upvotes`;
- viabilidad del problema de ranking;
- longitud de las respuestas;
- distribución temática;
- relación entre `views` y `upvotes`.

## Preparación de los datos

El dataset limpio se genera mediante:

```bash
python src/build_dataset.py
```

Los conjuntos utilizados para entrenamiento, validación y test se preparan mediante:

```bash
python src/prepare_training.py
```

La separación se realiza por `questionID`, de forma que todas las respuestas asociadas a una misma pregunta permanecen en el mismo conjunto y se evita fuga de información entre train, validation y test.

La partición utilizada se conserva en:

`data/ranking/split_manifest.csv`

## Métodos estudiados

Durante el desarrollo se han evaluado diferentes aproximaciones:

- baseline aleatorio;
- baseline basado en longitud de respuesta;
- BM25 como método léxico;
- similitud semántica mediante MPNet;
- Cross-Encoder preentrenado;
- Cross-Encoder ajustado con datos de CounselChat.

Las métricas principales utilizadas para evaluar el ranking son:

- Precision@1;
- Mean Reciprocal Rank (MRR).

## Recuperación y ranking

La recuperación semántica de preguntas puede probarse mediante:

```bash
python src/retrieval.py
```

El sistema completo puede ejecutarse mediante:

```bash
python src/main.py
```
## Estado experimental

Los experimentos realizados muestran que la similitud semántica por sí sola no reproduce completamente la preferencia observada mediante `upvotes`.

También se ha observado que características como la longitud de las respuestas y, especialmente, su exposición mediante `views`, presentan una asociación relevante con los votos obtenidos.

Estos resultados motivan un análisis crítico de los `upvotes` como señal de referencia y de las limitaciones de utilizar esta variable como aproximación a la preferencia o calidad de una respuesta en un contexto de apoyo en salud mental.

## Reproducibilidad

Los datos procesados, embeddings y modelos entrenados pueden regenerarse mediante los scripts incluidos en `src/`.

Los archivos generados de gran tamaño no se almacenan directamente en el repositorio. El archivo `split_manifest.csv` permite conservar la partición experimental utilizada durante el desarrollo.
