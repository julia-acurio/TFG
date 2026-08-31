# Ranking de respuestas en conversaciones de apoyo en salud mental

Trabajo Fin de Grado centrado en el diseño y evaluación de un sistema de recuperación y ranking de respuestas de apoyo en salud mental utilizando el dataset CounselChat.

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
