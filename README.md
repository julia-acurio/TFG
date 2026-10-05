# Ranking de respuestas en conversaciones de apoyo en salud mental

Este repositorio contiene el código y los experimentos desarrollados para el Trabajo Fin de Grado sobre ranking de respuestas en conversaciones de apoyo en salud mental utilizando el dataset CounselChat.

## Objetivo

El objetivo principal es predecir qué respuesta será la preferida entre las distintas respuestas disponibles para una misma pregunta.

Los `upvotes` se utilizan como señal de preferencia observada: una respuesta se considera top cuando presenta el mayor número de votos dentro de su pregunta.

Esta señal no se interpreta como una medida directa de calidad terapéutica, sino como una aproximación a la preferencia observada en el dataset.

## Datos

Se utiliza el dataset CounselChat, que contiene preguntas relacionadas con salud mental y respuestas proporcionadas por profesionales.

Para el problema de ranking se seleccionan únicamente preguntas que:

- disponen de al menos dos respuestas candidatas;
- presentan alguna señal positiva de `upvotes`.

Cuando varias respuestas de una misma pregunta comparten el máximo número de `upvotes`, todas ellas se consideran respuestas correctas durante la evaluación.

## Particionado

Las divisiones se realizan siempre a nivel de `questionID`, de forma que todas las respuestas de una misma pregunta permanecen dentro de una única partición.

El split train-test y los folds de validación se han estratificado teniendo en cuenta:

- el número de respuestas candidatas por pregunta;
- la presencia o ausencia de empate entre las respuestas con mayor número de `upvotes`.

El objetivo es conseguir particiones aproximadamente comparables entre sí.

## Métodos estudiados

Se han evaluado diferentes aproximaciones:

- baseline aleatorio;
- selección de la respuesta más larga;
- selección de la respuesta más corta;
- solapamiento léxico entre pregunta y respuesta;
- TF-IDF + regresión logística;
- BERT pairwise.

### BERT pairwise

En el enfoque pairwise, BERT aprende a comparar dos respuestas pertenecientes a una misma pregunta.

Para cada pareja con diferente número de `upvotes`, la respuesta con más votos se utiliza como preferida. Las parejas con el mismo número de votos se descartan porque no proporcionan una preferencia clara.

Durante la evaluación, se comparan entre sí todas las respuestas candidatas de una pregunta y se selecciona una única respuesta top-1 mediante un sistema de victorias y probabilidades acumuladas.

## Evaluación

La métrica principal utilizada actualmente es:

- `Accuracy@1`: porcentaje de preguntas para las que el modelo selecciona una de las respuestas con mayor número de `upvotes`.

El objetivo posterior es extender la evaluación al ranking completo mediante métricas como MRR y NDCG.

## Notebooks

Los principales análisis y experimentos se encuentran en `notebooks/`.

- `01_EDA_CounselChat.ipynb`: análisis exploratorio del dataset y estudio inicial de las señales de preferencia.
- `03_...`: experimentos con BERT pairwise y análisis de la composición de los folds de validación y test.
- `04_BERT_Particiones_Equilibradas.ipynb`: nuevo split estratificado, folds equilibrados y evaluación de BERT pairwise.

Los experimentos anteriores se conservan como referencia de la evolución metodológica del proyecto.

## Estado actual

Con las nuevas particiones, BERT pairwise obtiene:

- `Accuracy@1` media en validación cruzada: **44,09 %**
- desviación estándar: **5,10 %**
- `Accuracy@1` en test: **37,50 %**

También se ha analizado el límite de longitud de `bert-base-uncased`. El **29,34 % de los pares de entrenamiento supera los 512 tokens**, por lo que parte de la información se trunca durante el entrenamiento.

Este resultado motiva estudiar posteriormente modelos o estrategias capaces de trabajar con secuencias más largas.

## Reproducibilidad

Los experimentos utilizan semillas fijas y las divisiones se realizan por pregunta para evitar que respuestas de una misma consulta aparezcan simultáneamente en entrenamiento y evaluación.

Los notebooks conservan tanto el proceso de preparación de los datos como las particiones, el entrenamiento y la evaluación de los distintos modelos.
