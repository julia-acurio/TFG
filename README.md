# Sistema de predicción de preferencias sobre respuestas en orientación psicológica

Repositorio del Trabajo de Fin de Grado en **Ciencia e Ingeniería de Datos** de la Universidad Rey Juan Carlos.

## Descripción

Este proyecto estudia la predicción de preferencias entre respuestas de profesionales a consultas de orientación psicológica, utilizando el conjunto de datos **CounselChat**.

El objetivo es identificar qué respuesta tiene mayor probabilidad de ser preferida, tomando como referencia los votos positivos recibidos.

## Metodología

Se han desarrollado y comparado tres enfoques:

- **Basados en el texto:** características lingüísticas y modelos de procesamiento del lenguaje natural, incluidos BERT y ModernBERT.
- **Basados en el historial del terapeuta:** información sobre las preferencias recibidas por sus respuestas anteriores.
- **Enfoques combinados:** integración de características textuales e históricas.

La evaluación utiliza una separación de entrenamiento y prueba por pregunta y validación cruzada.

## Resultados

El modelo seleccionado utiliza regresión logística y dos características del historial del terapeuta.

En el conjunto de prueba alcanza una **exactitud del 58,04 %**, frente al **37,57 % esperado de un método aleatorio**.

Estos resultados reflejan la capacidad de predecir preferencias expresadas mediante votos, no la calidad clínica de las respuestas.

## Estructura del repositorio

- `notebooks/`: cuadernos de análisis, experimentación y evaluación.
- `data/`: archivos de datos utilizados en el proyecto.
- `experiments/`: archivos relacionados con los experimentos.
- `src/`: código auxiliar del proyecto.

## Autora

Julia Acurio Ajila  
Universidad Rey Juan Carlos
