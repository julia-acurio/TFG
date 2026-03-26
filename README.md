# MentalRiskES 2026 — Tarea 2: Selección de Respuesta Terapéutica

**Trabajo de Fin de Grado**

**Autora:** Julia Acurio Ajila

**Universidad:** Universidad Rey Juan Carlos

**Tutoras:** Natalia Madrueño Sierro y Soto Montalvo

**Competición:** MentalRiskES@IberLEF 2026

**Web oficial:** https://sites.google.com/view/mentalriskes2026

---

## Descripción del proyecto

Este Trabajo de Fin de Grado se enmarca en la competición
**MentalRiskES@IberLEF 2026**, celebrada en el contexto de la conferencia
**SEPLN 2026** en León (España). La competición propone retos de Procesamiento
del Lenguaje Natural (PLN) aplicados a la detección temprana de riesgo en
salud mental en español.

El presente trabajo aborda la **Tarea 2: Selección de Respuesta Terapéutica**,
cuyo objetivo es desarrollar un sistema de apoyo a la decisión clínica en
entornos terapéuticos. Dado el último mensaje de un paciente en una
conversación terapéutica y tres respuestas candidatas del terapeuta, el
sistema debe seleccionar automáticamente la respuesta más adecuada conforme
a las buenas prácticas clínicas definidas por expertos.

El corpus utilizado consiste en conversaciones terapéuticas simuladas pero
clínicamente plausibles en español, generadas mediante una combinación de
diálogos redactados por humanos y síntesis automática, y revisadas por
profesionales del ámbito clínico.

Una de las principales dificultades de la tarea reside en que el historial
completo de la conversación **no se suministra en cada paso**. El sistema
debe acumular e integrar el contexto de la conversación de forma progresiva
a lo largo de los turnos sucesivos.

---

## Enfoque propuesto: RAG (Retrieval-Augmented Generation)

El sistema desarrollado se basa en la arquitectura
**RAG (Retrieval-Augmented Generation)** (Lewis et al., 2020), que combina
la capacidad generativa de los modelos de lenguaje con la recuperación de
información externa relevante.

La motivación clínica de este enfoque es directa: un terapeuta experto no
responde únicamente a partir de su experiencia inmediata, sino apoyándose
en un corpus de conocimiento especializado. El sistema replica este
comportamiento consultando una base de conocimiento clínico antes de
seleccionar la respuesta más apropiada.

El sistema se compone de tres módulos principales:

**1. Base de conocimiento clínico**
Colección de documentos especializados en psicoterapia: incluyendo manuales
de Terapia Cognitivo-Conductual (TCC), Terapia de Aceptación y Compromiso
(ACT) y Entrevista Motivacional, indexados mediante representaciones
vectoriales densas (embeddings semánticos).

**2. Módulo de recuperación semántica**
Dado el mensaje del paciente en cada turno, este módulo recupera
automáticamente los fragmentos más relevantes de la base de conocimiento
mediante búsqueda por similitud semántica en un espacio vectorial.

**3. Módulo de selección mediante LLM**
Un modelo de lenguaje de gran escala (GPT-4o) recibe como entrada el
historial acumulado de la conversación, los fragmentos recuperados y las
tres respuestas candidatas, y selecciona la más adecuada según el contexto
clínico.

---

## Estructura del repositorio

```
├── data/             # Dataset de MentalRiskES 2026 (no incluido por acuerdo de uso)
├── knowledge_base/   # Documentos clínicos indexados para el módulo RAG
├── src/              # Código fuente del sistema
│   ├── indexer.py        # Construcción de la base de datos vectorial
│   ├── retriever.py      # Recuperación semántica de fragmentos relevantes
│   └── selector.py       # Selección de respuesta mediante LLM
├── notebooks/        # Análisis exploratorio y experimentación
├── results/          # Resultados de los experimentos realizados
└── docs/             # Documentación y acuerdos de uso de la competición
```

## Requisitos e instalación

```bash
git clone https://github.com/julia-acurio/TFG.git
cd TFG
pip install -r requirements.txt
```

El corpus de MentalRiskES 2026 está sujeto a un acuerdo de uso restringido.
Para acceder al dataset es necesario registrarse en la competición y firmar
el acuerdo disponible en el directorio `docs/` de este repositorio y en la
web oficial de la competición.

---

## Referencias

- Mármol-Romero, A. M., Moreno-Muñoz, A., Plaza-del-Arco, F. M.,
  Molina-González, M. D., Martín-Valdivia, M. T., Ureña-López, L. A., y
  Montejo-Ráez, A. (2024). Overview of MentalRiskES at IberLEF 2024: Early
  Detection of Mental Disorders Risk in Spanish. *Procesamiento del Lenguaje
  Natural*, 73, 435–448.
  https://dialnet.unirioja.es/servlet/articulo?codigo=9736877

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N.,
  ... y Kiela, D. (2020). Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing
  Systems (NeurIPS)*, 33, 9459–9474.
  https://arxiv.org/abs/2005.11401

- MentalRiskES@IberLEF 2026 — Web oficial:
  https://sites.google.com/view/mentalriskes2026
