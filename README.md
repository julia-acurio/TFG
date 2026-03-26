# MentalRiskES 2026 — Task 2: Therapist Response Selection

**Trabajo de Fin de Grado**
Autora: Julia Acurio Ajila

---

## Descripción

Sistema de selección automática de respuestas terapéuticas basado en
**RAG (Retrieval-Augmented Generation)** para la tarea 2 de la competición
MentalRiskES@IberLEF 2026.

Dado el mensaje más reciente de un paciente y tres respuestas candidatas
del terapeuta, el sistema selecciona automáticamente la más adecuada según
las buenas prácticas clínicas, apoyándose en una base de conocimiento de
literatura terapéutica especializada.

---

## Enfoque

El sistema combina tres componentes principales:

1. **Base de conocimiento clínico**: documentos de terapia cognitivo-conductual,
   entrevista motivacional y comunicación terapéutica indexados como vectores.
2. **Recuperador semántico**: dado el mensaje del paciente, recupera los
   fragmentos más relevantes de la base de conocimiento.
3. **LLM como selector**: un modelo de lenguaje recibe el contexto de la
   conversación, los fragmentos recuperados y las 3 opciones, y selecciona
   la más apropiada.


---

## Referencias
