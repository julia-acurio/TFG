# TFG — Ranking de Respuestas Terapéuticas en Conversaciones de Apoyo en Salud Mental

**Autora:** Julia Acurio Ajila  
**Universidad:** Universidad Rey Juan Carlos  
**Tutoras:** Natalia Madrueño Sierro y Soto Montalvo Herranz

---

## Descripción

Este Trabajo de Fin de Grado estudia y compara sistemas de ranking para la 
selección de respuestas candidatas en conversaciones de apoyo en salud mental, 
utilizando los upvotes de CounselChat como señal débil de preferencia dentro 
del marco experimental.

Dado el mensaje de un paciente y un conjunto variable de respuestas de 
terapeutas certificados, el sistema rankea las respuestas según su calidad.

---

## Dataset

**CounselChat** (Bertagnolli, 2020): plataforma online donde terapeutas 
certificados responden preguntas de pacientes. Las respuestas incluyen 
valoraciones de la comunidad (upvotes) que se usan como señal de calidad.

- 2.775 ejemplos
- 940 preguntas únicas
- 31 temas clínicos
- Disponible en: https://huggingface.co/datasets/nbertagnolli/counsel-chat

---

## Métricas de evaluación

- **Precision@1:** ¿el sistema identifica correctamente la mejor respuesta?
- **MRR (Mean Reciprocal Rank):** posición media de la respuesta correcta en el ranking

---

## Estructura del repositorio
├── data/             # Dataset y corpus de prueba
├── docs/             # Documentación
├── knowledge_base/   # Documentos clínicos para RAG
├── notebooks/        # Experimentos y análisis
├── reports/          # Resultados de los experimentos
├── results/          # Resultados intermedios
└── src/              # Código fuente

---

## Referencias

- Bertagnolli, N. (2020). Counsel Chat: Bootstrapping High-Quality Therapy Data.
- Ahmed et al. (2024). Analysis of Mental Health Counseling Conversation Using NLP. Journal of Computer Science, 20(3), 303-309.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
