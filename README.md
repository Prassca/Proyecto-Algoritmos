# Informe de Avance: Sistema Automático de Detección de Grietas mediante Redes Neuronales Convolucionales (CNN)

Repositorio oficial para el registro, desarrollo y evaluación del proyecto de visión artificial enfocado en la detección de anomalías estructurales en superficies.

## Datos del Proyecto y Autores
* **Asignatura:** Algoritmos y Programación
* **Docente:** Jheyston Serrano
* **Estudiantes:** 
  * Kheylor Daniel Prasca Pedraza (2264075)
  * Juan David Rojas Borray (2264163)
* **Programa:** Ingeniería en Inteligencia Artificial (Segundo Semestre)
* **Institución:** Universidad Industrial de Santander (UIS) - Barrancabermeja

---

## Estructura y Mapa de Recursos

| Recurso / Archivo | Descripción de Contenido | Propósito en el Proyecto |
| :--- | :--- | :--- |
| `detextar_grietas.py` | Script principal de declaración de clases. | Configura datos, entrena modelo CNN y exporta pesos. |
| `generar_graficas.py` | Script analítico de métricas y evaluación. | Evalúa el subset de prueba (15%) y genera la matriz visual. |
| `modelo_grietas.pth` | Archivo binario con pesos optimizados. | Permite despliegue e inferencia rápida sin reentrenar. |
| `matriz.png` | Figura de evaluación exportada. | Documentación visual de resultados. |

---

## Dataset y Preprocesamiento
El modelo fue entrenado con el conjunto de datos Surface Crack Detection, compuesto por un total de 40,000 imágenes de superficies de concreto, dividido equitativamente entre clases positivas y negativas (20,000 imágenes por clase). Los datos se particionaron de la siguiente manera:
* Entrenamiento: 70%
* Validación: 15%
* Prueba: 15%

---

## Arquitectura del Modelo Computacional
El sistema utiliza una arquitectura de red neuronal profunda que consta de dos bloques principales:
1. **Extractor de Características:** Tres bloques convolucionales (Conv2d) con tamaños de canal de 32, 64 y 128. Cada capa está seguida por normalización por lotes (BatchNorm2d), función de activación ReLU y reducción espacial (MaxPool2d).
2. **Red Clasificadora:** Las características se aplanan y pasan por una capa completamente conectada de 256 neuronas, utilizando un Dropout del 0.5 para regularización.

Se implementó el optimizador Adam con una tasa de aprendizaje de 0.0001, minimizando la función de pérdida de entropía cruzada binaria con logits (BCEWithLogitsLoss).

---
