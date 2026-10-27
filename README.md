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

## Instrucciones de Ejecución y Despliegue Web

### 1. Instalación de Dependencias
Asegúrate de instalar los requerimientos ejecutando en tu terminal:
```bash
pip install -r requirements.txt
```

### 2. Ejecución del Servidor Local
Para iniciar la aplicación web en tu computadora:
```bash
python app.py
```
Abre en tu navegador: **`http://localhost:5000`**

### 3. Acceso desde el Móvil mediante Cloudflare Tunnel (HTTPS)
Para utilizar la cámara en tiempo real desde tu teléfono móvil sin bloqueos de seguridad del navegador:
1. Mantén corriendo `python app.py` en tu terminal principal.
2. Abre una segunda terminal y ejecuta el túnel seguro:
   ```bash
   ./cloudflared tunnel --url http://localhost:5000
   ```
3. Copia el enlace HTTPS generado (ejemplo: `https://xxxx.trycloudflare.com`) y ábrelo en el navegador de tu celular.

---

## Estructura y Mapa de Recursos

| Recurso / Archivo | Descripción de Contenido | Propósito en el Proyecto |
| :--- | :--- | :--- |
| `app.py` | Aplicación web en Flask con integración PyTorch. | Servidor backend para inferencia web, subida de imágenes y cámara en vivo. |
| `templates/index.html` | Interfaz web responsiva en modo oscuro. | Interfaz de usuario para carga de imágenes y streaming de cámara. |
| `detextar_grietas.py` | Script principal de declaración de clases. | Configura datos, entrena modelo CNN y exporta pesos. |
| `generar_graficas.py` | Script analítico de métricas y evaluación. | Evalúa el subset de prueba (15%) y genera la matriz visual. |
| `modelo_grietas.pth` | Archivo binario con pesos optimizados. | Permite despliegue e inferencia rápida sin reentrenar. |

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

## Interfaz Web y Características de la Aplicación
La aplicación cuenta con una interfaz web responsiva desarrollada en diseño moderno de tonos oscuros (Dark Mode), optimizada tanto para computadoras como para dispositivos móviles. Sus funcionalidades principales son:
* **Carga de Imágenes:** Permite arrastrar o seleccionar fotografías de muros, pavimentos o columnas para su análisis instantáneo.
* **Cámara en Vivo:** Integración con la cámara del dispositivo (con opción de alternar entre cámara trasera y delantera en móviles) con indicador visual de escaneo.
* **Clasificación de Peligrosidad:** El sistema evalúa el nivel de certeza y el área afectada para clasificar automáticamente entre:
  * Estructura Sana (Sin riesgo)
  * Grieta Leve / No Peligrosa (Superficial / Estética)
  * Grieta Severa / Peligrosa (Riesgo Estructural Alto)
* **Dictamen y Recomendación Técnica:** Informe detallado que indica la posible causa raíz (asentamientos, sobrecargas, contracción por fraguado) y los pasos profesionales recomendados.
