# Proyecto Final: Detección y Clasificación de Motocicletas Fáciles de Manejar


Este repositorio contiene los experimentos y aprendizajes realizados para el proyecto final del **Bootcamp de Colombia Tech**. El proyecto se centra en el desarrollo de un modelo de visión artificial capaz de reconocer y clasificar motocicletas de una subcategoría específica ("fáciles de manejar") a partir de fotografías.

## Estructura del Repositorio

A continuación, se detalla el contenido de cada una de las carpetas principales del proyecto:

### `API/`

Contiene la implementación del modelo entrenado a través de una API RESTful, desarrollada con **FastAPI**. Esto permite que el modelo pueda ser consumido fácilmente por otras aplicaciones o servicios.

### `limpieza_yolo/`

En esta carpeta se encuentra el código utilizado para la fase de **limpieza y preprocesamiento de datos**. Se aprovechó la capacidad de detección de objetos de **YOLO (You Only Look Once)** para identificar y filtrar las imágenes de motocicletas de manera eficiente.

### `notebooks/`

Aquí se alojan los cuadernos de Jupyter donde se llevó a cabo el **entrenamiento y la validación** de los diferentes modelos. Es el lugar donde se documentan los experimentos, las métricas de rendimiento y las decisiones tomadas durante el desarrollo.

### `segmentacion_YOLO/`

Esta carpeta contiene el código que se utilizó para obtener las **máscaras de segmentación** necesarias. Este proceso fue crucial para el entrenamiento de los modelos de segmentación, permitiendo aislar la motocicleta del fondo en las imágenes.

### `Tag_grah/`

Una herramienta de etiquetado desarrollada para este proyecto. Permite evaluar y etiquetar las fotografías de forma colaborativa a través de un sistema de votación que muestra **6 fotografías en simultáneo**, agilizando el proceso de curación de datos.

## Tecnologías Utilizadas

-   **Python**: Lenguaje de programación principal.
    
-   **FastAPI**: Framework web para la creación de la API.
    
-   **YOLO**: Modelo de detección de objetos.
    
-   **Jupyter Notebooks**: Para el desarrollo, experimentación y documentación.
    

## Cómo Empezar

Para utilizar este proyecto, clona el repositorio e instala las dependencias necesarias. Las instrucciones detalladas para cada módulo se encuentran dentro de sus respectivas carpetas.