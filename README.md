# Sistema de análisis de textos

Este repositorio contiene el código fuente de un sistema de análisis de sentimientos en español desarrollado utilizando la biblioteca FastAPI de Python. El sistema permite a los usuarios ingresar texto y obtener análisis de sentimiento tanto para comentarios individuales como para lotes de comentarios junto con su contexto. Además, incluye la detección de discurso de odio en un contexto dado.

***Funcionalidades implementadas***

  - **Análisis de Sentimientos Individual**: Los usuarios pueden ingresar un comentario y obtener su análisis de sentimiento correspondiente, que incluye el sentimiento y la puntuación asociada.
  
  - **Análisis de Sentimientos de comentarios**:El sistema puede procesar múltiples comentarios simultáneamente, proporcionando análisis de sentimientos para grupos de comentarios junto con etiquetas y probabilidades de discurso de odio.
  
  - **Análisis de Sentimientos de comentarios con Contexto**: Se implementa la capacidad de detectar un comnetario dentro de un contexto específico, lo que permite una evaluación más completa.

***Instrucciones de ejecución***

1. Clona este repositorio en tu máquina local utilizando Git:

 ```
  git clone https://github.com/IvonneSierra/AnalizarTexto.git
  ```
2. Accede al directorio del proyecto:

  ```
  cd apimodelo
  ```
3. Instala las dependencias del proyecto utilizando pip:

  ```
  pip install -r requirements.txt
  ```
4. Ejecuta la aplicación utilizando el siguiente comando:

  ```
  python -m uvicorn main:app --port 8000 --reload
  ```
5. Abre tu navegador web y accede a la siguiente dirección URL:

  ```
  http://localhost:8000/
  ```
6. ¡Disfruta de la aplicación! Ingresa texto para análisis de sentimientos individual o selecciona la opción para analizar lotes de comentarios.


Desarrollado por: Andrea Cubillos, Ivonne Sierra y Santiago Lozano.
