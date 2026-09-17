# ♻️ ReClasifica

Aplicación académica de clasificación de residuos con **MobileNetV2** y aprendizaje por transferencia. Reconoce **metal, papel/cartón y plástico** a partir de una fotografía. Incluye probabilidades por clase, métricas de evaluación, registro temporal de pruebas y un asistente de reciclaje opcional con OpenAI.

## Modelo y resultados

- Modelo entrenado: `artifacts/model.pt` (aproximadamente 9,1 MB).
- Inferencia en CPU, sin descargar pesos adicionales ni reentrenar.
- 1.200 imágenes de TrashNet: 400 por clase; partición 840/180/180.
- Prueba interna: **95,56 % de exactitud**, **0,9553 de F1 macro**, 180 imágenes.
- Los tres ejemplos incluidos pertenecen al conjunto de prueba de TrashNet; no constituyen una evaluación con fotografías externas.
- El clasificador siempre elige entre tres clases: no detecta «ninguno de los anteriores». Las probabilidades softmax no están calibradas. Los fondos y objetos distintos del dataset pueden reducir el rendimiento.

El modelo, su preprocesamiento y las métricas se conservan sin modificaciones respecto al proyecto entrenado. El manifiesto de esta distribución solo enumera las tres imágenes de ejemplo; no es el manifiesto completo del entrenamiento.

SHA256 del modelo:

```text
0f7e32fe697e503a62e70e6032a206247928ef54d31be412a7e56d9a8787ffc5
```

## Ejecutar

Con Python 3.11:

```bash
python -m venv .venv
# Activa el entorno virtual según tu sistema operativo.
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Las dependencias de PyTorch son de CPU. El entorno original de entrenamiento con GPU se conserva aparte.

## Publicar en Streamlit Community Cloud

Consulta [PUBLICAR_STREAMLIT.md](PUBLICAR_STREAMLIT.md). Repositorio: `AnthonySalazar09/reclasifica`, rama: `main`, archivo: `app.py`, Python: **3.11**.

La aplicación funciona sin clave de OpenAI: mantiene la clasificación y las orientaciones locales. Para activar las respuestas generadas, configura la clave en **Advanced settings → Secrets** o **Settings → Secrets**. No la incluyas en GitHub.

## Asistente y tratamiento de datos

El asistente recibe el texto de las preguntas y el resultado numérico del clasificador; no recibe la fotografía ni su nombre. Las fotografías y el historial de pruebas se manejan en la sesión, sin guardarse en archivos por esta aplicación. Las conversaciones están limitadas a reciclaje y explicación del resultado. Las respuestas del asistente pueden ser inexactas; consulta las indicaciones del gestor local.

El límite de 40 solicitudes de OpenAI por sesión también incluye las orientaciones automáticas. No es un límite global de gasto; nuevas sesiones tienen su propio contador. El alojamiento gratuito de Streamlit no incluye el consumo de la API de OpenAI.

## Comprobaciones

```bash
python -m unittest test_app test_chat -v
```

Estas pruebas usan el modelo real para la clasificación y simulan las llamadas a OpenAI, sin consumir créditos.

## Créditos

Dataset **TrashNet**, de Gary Thung y Mindy Yang: https://github.com/garythung/trashnet. Las tres fotografías de ejemplo provienen de su versión redimensionada; se conserva su licencia MIT en [THIRD_PARTY_LICENSES/TrashNet_LICENSE.txt](THIRD_PARTY_LICENSES/TrashNet_LICENSE.txt).

La arquitectura MobileNetV2 y su implementación proceden de PyTorch/Torchvision. Los pesos exportados corresponden al clasificador adaptado en este proyecto académico.
