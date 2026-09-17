# Publicar ReClasifica

## 1. Crear la aplicación

En https://share.streamlit.io pulsa **Create app**, y elige **Yup, I have an app** (desplegar una aplicación existente).

| Campo | Valor |
| --- | --- |
| Repository | `AnthonySalazar09/reclasifica` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL (opcional) | `reclasifica-anthony` si está disponible |

En **Advanced settings**, selecciona **Python 3.11**.

Si el repositorio privado no aparece, revisa que Streamlit tenga acceso a repositorios privados en tu conexión de GitHub. No es necesario volver público el repositorio.

## 2. Activar el chatbot

En **Advanced settings → Secrets**, pega lo siguiente y reemplaza el texto de ejemplo por tu clave real, conservando las comillas:

```toml
OPENAI_API_KEY = "PEGA_TU_CLAVE_AQUI"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
CHAT_MAX_CALLS_PER_SESSION = "40"
```

Usa la clave que ya configuraste localmente en `Proyecto_Final_Residuos/.env`, en la variable `OPENAI_API_KEY`. No copies todo ese archivo y no envíes la clave por chat. Puedes configurar los secretos después desde **Settings → Secrets**.

El clasificador funciona sin esa clave. El chatbot requiere créditos de OpenAI y su consumo se carga a la cuenta de la clave. Las 40 llamadas son por sesión, no un límite global.

## 3. Desplegar y compartir

Pulsa **Save**, después **Deploy**. Espera a que termine la instalación. No necesitas mantener tu computadora encendida ni comprar un dominio: Streamlit asigna una dirección `*.streamlit.app`.

Al proceder de un repositorio privado, la aplicación empieza privada. Para compartirla libremente con tus compañeros, entra en **Settings → Sharing → Who can view this app** y selecciona **This app is public and searchable**. Esto publica la aplicación, pero conserva el repositorio de GitHub privado. Como alternativa puedes limitar el acceso a personas específicas.

Abre el enlace en una ventana privada o en otro dispositivo y comprueba:

1. Clasifica los tres ejemplos de la aplicación.
2. Carga una foto propia y revisa las probabilidades.
3. Abre la pestaña de rendimiento.
4. Comprueba una pregunta de reciclaje y una pregunta fuera del tema en el chatbot.

Si falla, revisa **Manage app → logs**. No compartas capturas que muestren la clave.

## Documentación oficial

- Despliegue: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secretos: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
- Privacidad y enlace: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
