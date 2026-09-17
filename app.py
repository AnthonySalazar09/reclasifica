"""Aplicación local de pruebas con el MobileNetV2 ya entrenado."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json
import time
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import streamlit as st
import torch
from torchvision import transforms
from inference import load_predictor, predict_image, DISPLAY_NAMES
from image_input import decode_image
from chat_ui import render_result_guide, render_chat_widget

ART = ROOT / "artifacts"
st.set_page_config(page_title="ReClasifica · Prueba tus residuos", page_icon="♻️", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.stApp {background:#202224;color:#e3e6e4;color-scheme:dark;}
.block-container {max-width:1180px;padding-top:2rem;padding-bottom:2rem;}
header[data-testid="stHeader"] {background:transparent;}
h1,h2,h3 {letter-spacing:-.035em;color:#e3e6e4;}
h1 {font-size:2.9rem!important;font-weight:750!important;line-height:1.13!important;}
h3 {font-size:1.3rem!important;}
.brand {display:flex;align-items:center;justify-content:space-between;padding-bottom:1.15rem;border-bottom:1px solid #404548;margin-bottom:1.6rem;}
.brand-name {font-weight:750;font-size:1.3rem;letter-spacing:-.04em;color:#d7e9df;}
.brand-icon {background:#335f4b;color:#e4f2ea;border-radius:12px;padding:7px 11px;margin-right:9px;}
.badge {font-size:.73rem;letter-spacing:.09em;text-transform:uppercase;color:#a9d8bd;background:#303d36;padding:7px 12px;border-radius:30px;}
.eyebrow {font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:#9cbca8;margin:0 0 12px;}
.subtitle {color:#b3bcb7;font-size:1.05rem;max-width:630px;line-height:1.6;margin:5px 0 22px;}
.hero-side {background:#2b3430;border:1px solid #3e4b43;border-radius:18px;padding:24px;margin-top:8px;min-height:140px;}
.hero-side strong {display:block;font-size:1.2rem;color:#d7e9df;margin:8px 0;}
.hero-side p {color:#b3bcb7;font-size:.9rem;margin-bottom:0;}
.empty-result {min-height:315px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;border:1px dashed #48544d;border-radius:15px;padding:28px;background:#262a28;}
.empty-icon {font-size:2.4rem;color:#94b8a1;margin-bottom:15px;}
.empty-result h3 {margin:0 0 10px!important;}
.empty-result p {font-size:.95rem;color:#b3bcb7;max-width:290px;}
.result-label {font-size:2.2rem;font-weight:750;color:#8dd5ac;letter-spacing:-.04em;}
.confidence {font-size:1.05rem;color:#bccdc2;margin-bottom:16px;}
.bar-heading {display:flex;justify-content:space-between;font-size:.9rem;margin:13px 0 6px;}
.bar-track {background:#39423d;border-radius:6px;height:9px;overflow:hidden;}
.bar-fill {height:9px;border-radius:6px;background:#83a894;}
.recommendation {background:#2b3830;border-left:3px solid #76b48d;padding:15px 17px;border-radius:0 10px 10px 0;margin-top:22px;font-size:.92rem;line-height:1.5;color:#d4e1d9;}
.foot {border-top:1px solid #404548;margin-top:2rem;padding-top:1rem;font-size:.78rem;color:#a6b1aa;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:16px!important;}
[data-testid="stMetric"] {background:#2b2e31;border:1px solid #454b47;border-radius:13px;padding:16px;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p {color:#aeb8b1!important;}
button[kind="primary"],button[kind="primaryFormSubmit"] {color:#14271c!important;}
.st-key-performance_matrix img,.st-key-performance_curves img {filter:invert(.87) hue-rotate(180deg);border-radius:10px;}
[data-testid="stTabs"] button p {font-size:.97rem;}
@media(max-width:700px){h1{font-size:2.1rem!important;}.block-container{padding-top:1rem;}.badge{font-size:.6rem;}.hero-side{min-height:0;margin:0 0 18px;}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(stamp):
    torch.set_num_threads(4)
    return load_predictor(ART / "model.pt", device="cpu")


def read_metrics():
    try:
        return json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def example_choices():
    # Una imagen de cada clase del test; se señalan como demo, nunca como prueba externa.
    manifest_path = ART / "split_manifest.csv"
    if not manifest_path.exists():
        return {}
    frame = pd.read_csv(manifest_path)
    examples = {}
    for name, label in zip(DISPLAY_NAMES, ["metal", "papel_carton", "plastico"]):
        group = frame[(frame.split == "test") & (frame["class"] == label)]
        if len(group):
            path = ROOT / group.iloc[0].path
            if path.is_file():
                examples[name] = path
    return examples


def download_json(payload, name):
    st.download_button("Descargar resultado · JSON", json.dumps(payload, ensure_ascii=False, indent=2),
                       file_name=name, mime="application/json", width="stretch")


st.markdown('<div class="brand"><div class="brand-name"><span class="brand-icon">♻</span>ReClasifica</div><span class="badge">Laboratorio de reciclaje</span></div>', unsafe_allow_html=True)
hero, aside = st.columns([2.2, 1], gap="large")
with hero:
    st.markdown('<p class="eyebrow">Una foto. Tres materiales.</p>', unsafe_allow_html=True)
    st.title("Dale una segunda mirada a tus residuos.")
    st.markdown('<p class="subtitle">Fotografía un objeto y explora cómo la inteligencia artificial reconoce su material.</p>', unsafe_allow_html=True)
with aside:
    st.markdown('<div class="hero-side"><span class="eyebrow">Qué puedes analizar</span><strong>Metal · Papel/Cartón · Plástico</strong><p>Un objeto a la vez, centrado y con buena luz.</p></div>', unsafe_allow_html=True)

metrics = read_metrics()
if "trial_log" not in st.session_state:
    st.session_state.trial_log = {}
testing_tab, performance_tab, history_tab = st.tabs(["Clasificar una imagen", "Rendimiento del modelo", "Mis pruebas"])

with testing_tab:
    st.write("")
    left, right = st.columns([1.05, 1], gap="large")
    data, file_name, source = None, None, None
    with left:
        with st.container(border=True):
            st.markdown("### 01 · Elige tu imagen")
            mode = st.radio("Origen de la imagen", ["Subir foto", "Usar cámara", "Ver ejemplo"], horizontal=True, label_visibility="collapsed")
            if mode == "Subir foto":
                upload = st.file_uploader("Sube una fotografía", type=["jpg", "jpeg", "png", "webp"],
                                          max_upload_size=10, help="Hasta 10 MB y 20 megapíxeles. JPG, PNG o WebP.")
                if upload:
                    data, file_name, source = upload.getvalue(), upload.name, "foto propia"
                else:
                    st.caption("JPG, PNG o WebP · Máximo 10 MB")
            elif mode == "Usar cámara":
                capture = st.camera_input("Centra el objeto y toma una foto")
                if capture:
                    data, file_name, source = capture.getvalue(), "captura.jpg", "cámara"
            else:
                examples = example_choices()
                example = st.selectbox("Imagen de ejemplo", ["Selecciona un ejemplo", *examples])
                if example in examples:
                    data, file_name, source = examples[example].read_bytes(), examples[example].name, "demo TrashNet"
                st.caption("Los ejemplos pertenecen al test de TrashNet; no son fotografías nuevas.")
            image = None
            if data:
                try:
                    image = decode_image(data)
                except ValueError as error:
                    st.error(str(error))
            if image is not None:
                preview = image.copy()
                preview.thumbnail((800, 440))
                st.image(preview, caption=file_name)
                st.caption(f"{image.width} × {image.height} píxeles · {len(data)/1024:.0f} KB")
            else:
                st.markdown("**Para una mejor prueba**")
                st.markdown("1. Coloca un solo residuo sobre una superficie despejada.\n2. Acércate y deja el objeto en el centro.\n3. Evita sombras fuertes y reflejos.")
            analyze = st.button("Analizar imagen", type="primary", disabled=image is None, width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("### 02 · Explora el resultado")
            digest = hashlib.sha256(data).hexdigest() if image is not None else None
            identity = f"{source}:{digest}" if digest else None
            # No mostrar una predicción anterior si el archivo cambia o se retira.
            if st.session_state.get("active_identity") != identity:
                st.session_state.active_identity = identity
                st.session_state.pop("active_prediction", None)
            if analyze and image is not None:
                try:
                    with st.spinner("Analizando el material…"):
                        checkpoint = ART / "model.pt"
                        model = get_model(checkpoint.stat().st_mtime_ns)
                        start = time.perf_counter()
                        result = predict_image(image, model)
                        elapsed = time.perf_counter() - start
                    st.session_state.active_prediction = {
                        **result, "filename": file_name, "source": source, "image_sha256": digest,
                        "inference_seconds": elapsed, "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                except (OSError, RuntimeError, ValueError, KeyError):
                    st.error("No se pudo cargar o ejecutar el modelo. Revisa que artifacts/model.pt esté disponible y vuelve a intentarlo.")
            result = st.session_state.get("active_prediction")
            if result:
                st.markdown(f'<div class="eyebrow">Material estimado</div><div class="result-label">{html.escape(result["label"])}</div><div class="confidence">{result["confidence"]:.1%} de confianza del modelo</div>', unsafe_allow_html=True)
                for label, probability in sorted(result["probabilities"].items(), key=lambda pair: -pair[1]):
                    color = "#6bc69a" if label == result["label"] else "#83a894"
                    st.markdown(f'<div class="bar-heading"><span>{html.escape(label)}</span><strong>{probability:.1%}</strong></div><div class="bar-track"><div class="bar-fill" style="width:{probability*100:.3f}%;background:{color}"></div></div>', unsafe_allow_html=True)
                tips = {
                    "metal": "Si confirmas que es metal, vacía el envase y consulta dónde reciben este material en tu localidad.",
                    "papel_carton": "Si confirmas que es papel o cartón, mantenlo seco y separa las partes con restos de comida. Consulta las reglas de recolección locales.",
                    "plastico": "Si confirmas que es plástico, revisa el símbolo del envase y si tu punto de recolección acepta ese tipo de material.",
                }
                st.markdown(f'<div class="recommendation"><strong>Orientación para separar</strong><br>{tips[result["class_id"]]}</div>', unsafe_allow_html=True)
                st.caption(f"Procesado en {result['inference_seconds']:.2f} s · Confianza orientativa, no calibrada.")
                if result["confidence"] < 0.70:
                    st.info("La predicción es poco concluyente. Prueba otra toma con el objeto más cerca y mejor iluminado.")
                download_json(result, "resultado_residuo.json")
                render_result_guide(result)
                if source != "demo TrashNet":
                    st.divider()
                    st.markdown("**¿Qué material es realmente?**")
                    truth = st.selectbox("Etiqueta para evaluar tu prueba", ["Sin confirmar", *DISPLAY_NAMES, "Otro material / fuera de alcance"], key=f"truth_{digest}")
                    if st.button("Registrar esta prueba", width="stretch"):
                        record = {**result, "real_label": truth}
                        st.session_state.trial_log[digest] = record
                        st.success("Prueba registrada. Puedes revisarla o descargarla en Mis pruebas.")
                with st.expander("Ver el encuadre que analiza el modelo"):
                    crop = transforms.CenterCrop(224)(transforms.Resize(232)(image))
                    st.image(crop, width=224)
                    st.caption("El modelo analiza este recorte central. Si el objeto queda fuera, toma otra fotografía.")
            else:
                st.markdown('<div class="empty-result"><div class="empty-icon">◎</div><h3>Tu resultado aparecerá aquí</h3><p>Elige una fotografía y pulsa Analizar imagen para comparar los tres materiales.</p></div>', unsafe_allow_html=True)
    st.caption("Alcance: un objeto de metal, papel/cartón o plástico. Ante vidrio, otros materiales o escenas sin residuos, el modelo también elegirá una de estas tres clases. Comprueba visualmente el resultado.")

with performance_tab:
    st.write("")
    st.markdown("### Lo que sabemos del modelo")
    st.write("Resultados medidos en el conjunto de prueba reservado de TrashNet. Tus fotografías se evalúan por separado en Mis pruebas.")
    if metrics:
        a, b, c = st.columns(3)
        a.metric("Exactitud en test", f"{metrics['accuracy']:.2%}")
        b.metric("F1 macro", f"{metrics['macro_f1']:.3f}")
        c.metric("Imágenes de prueba", str(metrics["test_size"]))
        table_col, matrix_col = st.columns([1, 1], gap="large")
        with table_col:
            st.markdown("#### Resultados por material")
            rows = []
            for label in metrics["class_order"]:
                report = metrics["classification_report"][label]
                rows.append({"Material": label, "Precisión": f"{report['precision']:.1%}",
                             "Recall": f"{report['recall']:.1%}", "F1": f"{report['f1-score']:.3f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Precisión: cuántas predicciones de una clase eran correctas. Recall: cuántos objetos de esa clase se detectaron. F1 combina ambas medidas.")
            low, high = metrics["accuracy_wilson_95"]
            st.write(f"Intervalo de confianza del 95 % para la exactitud: **{low:.1%}–{high:.1%}**.")
            st.caption("Intervalo de Wilson; asume observaciones independientes y no mide el desempeño en otros entornos.")
        with matrix_col:
            matrix_file = ART / "figures/confusion_matrix.png"
            if matrix_file.exists():
                with st.container(key="performance_matrix"):
                    st.image(str(matrix_file), caption="Filas: material real · Columnas: predicción")
        with st.expander("Cómo se entrenó y qué falta comprobar"):
            st.markdown("MobileNetV2 preentrenada en ImageNet, adaptada a tres materiales. Se usaron 1.200 imágenes balanceadas: 840 de entrenamiento, 180 de validación y 180 de prueba. Se entrenó la cabeza y se ajustaron las últimas capas; el checkpoint se eligió por validación.")
            st.markdown("TrashNet contiene principalmente fondos blancos. Los aumentos de imágenes no garantizan buen rendimiento en una cocina o en la calle. La auditoría por similitud también puede descartar fotos distintas; está documentada en el notebook.")
            curve_file = ART / "figures/learning_curves.png"
            if curve_file.exists():
                with st.container(key="performance_curves"):
                    st.image(str(curve_file))
            st.markdown("Fuente de datos: [TrashNet, Gary Thung y Mindy Yang](https://github.com/garythung/trashnet).")
    else:
        st.info("No se encontró el archivo de métricas. La clasificación sigue disponible si están los pesos del modelo.")

with history_tab:
    st.write("")
    st.markdown("### Tu pequeño experimento")
    st.write("Registra fotos propias y confirma su material para comparar la predicción con la realidad.")
    trials = list(st.session_state.trial_log.values())
    if trials:
        confirmed = [r for r in trials if r["real_label"] in DISPLAY_NAMES]
        outside = sum(r["real_label"] == "Otro material / fuera de alcance" for r in trials)
        a, b, c = st.columns(3)
        a.metric("Fotos registradas", len(trials))
        b.metric("Fotos confirmadas de las 3 clases", len(confirmed))
        c.metric("Aciertos en fotos confirmadas", f"{sum(r['label']==r['real_label'] for r in confirmed)/len(confirmed):.1%}" if confirmed else "—")
        if outside:
            st.caption(f"{outside} fotos fuera del alcance: excluidas del porcentaje de aciertos de las tres clases.")
        rows = [{"Archivo": r["filename"], "Predicción": r["label"], "Confianza": r["confidence"],
                 "Material real": r["real_label"], "Coincide": "Sí" if r["label"] == r["real_label"] else "No" if r["real_label"] in DISPLAY_NAMES else "Sin evaluar",
                 "Fecha UTC": r["timestamp"]} for r in trials]
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, width="stretch", column_config={"Confianza": st.column_config.NumberColumn(format="percent")})
        st.download_button("Descargar mis pruebas · CSV", table.to_csv(index=False).encode("utf-8-sig"),
                           file_name="mis_pruebas_residuos.csv", mime="text/csv")
    else:
        st.info("Todavía no has registrado pruebas. Analiza una foto propia y pulsa Registrar esta prueba.")
    st.caption("El registro vive solo en esta sesión y puede perderse al recargar o cerrar. Descarga el CSV para conservarlo. Las imágenes no se guardan en disco ni se usan para reentrenar.")

st.markdown('<div class="foot">ReClasifica · Proyecto de Fundamentos de Inteligencia Artificial · Anthony Salazar</div>', unsafe_allow_html=True)
render_chat_widget(st.session_state.get("active_prediction"))
