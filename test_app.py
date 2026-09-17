"""Pruebas del flujo real de la aplicación, sin reentrenar el modelo."""
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
import unittest
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from PIL import Image
from streamlit.testing.v1 import AppTest
from image_input import decode_image
from chat_service import ChatSettings


class Upload(BytesIO):
    def __init__(self, data, name="prueba.jpg"):
        super().__init__(data)
        self.name = name


def button(app, label):
    return next(b for b in app.button if b.label == label)


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        # Las pruebas no consumen una clave real aunque ya esté configurada.
        self.chat_settings = patch("chat_ui.settings_now", return_value=ChatSettings())
        self.chat_settings.start()
        self.addCleanup(self.chat_settings.stop)

    def new_app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    def assert_ok(self, app):
        self.assertEqual(len(app.exception), 0, str(app.exception))

    def test_empty_and_all_three_examples(self):
        app = self.new_app().run()
        self.assert_ok(app)
        self.assertTrue(button(app, "Analizar imagen").disabled)
        app.radio[0].set_value("Ver ejemplo").run()
        for label in ["Metal", "Papel/Cartón", "Plástico"]:
            app.selectbox[0].set_value(label).run()
            self.assertIsNone(app.session_state.filtered_state.get("active_prediction"))
            button(app, "Analizar imagen").click().run()
            self.assert_ok(app)
            result = app.session_state["active_prediction"]
            self.assertEqual(result["source"], "demo TrashNet")
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=5)
            self.assertEqual(len(app.session_state["trial_log"]), 0)
        app.selectbox[0].set_value("Selecciona un ejemplo").run()
        self.assert_ok(app)
        self.assertIsNone(app.session_state.filtered_state.get("active_prediction"))
        self.assertTrue(button(app, "Analizar imagen").disabled)

    def test_upload_register_update_and_invalid_file(self):
        data = (ROOT / "examples/metal34.jpg").read_bytes()
        with patch("streamlit.file_uploader", return_value=Upload(data)) as uploader:
            app = self.new_app().run()
            button(app, "Analizar imagen").click().run()
            self.assert_ok(app)
            self.assertEqual(app.session_state["active_prediction"]["source"], "foto propia")
            app.selectbox[0].set_value("Metal").run()
            button(app, "Registrar esta prueba").click().run()
            self.assertEqual(len(app.session_state["trial_log"]), 1)
            app.selectbox[0].set_value("Otro material / fuera de alcance").run()
            button(app, "Registrar esta prueba").click().run()
            self.assert_ok(app)
            self.assertEqual(len(app.session_state["trial_log"]), 1)
            self.assertEqual(next(iter(app.session_state["trial_log"].values()))["real_label"], "Otro material / fuera de alcance")
            self.assertEqual(app.metric[-1].value, "—")
            uploader.return_value = Upload(b"not an image", "incorrecta.png")
            app.run()
            self.assert_ok(app)
            self.assertGreater(len(app.error), 0)
            self.assertIsNone(app.session_state.filtered_state.get("active_prediction"))

    def test_decoder_formats_orientation_and_limits(self):
        for mode, format in [("RGB", "JPEG"), ("RGBA", "PNG"), ("L", "PNG"), ("RGB", "WEBP")]:
            buffer = BytesIO()
            Image.new(mode, (40, 25)).save(buffer, format=format)
            image = decode_image(buffer.getvalue())
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (40, 25))
        image = Image.new("RGB", (40, 25))
        exif = image.getexif()
        exif[274] = 6
        buffer = BytesIO()
        image.save(buffer, format="JPEG", exif=exif)
        self.assertEqual(decode_image(buffer.getvalue()).size, (25, 40))
        for bad in [b"", b"bad", b"x" * (10 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                decode_image(bad)
        buffer = BytesIO()
        Image.new("RGB", (100, 100)).save(buffer, format="PNG")
        with patch("image_input.MAX_PIXELS", 100):
            with self.assertRaises(ValueError):
                decode_image(buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
