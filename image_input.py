"""Validación común para archivos subidos y capturas de cámara."""
from io import BytesIO
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Selecciona una imagen de hasta 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as im:
                if im.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Usa una imagen JPG, PNG o WebP.")
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError("La imagen supera 20 megapíxeles. Reduce su resolución e inténtalo de nuevo.")
                if getattr(im, "is_animated", False):
                    raise ValueError("Usa una imagen estática, no una animación.")
                im.load()
                return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("No se pudo abrir la imagen. Comprueba el archivo y vuelve a intentarlo.") from error
