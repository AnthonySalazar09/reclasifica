"""Inferencia reutilizable por el notebook y la futura aplicación, sin descargas."""
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image, ImageOps

CLASSES = ["metal", "papel_carton", "plastico"]
DISPLAY_NAMES = ["Metal", "Papel/Cartón", "Plástico"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_model(weights=None):
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, 3))
    return model


def evaluation_transform():
    # Igual al preprocesamiento oficial de MobileNetV2 IMAGENET1K_V2.
    return transforms.Compose([
        transforms.Resize(232), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def load_predictor(checkpoint_path=None, device="cpu"):
    checkpoint_path = checkpoint_path or Path(__file__).parent / "artifacts/model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("El orden de clases del checkpoint no coincide.")
    if checkpoint["preprocessing"] != "mobilenet_v2_imagenet1k_v2":
        raise ValueError("Preprocesamiento desconocido.")
    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_image(image, model):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
    elif isinstance(image, Image.Image):
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError("Proporciona una ruta o una imagen PIL.")
    device = next(model.parameters()).device
    tensor = evaluation_transform()(rgb).unsqueeze(0).to(device)
    scores = model(tensor).softmax(dim=1)[0].cpu().tolist()
    index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "class_id": CLASSES[index], "label": DISPLAY_NAMES[index],
        "confidence": scores[index],
        "probabilities": dict(zip(DISPLAY_NAMES, scores)),
        "note": "Probabilidades softmax no calibradas; el modelo solo conoce tres clases.",
    }
