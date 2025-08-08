import io
import torch

from app.config import REMOTE_WEIGHT_PATH
from PIL import Image


def load_model():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=REMOTE_WEIGHT_PATH,
    )
    return model


def run_inference(image_bytes: bytes, model: torch.nn.Module) -> Image:
    """Run object detection on an image

    Args:
        image_bytes: bytes
            Image bytes of chest CT/PET-CT scan

    Returns:
        Image:
            Image with bounding box predictions
    """

    image = Image.open(io.BytesIO(image_bytes))
    results = model(image)

    # annotate images with bounding boxes
    annotated = results.render()[0]
    return Image.fromarray(annotated)
