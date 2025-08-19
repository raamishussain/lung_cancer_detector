import logging
import torch

from app.config import REMOTE_WEIGHT_PATH
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model():
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=REMOTE_WEIGHT_PATH,
    )
    return model


def run_inference(image: Image, model: torch.nn.Module) -> Image:
    """Run object detection on an image

    Args:
        image: PIL.Image.Image
            Image bytes of chest CT/PET-CT scan

    Returns:
        PIL.Image.Image:
            Image with bounding box predictions
    """
    results = model(image)

    # annotate images with bounding boxes
    annotated = results.render()[0]
    return Image.fromarray(annotated)
