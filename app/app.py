import gradio as gr
import logging

from app.config import VOLUME_NAME
from app.inference import run_inference
from modal import (
    App,
    asgi_app,
    Image,
    mount,
    Volume,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up app
app = App("Lung Cancer Detector")
model_volume = Volume.persisted(VOLUME_NAME)

image = Image.from_dockerfile(
    "Dockerfile",
    context_mount=mount(".", "/app").mount(model_volume, "/model"),
)


@app.function(image=image, mounts=[model_volume])
@asgi_app
def web():
    """Gradio front end UI"""

    def predict(img):
        return run_inference(img)

    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Image(type="pil"),
        title="Lung Cancer Detector",
        description="Upload a chest PET/CT scan to detect tumors",
    )
    return iface.app
