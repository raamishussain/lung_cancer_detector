import gradio as gr
import logging

from app.config import VOLUME_NAME
from app.inference import load_model, run_inference
from modal import (
    App,
    asgi_app,
    Image,
    Volume,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = Image.from_dockerfile("Dockerfile")
model_volume = Volume.from_name(VOLUME_NAME)
app = App("lung_cancer_detector", image=image)


@app.function(image=image, volumes={"/model": model_volume})
@asgi_app()
def web():
    """Gradio front end UI"""

    def predict(img):
        model = load_model()
        return run_inference(img, model)

    with gr.Blocks(title="Lung Cancer Detector", fill_width=True) as ui:

        gr.Markdown(
            """
            # Lung Cancer Detector

            Upload a chest PET/CT scan to detect tumors
            """
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="pil", label="Upload Scan")
                submit = gr.Button("Analyze")

            with gr.Column():
                output_img = gr.Image(
                    type="pil", label="Scan with Detected Tumors"
                )

        submit.click(
            fn=predict,
            inputs=input_img,
            outputs=output_img,
        )

    return ui
