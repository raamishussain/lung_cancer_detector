import logging

from app.config import VOLUME_NAME
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


@app.function(image=image, volumes={"/model": model_volume}, max_containers=1)
@asgi_app()
def web():
    """Build the Gradio front end UI"""
    import gradio as gr
    import PIL
    from app.inference import load_model, run_inference
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from typing import Optional


    def toggle_submit(img):
        """Returns False if img is None. This disables submit button"""
        return img is not None


    def predict(img: Optional[PIL.Image.Image]) -> Optional[PIL.Image.Image]:
        if img is None:
            logger.warning("No Image provided")
            return None

        model = load_model()
        logger.info("Loaded YOLO model, running inference...")
        try:
            return run_inference(img, model)
        except Exception as e:
            logger.error(f"Exception: {e}")

    with gr.Blocks(title="Lung Cancer Detector", fill_width=True) as ui:

        gr.Markdown(
            """
            # Lung Cancer Detector

            Upload a chest PET/CT scan to detect tumors
            """
        )

        with gr.Row():
            with gr.Column():
                input_img = gr.Image(
                    type="pil",
                    label="Upload Scan",
                    interactive=True,
                )
                submit = gr.Button("Analyze", interactive=True)

            with gr.Column():
                output_img = gr.Image(
                    type="pil",
                    label="Scan with Detected Tumors",
                    interactive=False,
                )

        input_img.change(
            fn=toggle_submit, inputs=[input_img], outputs=[submit]
        )
        submit.click(
            fn=predict,
            inputs=[input_img],
            outputs=[output_img],
        )

    ui.queue(max_size=5)

    fastapi_app = FastAPI()

    mounted_app = mount_gradio_app(app=FastAPI(), blocks=ui, path="/")

    for route in fastapi_app.routes:
        try:
            logger.info(f"Route: {route.path} -> {route.name}")
        except AttributeError:
            pass

    return mounted_app
