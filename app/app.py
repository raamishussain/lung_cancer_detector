import io
import logging
import modal
import torch

from app.config import REMOTE_WEIGHT_PATH, VOLUME_NAME
from app.inference import run_inference
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image = modal.Image.from_dockerfile("Dockerfile")
model_volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("lung_cancer_detector")

def load_model():
    """Load YOLOv5 model with trained model weights"""
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=REMOTE_WEIGHT_PATH
    )
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan will load model on app startup"""
    logger.info("Loading YOLOv5 model weights...")
    app.state.model = load_model()
    logger.info("Model loaded successfully")
    yield


@app.function(image=image, volumes={"/model": model_volume}, max_containers=1)
@modal.asgi_app()
def fastapi_app():
    """Create FastAPI application with a /predict endpoint to run
    inference on images"""

    fastapi_app = FastAPI(lifespan=lifespan)

    @fastapi_app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> Response:
        model = fastapi_app.state.model
        
        logger.info("Running inference on image...")
        img = Image.open(io.BytesIO(await file.read()))

        try:
            annotated_img = run_inference(img, model)
            logger.info("Successfully ran YOLO model")

            buff = io.BytesIO()
            annotated_img.save(buff, format="PNG")
            buff.seek(0)

            return Response(content=buff.read(), media_type="image/png")
        except Exception as e:
            logger.error(f"Encountered exception when running inference: {e}")
            return HTTPException(
                status_code=400,
                detail=f"Encountered exception when running inference {e}"
            )

    return fastapi_app
