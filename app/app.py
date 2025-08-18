import io
import logging
import modal
import torch

from app import __version__
from app.config import API_KEY, DESCRIPTION, REMOTE_WEIGHT_PATH, VOLUME_NAME
from app.inference import run_inference
from contextlib import asynccontextmanager
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Security,
    UploadFile,
)
from fastapi.responses import Response
from fastapi.security.api_key import APIKey, APIKeyQuery
from PIL import Image
from starlette import status


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with trained model weights
image = modal.Image.from_dockerfile("Dockerfile")
model_volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App("lung_cancer_detector")


def load_model():
    """Load YOLOv5 model with trained model weights"""
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=REMOTE_WEIGHT_PATH
    )
    return model


# set up security
api_key_query = APIKeyQuery(name="api-key", auto_error=False)


async def get_api_key(api_key_query: str = Security(api_key_query)):
    if api_key_query == API_KEY:
        return api_key_query
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )


# lifespan will load model on app startup once instead of per request
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

    fastapi_app = FastAPI(
        title="Lung Cancer Detector",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    @fastapi_app.post("/predict")
    async def predict(
        file: UploadFile = File(...), api_key: APIKey = Depends(get_api_key)
    ) -> Response:
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
                detail=f"Encountered exception when running inference {e}",
            )

    return fastapi_app
