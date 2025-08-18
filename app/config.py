# flake8: noqa

import os

VOLUME_NAME = "lung-cancer-detector"
REMOTE_WEIGHT_PATH = "/model/model/best.pt"
API_KEY = os.getenv("LUNG_CANCER_API_KEY")

DESCRIPTION = """
An API which uses a YOLOv5 object detection model to detect tumors in lung CT/PET scans.

Upload a `png` file of a CT/PET scan in the `/predict` endpoint to get started.
"""
