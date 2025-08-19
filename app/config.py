# flake8: noqa

VOLUME_NAME = "lung-cancer-detector"
REMOTE_WEIGHT_PATH = "/model/best.pt"

DESCRIPTION = """
An API which uses a YOLOv5 object detection model to detect tumors in lung CT/PET scans.

Upload a `png` file of a CT/PET scan in the `/predict` endpoint to get started.
"""
