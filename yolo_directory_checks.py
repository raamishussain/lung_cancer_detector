import os
import numpy as np

from pathlib import Path


def check_for_duplicates(image_dir: str) -> np.ndarray:
    """Given a directory of images and labels in the YOLO
    format, this function checks if there are any duplicates
    between the training and validation sets.

    Args:
        image_dir (str): name of image directory

    Returns:
        duplicates (array): numpy array containing names of
            any duplicated files. Empty list means no duplicates
            are present

    """

    train_files = os.listdir(Path(image_dir, "train"))
    val_files = os.listdir(Path(image_dir, "val"))

    # split file suffix to get file names
    train_filenames = np.array([Path(file).stem for file in train_files])
    val_filenames = np.array([Path(file).stem for file in val_files])

    mask = np.isin(train_filenames, val_filenames)
    duplicates = train_filenames[mask]

    return duplicates


def valid_image_label_mapping(image_dir: str, label_dir: str) -> bool:
    """Given YOLO image and label directories, check
    if each image has a corresponding label.

    Args:
        image_dir (str): name of first directory
        label_dir (str): name of second directory

    Returns:

    """

    image_train_files = os.listdir(Path(image_dir, "train"))
    label_train_files = os.listdir(Path(label_dir, "train"))

    image_val_files = os.listdir(Path(image_dir, "val"))
    label_val_files = os.listdir(Path(label_dir, "val"))

    # first there should be the same number of files in each directory
    train_error = (
        "Training images and labels must have the same number of files"
    )
    assert len(image_train_files) == len(label_train_files), train_error

    val_error = (
        "Validation images and labels must have the same number of files"
    )
    assert len(image_val_files) == len(label_val_files), val_error

    # split file suffix to get address ids
    image_train_filenames = [Path(file).stem for file in image_train_files]
    label_train_filenames = [Path(file).stem for file in label_train_files]

    image_val_filenames = [Path(file).stem for file in image_val_files]
    label_val_filenames = [Path(file).stem for file in label_val_files]

    train_valid = set(image_train_filenames) == set(label_train_filenames)
    val_valid = set(image_val_filenames) == set(label_val_filenames)

    return train_valid and val_valid
