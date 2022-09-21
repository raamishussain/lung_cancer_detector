# Lung Cancer Detector

This repository contains a YOLOv5 model trained to detect lung cancer in MRI images.
The dataset used for training and validation comes from the Cancer Imaging Archive:
[Lung Cancer Dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70224216)

Some python code for visualization and data preparation can also be found at the above link.

## Data Preparation

For data preparation, create a virtual environment and install the requirements in the `data_prep` directory. 

Run `compile_annotated_images.py` with the proper paths to the downloaded `DICOM` files and `xml` annotation files from the Cancer Imaging Archive.

This script will convert the DICOM images into PNGs and then save them in a separate directory along with the associated annotations for each image.

Next we need to create training and validation directories in the YOLO format. Use the notebook `train_test_split.ipynb` to create these directories.

This notebook will create a train/test split by patient, and then sample random images from each patient but taken at different time (different stages of lung cancer). The end of the notebook checks to make sure there are no duplicated patients between the training and validation directories, and that there are no duplicated images used in training.

## Model Training

To run reproduce the model in this repo, you must clone the YOLOv5 repository:
[YOLOv5 Repo](https://github.com/ultralytics/yolov5)

Install the necessary requirements.

The data directory structure should look like the following:

```
- data
    - custom.yaml
    - images
        - train
        - val
    - labels
        - train
        - val
```

Create a `custom.yaml` file for your data set which should be in the parent directory of the dat
    a (next to the image and label directory). The contents of the `yaml` file should be:

```
names:
- cancer
nc: 1
train: /path/to/training/images/
val: /path/to/validation/images/
```

Now you can train the model by running the following command:

```
python /path/to/yolov5/train.py --img 512 --epochs 100 --data /path/to/custom.yaml --weights yolov5s.pt
```

