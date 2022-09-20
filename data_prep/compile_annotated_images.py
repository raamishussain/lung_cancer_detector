import argparse, os, cv2, PIL
import pydicom as pdm
import numpy as np
import xml.etree.ElementTree as ET

from getUID import getUID_path
from get_gt import *
from get_data_from_XML import XML_preprocessor 
from pathlib import Path

from xml.etree.ElementTree import ParseError

def compile_annotated_images(dicom_path,annotation_path):
    r""" This function gathers DICOM images and their associated 
    annotations and returns a list of dictionaries where each
    element in the list is a patient and each key-value pair 
    within the dictionary represents the path to the DICOM image
    and annotation file respectively

    Args:
        dicom_path (str):
            Path to DICOM image
        annotation_path (str):
            Path to XML file containing annotation
    
    Returns:
        list_of_dicts (list):
            List of dicts containing dicom images and annotations
            for each patient
    """
    dcm_dirs = os.listdir(dicom_path)

    list_of_dicts = []
    for dir in dcm_dirs:
        if dir==".DS_Store" or dir=="LICENSE":
            continue
        
        local_path = Path(dicom_path,dir)
        patient = dir.split('-')[1]
        anno_path = Path(annotation_path,patient)
        dict = getUID_path(local_path)
        try:
            annotations = XML_preprocessor(anno_path, num_classes=4).data
        except (FileNotFoundError, ParseError):
            print(f"Invalid or Missing annotation for patient {patient}")
            continue
        annotated_dict = {}

        # compile all images which have an annotation for each patient
        for k, v in annotations.items():
            try:
                dcm_path, dcm_name = dict[k[:-4]]
            except KeyError:
                # print(f"Patient {patient} key missing: {k[:-4]}")
                continue
            annotated_dict[dcm_path] = Path(anno_path,k)

        list_of_dicts.append(annotated_dict)

    
    return list_of_dicts

def convert_dcm_2_png(dicom_path,save_path):
    r""" Convert a DICOM image file into a PNG
    
    Args:
        dcm_path (str):
            Location of DICOM file
        save_path (str):
            Location to save PNG file
    """

    ds = pdm.dcmread(dicom_path)
    pixel_array = ds.pixel_array
    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0 

    cv2.imwrite(save_path,pixel_array)


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def convert_annotations(anno_path,save_path):
    r""" Convert annotations from XML format to 
    text file in the YOLOV5 label format
    """

    tree = ET.parse(anno_path)
    root = tree.getroot()

    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    text_lines = []

    for obj in root.findall('object'):


        pil_bbox = [int(x.text) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        text_lines.append(f"0 {bbox_string}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

def main():

    # Parse arguments
    parser = argparse.ArgumentParser('Annotation Visualization')

    parser.add_argument('--dicom-path', type=str,
                        help='path to the folder stored dicom files (.DCM)')
    parser.add_argument('--annotation-path', type=str,
                        help='path to the folder stored annotation files (.xml) or a path to a single annotation file')
    parser.add_argument('--classfile', type=str, default='category.txt',
                        help='path to the txt file stored categories')
    args = parser.parse_args()
    
    list_of_dicts = compile_annotated_images(args.dicom_path,args.annotation_path)
    PNG_FOLDER = "/Users/raamis/lung_cancer_detector/data/images"
    TXT_FOLDER = "/Users/raamis/lung_cancer_detector/data/labels"

    for d in list_of_dicts:
        count = 0
        for k,v in d.items():
            patient_id = k.split('/')[3].split('-')[1]
            png_path = os.path.join(PNG_FOLDER,patient_id+f'_{count}.png')
            txt_path = os.path.join(TXT_FOLDER,patient_id+f'_{count}.txt')
            count+=1

            convert_dcm_2_png(k,png_path)
            convert_annotations(v,txt_path)




if __name__ == '__main__':
    main()