import os
import numpy as np
import cv2
import random
import fnmatch
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", dest="input_path",
    default="P:\\f2-folder-train", # P:\\lottes2018iros_datasets\\CKA_160523
    help="Root directory where images reside."
)
parser.add_argument(
    "--output", dest="output_path",
    default="P:\\image-procs-f2-train",
    help="Root directory for (intermediate) output"
)
parser.add_argument("--max-images", metavar="#IMAGES", default=0, type=int, help="Limits the number of images. Useful for testing")
args = parser.parse_args()

FOLDERS = {
    "RGB": 'images\\rgb',
    "NIR": 'images\\nir',
    "IMAP": 'annotations\\dlp\\iMapCleaned',
    "STEM": 'annotations\\dlp\\stem\\classwise'
}

CLASSES = ["CROP", "WEED", "GRASS"]

MASK_INPUT_DIR = os.path.join(args.input_path, FOLDERS["IMAP"])
MASK_OUTPUT_DIR = { key : os.path.join(args.output_path, key) for key in CLASSES }

# Mask output is input for JSON
JSON_INPUT_DIR = dict(MASK_OUTPUT_DIR)


#label mapping
labelMap = {
    'soil':  {'id': [0, 1, 97]},
    'crop':  {'id': [10000, 10001, 10002]},
    'weed':  {'id': [2]},
    'dycot': {'id': [20000, 20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010, 20011]},
    'grass': {'id': [20100, 20101, 20102, 20103, 20104, 20105]}
}

dlpLabels = {
    'soil':       {'color': (0, 0, 0), 'id': 0},
    'crop':       {'color': (0, 1, 0), 'id': 1},
    'weed':       {'color': (0, 0, 1), 'id': 2},
    'dycot':      {'color': (0, 0, 1), 'id': 3},
    'grass':      {'color': (1, 0, 0), 'id': 4},
    'vegetation': {'color': (1, 1, 1), 'id': 5}
}

# Categories
categories = [
    {"supercategory": "plant", "id": 1, "name": "crop"},
    {"supercategory": "plant", "id": 2, "name": "weed"},
    {"supercategory": "plant", "id": 3, "name": "grass"}
]

def toColor(x, typ):

    x = np.argmax(x, axis=-1)
    shape = x.shape
    color = np.zeros((shape[0], shape[1], 3), np.uint8)

    if typ == 'cdgs':
        classes = ['soil', 'crop', 'dycot', 'grass']
    elif typ == 'stem':
        classes = ['soil', 'crop', 'dycot']

    for c, cl in enumerate(classes):
        mask = (x == c)
        color[:, :, 0] += np.uint8(mask * dlpLabels[cl]['color'][0])
        color[:, :, 1] += np.uint8(mask * dlpLabels[cl]['color'][1])
        color[:, :, 2] += np.uint8(mask * dlpLabels[cl]['color'][2])

    return color


def getColorImageCrop(iMap):
    iMap = np.expand_dims(iMap, axis=-1)
    ones = np.ones(iMap.shape)
    zeros = np.zeros(iMap.shape)
    soil = np.zeros(iMap.shape)

    crop = np.zeros(iMap.shape)
    for label in labelMap['crop']['id']:
        crop += np.where(np.equal(iMap, label),
                         ones,
                         zeros)

    dlpImg = np.concatenate([soil, crop], axis=-1)
    dlpImg = toColor(dlpImg, 'cdgs')
    dlpImg = cv2.cvtColor(dlpImg, cv2.COLOR_RGB2GRAY)

    return (dlpImg * 255).astype(np.uint8)

def getColorImageWeed(iMap):

    iMap = np.expand_dims(iMap, axis=-1)
    ones = np.ones(iMap.shape)
    zeros = np.zeros(iMap.shape)
    soil = np.zeros(iMap.shape)


    weed = np.zeros(iMap.shape)
    for label in labelMap['weed']['id']:
        weed += np.where(np.equal(iMap, label),
                         ones,
                         zeros)
    dycot = np.zeros(iMap.shape)
    for label in labelMap['dycot']['id']:
        dycot += np.where(np.equal(iMap, label),
                          ones,
                          zeros)

    dlpImg = np.concatenate([soil, (weed + dycot)], axis=-1)
    dlpImg = toColor(dlpImg, 'cdgs')
    dlpImg = cv2.cvtColor(dlpImg, cv2.COLOR_RGB2GRAY)

    return (dlpImg * 255).astype(np.uint8)


def getColorImageGrass(iMap):

    iMap = np.expand_dims(iMap, axis=-1)
    ones = np.ones(iMap.shape)
    zeros = np.zeros(iMap.shape)
    soil = np.zeros(iMap.shape)


    grass = np.zeros(iMap.shape)
    for label in labelMap['grass']['id']:
        grass += np.where(np.equal(iMap, label),
                          ones,
                          zeros)

    dlpImg = np.concatenate([soil, grass], axis=-1)
    dlpImg = toColor(dlpImg, 'cdgs')
    #to black and white
    dlpImg = cv2.cvtColor(dlpImg, cv2.COLOR_RGB2GRAY)

    return (dlpImg * 255).astype(np.uint8)

def getColorImage(clazz, iMap):
    if clazz == "CROP":
        return getColorImageCrop(iMap)
    if clazz == "WEED":
        return getColorImageWeed(iMap)
    if clazz == "GRASS":
        return getColorImageGrass(iMap)
    assert(f"Unknown type {clazz}")

def filter_for_png(root, files):
    return [path for path in (os.path.join(root, f) for f in files) if fnmatch.fnmatch(path, "*.png")]

def computeBoundingBoxes(images, category_id, iscrowd):
    output = []
    for index, path in enumerate(images):
        img1 = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        # Apply the thresholding
        a = img1.max()
        _, thresh = cv2.threshold(img1, a/2+60, a,cv2.THRESH_BINARY)

        # Dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
        dilate = cv2.dilate(opening, dilate_kernel, iterations=1)

        # Get contours
        contours = cv2.findContours(dilate,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            area = w*h
            # [bbox,area, image name, image id, category, iscrowd]
            output.append([[x,y,w,h],area, os.path.basename(path), index, category_id, iscrowd])
    return output

def ensureDirectoriesExist(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            print(f"Creating {dir}")
            os.makedirs(dir)


ensureDirectoriesExist(MASK_OUTPUT_DIR.values())

#export binary masks
for root, _, files in os.walk(MASK_INPUT_DIR):
    image_files = filter_for_png(root, files)
    if args.max_images > 0:
        # Only consider a subset (useful for testing)
        image_files = image_files[:args.max_images]

    for path in image_files:
        #print(path)
        imap = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

        for clazz in CLASSES:
            out = getColorImage(clazz, imap)
            out_path = os.path.join(MASK_OUTPUT_DIR[clazz], os.path.basename(path))
            cv2.imwrite(out_path, out)


#get files names
image_files = {}
for clazz in CLASSES:
    for root, _, files in os.walk(JSON_INPUT_DIR[clazz]):
        image_files[clazz] = filter_for_png(root, files)

input_images_all = [
    {"file_name": os.path.basename(path), "id": index}
    for index, path in enumerate(image_files["CROP"])
]

# ANNOTATIONS dict

crop = computeBoundingBoxes(image_files["CROP"], 1, 0)
weed = computeBoundingBoxes(image_files["WEED"], 2, 0)
grass4 = computeBoundingBoxes(image_files["GRASS"], 3, 0)
ann_total = crop + weed + grass4

#sanity check
for label_id, thing in enumerate(ann_total[1:4]):
    print(thing)
    bbox, area, filename, image_id, category_id, iscrowd = thing
    print("Label id:", label_id)
    print("Bounding box:", bbox)
    print("File name:", filename)
    print("Image id:", image_id)
    print("Category:", category_id)
    print("Is crowd:", iscrowd)
    print("Area:", area)
    print()

ann_final_list = [
    {
        "id": label_id,
        "bbox": bbox,
        "area": area,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": iscrowd
    } for label_id, [bbox, area, _, image_id, category_id, iscrowd] in enumerate(ann_total)
]


#all together
everything_dict = {
    "images": input_images_all,
    "annotations": ann_final_list,
    "categories": categories
}

#export to json
with open(os.path.join(args.output_path, "coco-style-dataset-f2-July30-train.json"), "w") as outfile:
    json.dump(everything_dict, outfile)

