import json
import pandas as pd
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


result = pd.read_csv("P:\\Test_f2_with_bboxes_pretty.csv")
data_path = 'P:\\DATASET-F2\\test'

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files

def infer(images_path, result):
    duration = 0
    # for img_sample, boxes in enumerate(result['file_name'], result['bbox']) also does not work
    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        print("processing...{}".format(filename))
        orig_image = Image.open(img_sample)

        start_t = time.perf_counter()
        end_t = time.perf_counter()

        img = np.array(orig_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for index, row in result.iterrows():
            if filename == row['file_name']:
                cv2.rectangle(img,
                              (int(row["x"]), int(row["y"])),
                              (int(row["w"]) + int(row["x"]), int(row["h"]) + int(row["y"])),
                               (255, 255, 0), 1)




        #not correct because it prints for each filename, all bboxes
        #for box in boxes:
            #print(filename, box)

            #cv2.rectangle(img,
                      #(box[0], box[1]),
                      #(box[2] + box[0], box[3] + box[1]),
                      #(220, 0, 0), 1)


        # img_save_path = os.path.join(output_path, filename)
        # cv2.imwrite(img_save_path, img)
        cv2.imshow("img", img)
        cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        print("Processing...{} ({:.3f}s)".format(filename, infer_time))


    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    image_paths = get_images(data_path)

    infer(image_paths, result)