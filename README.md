# MP_final
scripts for data preprocessing - master project Eva 
1. Train-test-split.ipynb --> splits RGB img folder into 80% training and 20% test set + creates equivalent sets for the Imap masks from Lottes as well 
2. Masks_and_json.py --> using the data obtained above, it creates binary masks for each class and the corresponding annotation json file in COCO format
3. From_json_to_csv_and_stats --> inspect json files, check counts of annotations, miscelaneous. (The csv format is more readable, idem using pandas)
4. Pretty_bbox_format.py --> input: the csv obtained in [3]; it outputs another csv which has the bounding box values instead of [x,y,w,h] in a list, as 4 separate columns named "x", "y", "w", "h" respectively. This format makes plotting easier
5. Draw_BB.py --> input: the new csv format outputed from [4]; output: visualization of images + ground truth bounding boxes overlayed.
