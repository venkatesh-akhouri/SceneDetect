import numpy as np
from cityscapesscripts.helpers.labels import labels as city_labels
import cv2
from pathlib import Path
from tqdm import tqdm

#get a dictionary of what ids need to be replaced with training ids

id_to_train_ids={city_label.id : city_label.trainId for city_label in city_labels}

remap_label_base_name="_labelTrainIds.png"

#create a lookkup array
lookup_array=np.full(256,fill_value=255,dtype=np.uint8)

#fill lookup array with train ids at index with raw ids(ids)

for raw_id,train_id in id_to_train_ids.items():
    if raw_id>-1:
        lookup_array[raw_id]=train_id
    

# print(lookup_array[7])
# print(lookup_array[23])
# print(lookup_array[255])

#read the image files and remap



cityscrape_labels_dir=Path("/Users/venky/scenedetect/data/cityscapes/gt/")

label_id_files=list(cityscrape_labels_dir.rglob("*_labelIds.png"))
#loop through image files

for label_id_file in tqdm(label_id_files,desc="Remapping files"):
    
    #read the image
    img=cv2.imread(str(label_id_file),flags=cv2.IMREAD_UNCHANGED)
    remmaped_img=lookup_array[img]
    label_base_name=str(label_id_file.name).split("_labelIds.png")[0]
    remapped_image_name=label_base_name+remap_label_base_name
    
    remmaped_image_path=label_id_file.parent/ remapped_image_name
    
    #save image
    # print(f"Saving remapped file : {remmaped_image_path}")
    cv2.imwrite(str(remmaped_image_path),remmaped_img)
    
print("finished remapping images")
    


# img=cv2.imread('/Users/venky/scenedetect/data/cityscapes/gt/train/aachen/aachen_000001_000019_gtFine_labelIds.png',flags=cv2.IMREAD_UNCHANGED)
# remap_img=cv2.imread('/Users/venky/scenedetect/data/cityscapes/gt/train/aachen/aachen_000001_000019_gtFine_labelTrainIds.png',flags=cv2.IMREAD_UNCHANGED)
#
# print(np.unique(img))
# print(np.unique(remap_img))
#
# print("---------------------")
# r, c = 500, 500
# print(img[r, c], lookup_array[img[r, c]], remap_img[r, c])
#
