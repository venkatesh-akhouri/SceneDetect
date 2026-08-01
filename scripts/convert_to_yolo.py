import os
from PIL import Image
from tqdm import tqdm


class_map={"Car":0,
           "Pedestrian":1,
           "Cyclist":2,
           "Van":3,
           "Truck":4,
           "Person_sitting":1}


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
#get images dir
images_dir=os.path.join(project_dir, "data","kitti","data_object_image_2","training","image_2")
#get labels directory
labels_dir=os.path.join(project_dir, "data","kitti","data_object_image_2","training_labels","label_2")

#create new labels directpry
new_labels_dir=os.path.join(project_dir, "data","kitti","data_object_image_2","training","new_labels","label_2")
print("Making new directory")
os.makedirs(new_labels_dir,exist_ok=True)


#loop around each label
labels_files=os.listdir(labels_dir)
for filename in tqdm(labels_files,desc="Converting labels from KITTI to YOLO"):
    converted_lines=[]
    
    if filename.endswith(".txt"):
        temp = filename.split(".")[0]
        print(f"filename: {filename}")
        # print(f"temp {temp}")
        image_name = os.path.join(images_dir, f"{temp}.png")
        img = Image.open(image_name)
        img_w, img_h = img.size
        try:
            with open(os.path.join(labels_dir,filename),"r") as f:
                for line in f.readlines():
                    values=line.strip().split()
                    if values[0] in class_map.keys():
                        class_id=class_map[values[0]]
                        x_center=(float(values[6])+float(values[4]))/2/img_w
                        y_center=(float(values[7])+float(values[5]))/2/img_h
                        box_width=(float(values[6])-float(values[4]))/img_w
                        box_height=(float(values[7])-float(values[5]))/img_h
                        converted_lines.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")
                        
        except OSError as e:
            raise RuntimeError(f"Cannot open file {filename}: {e}")
        
        #write in a new file
        with open(os.path.join(new_labels_dir,filename),"w") as file:
            file.writelines(converted_lines)
        
        
        