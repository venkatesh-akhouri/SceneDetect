#split strategy
#group file names by rarest class
#then split those file name - 70/15/15

import os
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import random

#craete groups
car=set()
pedestrian=set()
cyclist=set()
van=set()
truck=set()

class_map={"Car":0,
           "Pedestrian":1,
           "Cyclist":2,
           "Van":3,
           "Truck":4}

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
yolo_labels_dir=os.path.join(project_dir, "data","kitti","data_object_image_2","training","new_labels","label_2")

imgs_src_dir=os.path.join(project_dir, "data","kitti","data_object_image_2","training","image_2")
img_dest = os.path.join(project_dir, "data", "kitti", "data_object_image_2", "training", "images")
label_dest = os.path.join(project_dir, "data", "kitti", "data_object_image_2", "training", "labels")



def parse_label_file(args):
    yolo_labels_dir,filename=args
    file_path=os.path.join(yolo_labels_dir,filename)
    
    classes_in_file=set()
    try:
        with open(file_path,"r") as f:
            for line in f:
                words=line.strip().split()
                classes_in_file.add(words[0])
    
    except OSError:
        raise RuntimeError(f"File {file_path} not found.")
    
    return filename,classes_in_file


def split(yolo_labels_dir,train_ratio=0.7,val_ratio=0.15):
    file_args=[(yolo_labels_dir,f) for f in os.listdir(yolo_labels_dir)]
    
    classes_and_file={} #keeps track of file names and count of classes present in it
    global_counts=Counter()
    
    with ProcessPoolExecutor() as executor:
        results=executor.map(parse_label_file,file_args)
    
    #write the file name and class count in classes and file data structure
    for filename,classes in results:
        classes_and_file[filename]=classes
        for cls in classes:
            global_counts[cls]+=1
            
            
    sorted_dict=[cls for cls,_ in sorted(global_counts.items(), key=lambda item:item[1])]
    grouped_files=defaultdict(list)
    for filename,classes in classes_and_file.items():
        for cls in  sorted_dict:
            if cls in classes:
                grouped_files[cls].append(filename)
                break
    
    train_files,val_files,test_files=[],[],[]
    for cls, files in grouped_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train: n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    
    return train_files, val_files, test_files


def make_symlinks(files_list,split_name,img_src_dir,label_src_dir,img_dest,label_dest):
    
    #create directories
    images_destination=os.path.join(img_dest,split_name)  #images/train
    labels_destination=os.path.join(label_dest,split_name) #labels/train
    
    os.makedirs(images_destination,exist_ok=True)
    os.makedirs(labels_destination,exist_ok=True)
    
    #loop through the labels file list
    #get base name
    for label_file_name in files_list:
        base_name=label_file_name.split(".")[0]
        img_file_name=f"{base_name}.png"
        
        
        src_image=os.path.join(img_src_dir,img_file_name)
        src_label=os.path.join(label_src_dir,label_file_name)
        
        dest_image=os.path.join(images_destination,img_file_name)
        dest_label=os.path.join(labels_destination,label_file_name)
        
        
        os.symlink(src_image,dest_image)
        os.symlink(src_label,dest_label)


if __name__ == "__main__":
    train_files,val_files,test_files=split(yolo_labels_dir,train_ratio=0.7)
    
    #call symlink
    make_symlinks(train_files, "train", imgs_src_dir, yolo_labels_dir, img_dest, label_dest)
    make_symlinks(val_files, "val", imgs_src_dir, yolo_labels_dir, img_dest, label_dest)
    make_symlinks(test_files, "test", imgs_src_dir, yolo_labels_dir, img_dest, label_dest)