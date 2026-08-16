import torch
from torch.utils.data import Dataset
import pathlib
import cv2



class CityScapesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir  #pathlib object
        self.labels_dir = labels_dir   #pathlib object
        self.samples=[]
        #get list of paths all images
        image_files = pathlib.Path(self.images_dir).rglob('*.png')
        for img_file in image_files:
            #get base name
            base_name = img_file.stem
            #get label file name
            label_file_base_name=base_name.replace('leftImg8bit',"gtFine_labelTrainIds.png")
            #get city name
            city=img_file.parent.name
            #create the label path
            label_file=self.labels_dir/city/label_file_base_name
            #append image_file and label_file to list
            if label_file.exists():
                self.samples.append((img_file,label_file))
            
            
        
    
    def __len__(self):
        return len(self.samples)
      
    
    def __getitem__(self, idx):
        img_file,label_file=str(self.samples[idx][0]),str(self.samples[idx][1])
        img_file=cv2.imread(img_file)
        img_file=cv2.cvtColor(img_file,cv2.COLOR_BGR2RGB)
        img_file=img_file.transpose(2,0,1)
        label_file=cv2.imread(label_file,flags=cv2.IMREAD_GRAYSCALE)
        img_file=torch.tensor(img_file)
        label_file=torch.tensor(label_file)
        return img_file,label_file
    
    