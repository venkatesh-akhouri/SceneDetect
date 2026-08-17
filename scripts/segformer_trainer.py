import torch
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW
from pathlib import Path
import cv2
import albumentations as A
from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
import os
import argparse
from tqdm import tqdm
import evaluate
import wandb
import random
import numpy as np
import matplotlib.pyplot as plt
from cityscapesscripts.helpers.labels import trainId2label

#set device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using GPU")
else:
    device = torch.device('cpu')
    print("Using CPU")
    
    
#set model id
model_id='nvidia/segformer-b0-finetuned-ade-512-512'

#set model
model = SegformerForSemanticSegmentation.from_pretrained(model_id,
                                                         num_labels=19,
                                                         ignore_mismatched_sizes=True)

#shift model to device
model=model.to(device)

#set metric
metric=evaluate.load("mean_iou")

#GLOBAL VARIABLES
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.dirname(SCRIPT_DIR)
TRAIN_IMAGES_DIR=os.path.join(ROOT_DIR,'data','cityscapes','images','train')
VAL_IMAGES_DIR=os.path.join(ROOT_DIR,'data','cityscapes','images','val')
TRAIN_LABELS_DIR=os.path.join(ROOT_DIR,'data','cityscapes','gt','train')
VAL_LABELS_DIR=os.path.join(ROOT_DIR,'data','cityscapes','gt','val')
EVAL_PATH=os.path.join(ROOT_DIR,'data','evaluation')




#define transformations
train_transform = A.Compose([
                # flip
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=(-10, 10), p=0.3),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)])

class CityScapesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir  #pathlib object
        self.labels_dir = labels_dir   #pathlib object
        self.samples=[]
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        #get list of paths all images
        image_files = Path(self.images_dir).rglob('*.png')
        for img_file in list(image_files):
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
            
        self.transform = transform
        
        self.val_transform=None
        
    
    def __len__(self):
        return len(self.samples)
      
    
    def __getitem__(self, idx):
        img_file,label_file=str(self.samples[idx][0]),str(self.samples[idx][1])
        img_arr=cv2.imread(img_file)
        img_arr=cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
      
        label_arr=cv2.imread(label_file,flags=cv2.IMREAD_UNCHANGED)
        
        if self.transform:
            augmentation = self.transform(image=img_arr,mask=label_arr)
            img,label=augmentation['image'],augmentation['mask']
        else:
            img=img_arr
            label=label_arr
        encoded=self.processor(images=img,segmentation_maps=label,return_tensors="pt")
        
        pixel_values=encoded['pixel_values'].squeeze(0)
        label=encoded['labels'].squeeze(0)
        
       
        return pixel_values,label

#get dataset set
train_dataset=CityScapesDataset(Path(TRAIN_IMAGES_DIR),Path(TRAIN_LABELS_DIR),train_transform)
val_dataset=CityScapesDataset(Path(VAL_IMAGES_DIR),Path(VAL_LABELS_DIR))





#train function
def train(model,epochs,optimiser,metric,train_dataloader,val_dataloader):
    train_loss=[]
    val_loss=[]
    train_mIOU=[]
    val_mIOU=[]
    
    for epoch in range(epochs):
        train_epoch_loss=0
        val_epoch_loss=0
        train_epoch_mIOU=0
        val_epoch_mIOU=0
        
        #switch to train
        model.train()
        #fecth a bacth from train loader
        for batch in tqdm(train_dataloader,desc=f'Epoch {epoch+1}/{epochs}'):
            
            #make gradients zero
            optimiser.zero_grad()
            
            #get pixel values and labels from batch
            img,label=batch
            
            #put them on gpu
            img=img.to(device)
            label=label.to(device)
            #forward pass
            output_logits=model(pixel_values=img,labels=label)
            upsampled_logits=torch.nn.functional.interpolate(
                output_logits.logits, size=label.shape[-2:], mode="bilinear", align_corners=False
            )
            
            #get prections
            preds=torch.argmax(upsampled_logits,dim=1)
            
            #get the loss
            #loss is calculated under the hood
            batch_loss=output_logits.loss
            train_epoch_loss += batch_loss.item()
            
            #calculate mean_iou
            metric.add_batch(predictions=preds.cpu().numpy(), references=label.cpu().numpy())
            
            batch_loss.backward()
            optimiser.step()
        
        #compute the metrics
        results = metric.compute(num_labels=19, ignore_index=255)
        # add this to epoch loss and epoch mIOU
        
        train_epoch_mIOU = results['mean_iou']
        
        avg_train_loss = train_epoch_loss/len(train_dataloader)
   
        
        #append to list
        train_loss.append(avg_train_loss)
        train_mIOU.append(train_epoch_mIOU)
        
        # validate on validation data
        model.eval()
        with torch.no_grad():
            
            # load the validation batch
            for val_batch in tqdm(val_dataloader, desc=f'Epoch: {epoch + 1}/{epochs} Validation'):
                val_image, val_label = val_batch
                val_image = val_image.to(device)
                val_label = val_label.to(device)
                
                val_op_logits = model(pixel_values=val_image, labels=val_label)
                upsampled_logits = torch.nn.functional.interpolate(
                    val_op_logits.logits, size=val_label.shape[-2:], mode="bilinear", align_corners=False
                )
                
                val_preds = torch.argmax(upsampled_logits, dim=1)
                
                val_batch_loss = val_op_logits.loss
                val_epoch_loss += val_batch_loss.item()
                
                metric.add_batch(predictions=val_preds.cpu().numpy(), references=val_label.cpu().numpy())
                
            
            #compute the metrics
            val_results = metric.compute(num_labels=19, ignore_index=255)
            
            val_epoch_mIOU = val_results['mean_iou']
            
        avg_val_loss = val_epoch_loss/len(val_dataloader)
        
        
        #append to list
        val_loss.append(avg_val_loss)
        val_mIOU.append(val_epoch_mIOU)
    
        print(f"Training loss: {avg_train_loss: .4f} | train mIOU: {train_epoch_mIOU: .4f})\n")
        print(f"Validation loss: {avg_val_loss: .4f} | val mIOU: {val_epoch_mIOU: .4f})")
        
        #log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/mIoU": train_epoch_mIOU,
            "val/loss": avg_val_loss,
            "val/mIoU": val_epoch_mIOU,
        })
        
def peek_segformer_results(model, val_dataset, device,file_name, num_images=2, seed=42):
    random.seed(seed)
    model.eval()

    indices = random.sample(range(len(val_dataset)), num_images)

    # build trainId -> RGB color lookup, using Cityscapes' official palette
    color_lookup = np.zeros((256, 3), dtype=np.uint8)
    for train_id, label in trainId2label.items():
        if train_id in (255, -1):
            continue
        color_lookup[train_id] = label.color

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            pixel_values, gt_mask = val_dataset[idx]
            input_tensor = pixel_values.unsqueeze(0).to(device)

            outputs = model(pixel_values=input_tensor)
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False
            )
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
            gt_mask = gt_mask.cpu().numpy()

            # un-normalize the image tensor back to a viewable 0-1 range
            img_disp = pixel_values.permute(1, 2, 0).cpu().numpy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

            gt_color = color_lookup[np.clip(gt_mask, 0, 255)]
            pred_color = color_lookup[np.clip(pred_mask, 0, 255)]

            axes[row, 0].imshow(img_disp); axes[row, 0].set_title(f"Input (idx {idx})"); axes[row, 0].axis("off")
            axes[row, 1].imshow(gt_color); axes[row, 1].set_title("Ground Truth"); axes[row, 1].axis("off")
            axes[row, 2].imshow(pred_color); axes[row, 2].set_title("Prediction"); axes[row, 2].axis("off")

    plt.tight_layout()
    out_path=os.path.join(EVAL_PATH, file_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved peek visualization to {out_path}")
        
        
def argument_paser():
    '''
    function to implement CLI
    will pas hyperparameters as arguments
    :return:
    '''
    
    parser = argparse.ArgumentParser(prog='segformer_trainer',
                                     description='Finetuning Segformer on CityScapes')
    
    # add arguments to parser
    parser.add_argument("--epochs", type=int, help="number of epochs", default=100)
    parser.add_argument("--batch", type=int, help="batch size", default=32)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--run_name", type=str, help="run name", default="Test Run")
    parser.add_argument("--file_name", type=int, help="file name to save visualisation", default="segformer_vis.png")
    
    
    return parser.parse_args()



if __name__=="__main__":
    
    args = argument_paser()
    # set wandb
    wandb.login()
    
    # initalise project
    wandb.init(project="Finetune Segformer",name=args.run_name)
    
    
    
    # craete dataloaders
    print("Creating data loaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    optimiser=torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    #start training
    print("Start Training....")
    train(model=model,epochs=args.epochs,optimiser=optimiser,metric=metric,train_dataloader=train_dataloader,val_dataloader=val_dataloader)
    print("training finished")
    
    #peek results
    print("Visualizing results...")
    peek_segformer_results(model=model,val_dataset=val_dataset,device=device,file_name=args.file_name)
