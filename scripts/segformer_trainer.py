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
import  torchvision.transforms.v2 as T
from torchvision import tv_tensors as TV

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
val_metric=evaluate.load("mean_iou")

#GLOBAL VARIABLES
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.dirname(SCRIPT_DIR)
TRAIN_IMAGES_DIR=os.path.join(ROOT_DIR,'data','cityscapes','images','train')
VAL_IMAGES_DIR=os.path.join(ROOT_DIR,'data','cityscapes','images','val')
TRAIN_LABELS_DIR=os.path.join(ROOT_DIR,'data','cityscapes','gt','train')
VAL_LABELS_DIR=os.path.join(ROOT_DIR,'data','cityscapes','gt','val')
EVAL_PATH=os.path.join(ROOT_DIR,'evaluation','results')
BEST_MODEL_PATH=os.path.join(ROOT_DIR,'models')
#in case if directory does not exist
os.makedirs(BEST_MODEL_PATH,exist_ok=True)



# #define transformations
# train_transform = A.Compose([
#                 # flip
#                 A.HorizontalFlip(p=0.3),
#                 A.Rotate(limit=(-10, 10), p=0.3),
#                 A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)])

#define augmentations
train_augmentions=T.Compose([
    T.RandomResizedCrop((512,512),scale=(0.5,1.0),interpolation=T.InterpolationMode.BILINEAR),
    T.RandomHorizontalFlip(p=0.4),
    T.RandomRotation(degrees=(-10,10),fill={TV.Image: 0, TV.Mask: 255}),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.RandomApply([T.GaussianBlur(kernel_size=(3, 5))], p=0.2),
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
        
        #not doing any preprocesing or transformation here
        # if self.transform:
        #     augmentation = self.transform(image=img_arr,mask=label_arr)
        #     img,label=augmentation['image'],augmentation['mask']
        # else:
        #     img=img_arr
        #     label=label_arr
        # encoded=self.processor(images=img,segmentation_maps=label,return_tensors="pt")
        #
        # pixel_values=encoded['pixel_values'].squeeze(0)
        # label=encoded['labels'].squeeze(0)
        #
        #
        # return pixel_values,label
        
        img_tensor=torch.from_numpy(img_arr).permute(2,0,1).float()/255.0
        label_tensor=torch.from_numpy(label_arr).long()
        
        return img_tensor,label_tensor

#get dataset set
train_dataset=CityScapesDataset(Path(TRAIN_IMAGES_DIR),Path(TRAIN_LABELS_DIR))
val_dataset=CityScapesDataset(Path(VAL_IMAGES_DIR),Path(VAL_LABELS_DIR))



transformation=T.Compose([T.Resize((512,512),interpolation=T.InterpolationMode.BILINEAR,antialias=True),
                          T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])


label_transformation=T.Resize(
    (512, 512),
    interpolation=T.InterpolationMode.NEAREST
)

def compute_inverse_class_weights(label_file_list):
    pixel_count=np.zeros(19,dtype=np.int64)
    for label_file in tqdm(label_file_list,desc="Calculating class weights..."):
        label_img_arr=cv2.imread(label_file,flags=cv2.IMREAD_UNCHANGED)
        pixel_val,count=np.unique(label_img_arr,return_counts=True)
        
        #create a boolean mask
        mask=pixel_val!=255
        pixel_count[pixel_val[mask]]+=count[mask]
    
    
    num_class=pixel_count.shape[0]
    
    #inverse frequencey class weights
    #weight_c = total_samples / (num_classes × count_c),
    total_pixel_count=np.sum(pixel_count)
    normalised_pxl_counts=pixel_count/total_pixel_count
    
    inverse_wt_count=1/np.log(1.02+normalised_pxl_counts)
    
    return inverse_wt_count
    
        
    
    

#train function
def train(model,epochs,optimiser,metric,val_metric,train_dataloader,val_dataloader,class_weights):
    train_loss=[]
    val_loss=[]
    train_mIOU=[]
    val_mIOU=[]
    
    best_val_loss=float('inf')
    #create learning scheduler
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,T_max=epochs)
    
    weighted_loss = torch.nn.CrossEntropyLoss(weight=class_weights,
                                              ignore_index=255)
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
            img=img.to(device,non_blocking=True)
            label=label.to(device,non_blocking=True)
            
            #wrap then into tv_tensors
            img_tv=TV.Image(img)
            label_tv=TV.Mask(label)
            
            #augment the data
            img_tv,label_tv=train_augmentions(img_tv, label_tv)
            
            # #befre forward pass, apply transformations
            # img=transformation(img_tv)
            # label=label_transformation(label_tv.unsqueeze(1)).squeeze(1)
            
            
            #forward pass
            output_logits=model(pixel_values=img_tv)
            upsampled_logits=torch.nn.functional.interpolate(
                output_logits.logits, size=label_tv.shape[-2:], mode="bilinear", align_corners=False
            )
            
            #
          
           
    
            
            #get prections
            preds=torch.argmax(upsampled_logits,dim=1)
            
            #get the loss
            #loss is calculated under the hood
            batch_loss=weighted_loss(upsampled_logits,label_tv)
            train_epoch_loss += batch_loss.item()
            
            #calculate mean_iou
            metric.add_batch(predictions=preds.cpu().numpy(), references=label_tv.cpu().numpy())
            
            batch_loss.backward()  #computed the gradiesnts and stores in .grad
            #apply gradient clipping
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            
            optimiser.step()
        
        #compute the metrics
        results = metric.compute(num_labels=19, ignore_index=255)
        # add this to epoch loss and epoch mIOU
        
        train_epoch_mIOU = results['mean_iou']
        
        avg_train_loss = train_epoch_loss/len(train_dataloader)
   
        
        #append to list
        train_loss.append(avg_train_loss)
        train_mIOU.append(train_epoch_mIOU)
        
        #lr schedular
        scheduler.step()
        
        # validate on validation data
        model.eval()
        with torch.no_grad():
            
            # load the validation batch
            for val_batch in tqdm(val_dataloader, desc=f'Epoch: {epoch + 1}/{epochs} Validation'):
                val_image, val_label = val_batch
                val_image = val_image.to(device,non_blocking=True)
                val_label = val_label.to(device,non_blocking=True)
                
                #transform validation data as well
                val_image=transformation(val_image)
                val_label=label_transformation(val_label.unsqueeze(1)).squeeze(1)
              
                
                
                val_op_logits = model(pixel_values=val_image, labels=val_label)
                upsampled_logits = torch.nn.functional.interpolate(
                    val_op_logits.logits, size=val_label.shape[-2:], mode="bilinear", align_corners=False
                )
                
                val_preds = torch.argmax(upsampled_logits, dim=1)
                
                val_batch_loss = val_op_logits.loss
                val_epoch_loss += val_batch_loss.item()
                
                val_metric.add_batch(predictions=val_preds.cpu().numpy(), references=val_label.cpu().numpy())
                
            
            #compute the metrics
            val_results = val_metric.compute(num_labels=19, ignore_index=255)
            
            
            
            val_epoch_mIOU = val_results['mean_iou']
            per_class_ious = {
                f"val/IoU_{trainId2label[i].name}": iou
                for i, iou in enumerate(val_results['per_category_iou'])
            }
            
        avg_val_loss = val_epoch_loss/len(val_dataloader)
        
        
        
        if avg_val_loss < best_val_loss:
            best_val_loss=avg_val_loss
            BEST_MODEL=model.state_dict()
            #save the best model
            print(f"Saving best model at epoch {epoch+1}")
            torch.save(BEST_MODEL,os.path.join(BEST_MODEL_PATH,'best_segformer_model_clss_wts_norm.pt'))
            
        
        #append to list
        val_loss.append(avg_val_loss)
        val_mIOU.append(val_epoch_mIOU)
    
        print(f"Training loss: {avg_train_loss: .4f} | train mIOU: {train_epoch_mIOU: .4f})")
        print(f"Validation loss: {avg_val_loss: .4f} | val mIOU: {val_epoch_mIOU: .4f})\n")
        
        #log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "train/mIoU": train_epoch_mIOU,
            "val/loss": avg_val_loss,
            "val/mIoU": val_epoch_mIOU,
            **per_class_ious
        })

    return os.path.join(BEST_MODEL_PATH,'best_segformer_model_clss_wts_norm.pt')

def peek_segformer_results(best_model, val_dataset, device, file_name, num_images=2, seed=42):
    random.seed(seed)
    best_model.eval()
    
    indices = random.sample(range(len(val_dataset)), num_images)
    
    color_lookup = np.zeros((256, 3), dtype=np.uint8)
    for train_id, label in trainId2label.items():
        if train_id in (255, -1):
            continue
        color_lookup[train_id] = label.color
    
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = np.expand_dims(axes, axis=0)
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            raw_img, raw_gt = val_dataset[idx]
            
            img_tensor = raw_img.unsqueeze(0).to(device)
   
            gt_tensor = raw_gt.unsqueeze(0).unsqueeze(0).to(device)
            
            transformed_img = transformation(img_tensor)
         
            transformed_gt = label_transformation(gt_tensor).squeeze(0).squeeze(0)
            
            #load best model
            
            outputs = best_model(pixel_values=transformed_img)
            upsampled_logits = torch.nn.functional.interpolate(
                outputs.logits, size=transformed_gt.shape[-2:], mode="bilinear", align_corners=False
            )
            
            pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
            gt_mask = transformed_gt.cpu().numpy()
            
            img_disp = transformed_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
            
            gt_color = color_lookup[np.clip(gt_mask, 0, 255)]
            pred_color = color_lookup[np.clip(pred_mask, 0, 255)]
            
            axes[row, 0].imshow(img_disp);
            axes[row, 0].set_title(f"Input (idx {idx})");
            axes[row, 0].axis("off")
            axes[row, 1].imshow(gt_color);
            axes[row, 1].set_title("Ground Truth");
            axes[row, 1].axis("off")
            axes[row, 2].imshow(pred_color);
            axes[row, 2].set_title("Prediction");
            axes[row, 2].axis("off")
    
    os.makedirs(EVAL_PATH, exist_ok=True)
    plt.tight_layout()
    out_path = os.path.join(EVAL_PATH, file_name)
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
    parser.add_argument("--file_name", type=str, help="file name to save visualisation", default="segformer_vis.png")
    
    
    return parser.parse_args()



if __name__=="__main__":
    
    args = argument_paser()
    # set wandb
    wandb.login()
    
    # initalise project
    wandb.init(project="Finetune Segformer",name=args.run_name,
               config={
                   "epochs": args.epochs,
                   "batch_size": args.batch,
                   "learning_rate": args.lr,
                   "model_id": model_id,
                   "weighting_strategy": "ENet (c=1.02)",
                   "optimizer": "AdamW",
                   "scheduler": "CosineAnnealingLR",
                   "image_size": (512, 512)}
               )
    
    
    
    # craete dataloaders
    print("Creating data loaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    
    optimiser=torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    #get list of label files
    label_files=[str(sample[1]) for sample in train_dataset.samples]
    
    #get inverse wt counts
    print("calculating class weights...")
    inverse_wt_counts=compute_inverse_class_weights(label_files)
    class_weights=torch.tensor(inverse_wt_counts,dtype=torch.float32).to(device)
    #start training
    print("Start Training....")
    model_path=train(model=model,epochs=args.epochs,optimiser=optimiser,metric=metric,val_metric=val_metric,train_dataloader=train_dataloader,val_dataloader=val_dataloader,
                     class_weights=class_weights)
    print("training finished")
    
    
    #peek results
    print("Visualizing results...")
    #load best model
    state_dict=torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict)
    
    
    peek_segformer_results(best_model=model,val_dataset=val_dataset,device=device,file_name=args.file_name)
