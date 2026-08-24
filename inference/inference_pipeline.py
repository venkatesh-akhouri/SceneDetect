import os
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation
import torch
from torchvision import transforms as T
import cv2
from cityscapesscripts.helpers.labels import trainId2label
import numpy as np
import argparse



if torch.cuda.is_available():
    device=torch.device('cuda')
    print("Using GPU")
else:
    device=torch.device('cpu')
    print("Using CPU")


#get directories
script_dir=os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.dirname(script_dir)
EVAL_PATH=os.path.join(root_dir,"inference","eval")
if not os.path.exists(EVAL_PATH):
    os.makedirs(EVAL_PATH,exist_ok=True)

segformer_model_id='nvidia/segformer-b0-finetuned-ade-512-512'

transformation=T.Compose([T.Resize((512,512),interpolation=T.InterpolationMode.BILINEAR,antialias=True),
                          T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

CITYSCAPES_COLOR_MAP=np.array([trainId2label[i].color for i in range(19)],dtype=np.uint8)



def load_models(yolo_path, segformer_path,device):
    yolo_model = YOLO(yolo_path)
    
    #load the architecture
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(segformer_model_id,
                                                                 num_labels=19,
                                                                 ignore_mismatched_sizes=True)
    #load out weights
    state_dict=torch.load(segformer_path,map_location=device)
    segformer_model.load_state_dict(state_dict)
    segformer_model.to(device)
    segformer_model.eval()
    
    return yolo_model, segformer_model


def run_inference_pipeline(yolo_model, segformer_model,image,device,save_file_name):
    yolo_op=yolo_model.predict(image,imgsz=960,device=device)
    
    image_array=cv2.imread(image)
    #convert to RGB
    image_array=cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
    #convert to image tensor
    image_tensor=torch.tensor(image_array).permute(2,0,1).float()/255
    image_tensor=transformation(image_tensor)
    image_tensor=image_tensor.unsqueeze(0).to(device)
    
    op_logits=segformer_model(pixel_values=image_tensor)
    #get upsampled logits
    upsampled_logits=torch.nn.functional.interpolate(op_logits.logits,size=image_array.shape[:2],mode='bilinear',align_corners=False)
    
    #get predicted segmentation mask
    op_segmentation_mask=torch.argmax(upsampled_logits,dim=1).cpu().numpy().squeeze(0)
    
    # numpy vectorisation
    cityscapes_color_mask=CITYSCAPES_COLOR_MAP[op_segmentation_mask]
    
    
    #overlay image
    overlayed_image=cv2.addWeighted(image_array,0.6,cityscapes_color_mask,0.4,0)
    
    BGR_image=cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)
    #draw the yolo boxes
    op_image=yolo_op[0].plot(img=BGR_image)
    
    #save this
    saved=os.path.join(EVAL_PATH,save_file_name)
    cv2.imwrite(saved,op_image)
    print(f"Saved at {saved}")


def argeparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image',type=str)
    parser.add_argument('--file_name',type=str)
    
    return parser.parse_args()
    
    


if __name__ == "__main__":
    
    args=argeparser()
    
    #load models
    yolo_path = "/workspace/scenedetect/runs/detect/SceneDetect-Freeze_Layers/Full_Fine_Tune_Baseline-2/weights/best.pt"
    segformer_path = "/workspace/scenedetect/models/best_segformer_model_clss_wts_norm.pt"
    
    yolo_model,segformer_model=load_models(yolo_path,segformer_path,device)
    
    #run inference
    print("Running inference pipeline")
    run_inference_pipeline(yolo_model,segformer_model,args.image,device,args.file_name)
    print("Finished inference pipeline")
    


