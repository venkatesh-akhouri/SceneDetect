from ultralytics import YOLO
import wandb
import os
import numpy as np
import pandas as pd
import argparse
import torch
from matplotlib import pyplot as plt
import random
import sys


if torch.cuda.is_available():
    device = 0
    print("using cuda...")
else:
    print("using cpu...")
    device = 'cpu'


#get script_dir
script_dir=os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.dirname(script_dir)
IMAGES_DIR=os.path.join(root_dir,"data","kitti","data_object_image_2","training","images")
EVAL_PATH=os.path.join(root_dir,"evaluation","results")


DATA=os.path.join(root_dir, "training",'configs','data.yaml')

parser=argparse.ArgumentParser()

parser.add_argument('--model_name',type=str,default='best_model.pt')
parser.add_argument('--file_name',type=str,default="test_results.png")
args=parser.parse_args()

model_path="/workspace/scenedetect/runs/detect/SceneDetect-Freeze_Layers/Full_Fine_Tune_Baseline-2/weights"
model_name=args.model_name
if os.path.exists(os.path.join(model_path,model_name)):
    #load model
    print("Evaluating Test performance of the model...")
    model=YOLO(os.path.join(model_path,model_name))
    results=model.val(data=DATA,split='test',device=device,imgsz=960)
    
    print(f"mAP50-95: {results.box.map}")
    print(f"mAP50: {results.box.map50}")
else:
    print(f"{model_path} doesn't exist")
    sys.exit(1)
    
    
#test on images
def peek_yolo_results(model, images_dir, eval_path, file_name, num_images=6, seed=42):
    random.seed(seed)

    test_images_dir = os.path.join(images_dir, "test")
    all_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    sample_images = random.sample(all_images, num_images)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, img_name in zip(axes, sample_images):
        img_path = os.path.join(test_images_dir, img_name)
        result = model.predict(source=img_path, device=device,verbose=False,imgsz=960)[0]

        annotated = result.plot()  # BGR numpy array, boxes/labels drawn in
        annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB for matplotlib

        ax.imshow(annotated_rgb)
        ax.set_title(img_name)
        ax.axis("off")

    os.makedirs(eval_path, exist_ok=True)
    plt.tight_layout()
    out_path = os.path.join(eval_path, file_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved YOLO test predictions to {out_path}")

peek_yolo_results(model,IMAGES_DIR,EVAL_PATH, args.file_name)
