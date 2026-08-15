from ultralytics import YOLO,settings
import pandas as pd
import os
import wandb
import torch
import random
from matplotlib import pyplot as plt

#check if torch is available
if torch.cuda.is_available():
    device = 0
    print("using cuda...")
else:
    print("using cpu...")
    device = 'cpu'


settings.update({"wandb":True})
wandb.login()





#get script_dir
script_dir=os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.dirname(script_dir)
results_csv=f"{root_dir}/results.csv"
val_path=os.path.join(root_dir,"data","kitti","data_object_image_2","training","images","val")




#define hyperparameters/variables

MODEL=YOLO("yolo11n.pt")  #get weights from a pretrained model for transfer learning
EPOCHS= 100  #5 epochs to check if everything runs end to end
imgsz=640
freeze=10
BATCH_SIZE=-1 #ultralytics automatically detects batch size basd on gpu choosen
DATA=os.path.join(root_dir, "training",'configs','data.yaml')

if os.path.exists(results_csv):
    results_df = pd.read_csv(results_csv)
else:
    results_df=None

# wandb.init(
#     project="SceneDetect-Freeze_Layers",
#     name="Freeze-10-Run",
#     config={
#         "epochs": EPOCHS,
#         "imgsz": imgsz,
#         "freeze": freeze,
#         "batch_size": BATCH_SIZE,
#         "model": "yolo11n.pt"
#     }
# )



def train_model(model,data,epochs,imgsz,freeze,batch,run_name,device):
    results=model.train(data=data,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch,
                        freeze=freeze,
                        project="SceneDetect-Freeze_Layers",
                        name=run_name,
                        plots=True,
                        device=device
                        )
    
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
    elif isinstance(results, dict):
        metrics = results
    else:
        metrics = {}
    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else None
    return metrics, save_dir
    
   


def track_results(metrics,results_df,epochs,imgsz,freeze,batch):
    #create a df if it does not exist
    new_row = dict(metrics)
    new_row.update({"Epochs": epochs,
                    "Batch Size": batch,
                    "Freeze Layers": freeze,
                    "Img_Size": imgsz})
    if results_df is None:
        results_df=pd.DataFrame([new_row])
    else:
        
        results_df=pd.concat([results_df,pd.DataFrame([new_row])],ignore_index=True)
    
    results_df.to_csv(results_csv,index=False)
    
    return results_df



#peek at few predictions on validations
def peek_results(best_model_path,val_path,random_seed):
    random.seed(random_seed)
    #get  best model
    best_model = os.path.join(best_model_path, "weights", "best.pt")
    
    model=YOLO(best_model)
    
    out_path = os.path.join(root_dir, "evaluation", "results", "val_predictions_peek.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    #get random validation images
    val_images=[os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith(".png")]
    selected_images=random.sample(val_images,k=4)
    
    #predict on validation images
    results=model.predict(source=selected_images,
                          save=True,
                          conf=0.25)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for ax, result in zip(axes, results):
        image = result.plot()  # image with predicted boxes
        ax.imshow(image[..., ::-1])  # BGR to RGB
        ax.axis("off")
        basename=os.path.basename(result.path)
        ax.set_title(basename)
    
    plt.tight_layout()
    print("saving predictions")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    
    
    
if __name__=="__main__":
    
    #call the training function
    print("Training Model...")
    metrics,best_model_path=train_model(model=MODEL,data=DATA,epochs=EPOCHS,imgsz=imgsz, freeze=freeze, batch=BATCH_SIZE,run_name="Run-2",device=device)
    
    #track and save results
    print("Saving Resuts")
    results_df=track_results(metrics,results_df,EPOCHS,imgsz,freeze,BATCH_SIZE)
    

    print("training complete\n")
    
    print("predicting on validation images")
    peek_results(best_model_path=best_model_path,val_path=val_path,random_seed=42)
    
    print("execution complete")