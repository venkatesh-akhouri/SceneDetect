#adding test file to data.yaml
import yaml
import os


script_dir=os.path.dirname(os.path.abspath(__file__))
root_dir=os.path.dirname(script_dir)
images_dir=os.path.join(root_dir, "data","kitti", "data_object_image_2","training","images")

yaml_path=os.path.join(root_dir, "training","configs")

#check if config folder exixts
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)


class_map={"Car": 0,
           "Pedestrian": 1,
           "Cyclist": 2,
           "Van": 3,
           "Truck": 4}


yaml_dict={
    "path":images_dir,
    "train":"train",
    "val":"val",
    "test":"test",
    "names":{value:key for key,value in class_map.items()}
}

#generate the yaml file
yaml_file=f"{yaml_path}/data.yaml"
with open(yaml_file, "w") as f:
    yaml.dump(yaml_dict, f)

print(f"Wrote {yaml_file}")

    