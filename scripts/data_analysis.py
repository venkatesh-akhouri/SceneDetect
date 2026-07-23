import os
from collections import Counter

class_counts=Counter()
script_dir = os.path.dirname(os.path.abspath(__file__)) # .../scenedetect/scripts
project_root = os.path.dirname(script_dir)

labels_path=os.path.join(project_root, 'data', 'kitti', 'data_object_image_2', 'training_labels', 'label_2')

if os.path.exists(labels_path):
    print("exists")
    for filename in os.listdir(labels_path):
        if filename.endswith('.txt'):
            #read file
            try:
                with open (os.path.join(labels_path, filename), 'r') as file:
                    for line in file:
                        words = line.strip().split()
                        if words:
                            class_name= words[0]
                            class_counts[class_name] += 1
            except OSError as e:
                raise RuntimeError(f"Could not open file {os.path.join(labels_path, filename)} {e}")
                        
                

print("-"*50)
print("class_Counts")
print("-"*50)

for cls, count in class_counts.items():
        print(f"{cls:<15}: {count}")

            
    