import os
import sys
import cv2
import matplotlib.pyplot as plt

# Inverse of CLASS_MAP, for labeling converted boxes by name instead of just ID
ID_TO_NAME = {0: "Car", 1: "Pedestrian", 2: "Cyclist", 3: "Van", 4: "Truck"}

# Colors (BGR, since we're using OpenCV to draw)
KITTI_COLOR = (60, 200, 60)     # green
YOLO_COLOR = (60, 60, 230)      # red


def draw_kitti_boxes(image, label_path):
    """Draw raw KITTI boxes (left, top, right, bottom in pixels) in green."""
    img = image.copy()
    if not os.path.exists(label_path):
        return img
    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            cls_name = values[0]
            left, top, right, bottom = map(float, values[4:8])
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), KITTI_COLOR, 2)
            cv2.putText(img, cls_name, (int(left), max(int(top) - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, KITTI_COLOR, 1, cv2.LINE_AA)
    return img


def draw_yolo_boxes(image, label_path):
    """Draw converted YOLO boxes (normalized center-based) in red, denormalized back to pixels."""
    img = image.copy()
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img
    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            cls_id = int(values[0])
            x_c, y_c, bw, bh = map(float, values[1:5])
            # denormalize back to pixel coordinates for drawing
            left = (x_c - bw / 2) * w
            right = (x_c + bw / 2) * w
            top = (y_c - bh / 2) * h
            bottom = (y_c + bh / 2) * h
            cls_name = ID_TO_NAME.get(cls_id, str(cls_id))
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), YOLO_COLOR, 2)
            cv2.putText(img, cls_name, (int(left), max(int(top) - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YOLO_COLOR, 1, cv2.LINE_AA)
    return img


def main(image_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    image_path = os.path.join(project_dir, "data", "kitti", "data_object_image_2",
                               "training", "image_2", f"{image_id}.png")
    kitti_label_path = os.path.join(project_dir, "data", "kitti", "data_object_image_2",
                                     "training_labels", "label_2", f"{image_id}.txt")
    yolo_label_path = os.path.join(project_dir, "data", "kitti", "data_object_image_2",
                                    "training", "new_labels", "label_2", f"{image_id}.txt")

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kitti_img = cv2.cvtColor(draw_kitti_boxes(image, kitti_label_path), cv2.COLOR_BGR2RGB)
    yolo_img = cv2.cvtColor(draw_yolo_boxes(image, yolo_label_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(kitti_img)
    axes[0].set_title(f"Original KITTI labels ({image_id}.txt)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(yolo_img)
    axes[1].set_title(f"Converted YOLO labels ({image_id}.txt)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle(f"Conversion sanity check -- image {image_id}", fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(project_dir, "evaluation", "results", f"verify_{image_id}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to: {out_path}")
    plt.show()


if __name__ == "__main__":
    image_id = sys.argv[1] if len(sys.argv) > 1 else "000079"
    main(image_id)