import argparse
import os

from inference.inference_pipeline import load_models, run_inference_pipeline, device

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(ROOT_DIR, "models", "best.pt")
SEGFORMER_PATH = os.path.join(ROOT_DIR, "models", "best_segformer_model_clss_wts_norm.pt")


def argparser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run the SceneDetect inference pipeline (YOLO11n + SegFormer) on an image",
    )
    parser.add_argument("--image", type=str, required=True, help="path to the input image")
    parser.add_argument("--file_name", type=str, default="output.png",
                         help="output file name, saved under inference/eval/")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()

    print("Loading models...")
    yolo_model, segformer_model = load_models(YOLO_PATH, SEGFORMER_PATH, device)

    print("Running inference pipeline...")
    run_inference_pipeline(yolo_model, segformer_model, args.image, device, args.file_name)
    print("Done.")
