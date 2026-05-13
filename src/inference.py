import argparse
import cv2
# from ultralytics import YOLO  # We will uncomment this once the model is trained

def run_inference(image_path, model_path="best.pt"):
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image. Check the file path.")
        return

    print("Loading YOLOv8 model...")
    # model = YOLO(model_path)
    # results = model(image)
    
    print("Inference complete! Displaying results...")
    # results[0].show()  # Opens a window showing the detected tubes
    # results[0].save("output.jpg") # Saves the image with drawn boxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 Inference on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the test image")
    parser.add_argument("--weights", type=str, default="best.pt", help="Path to trained model weights")
    args = parser.parse_args()
    
    run_inference(args.image, args.weights)
