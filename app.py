import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from flask import Flask, request, render_template, send_from_directory
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load MiDaS depth estimation model
def load_midas_model():
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    return midas, device

# Load DeepLabV3 for person segmentation
def load_deeplab_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = deeplabv3_mobilenet_v3_large(pretrained=True)
    model.to(device)
    model.eval()
    return model, device

midas, device = load_midas_model()
deeplab, device = load_deeplab_model()

def estimate_depth(image, midas, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth_map = midas(img_tensor)
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map

def segment_person(image, model, device):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    person_mask = (output_predictions == 15).astype(np.uint8) * 255
    return person_mask

def apply_depth_blur(image, depth_map, mask):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    width, height = pil_image.size  
    depth_pil = Image.fromarray(depth_map).convert("L").resize((width, height), Image.BILINEAR)
    mask_pil = Image.fromarray(mask).convert("L").resize((width, height), Image.BILINEAR)

    blurred_far = pil_image.filter(ImageFilter.GaussianBlur(20))
    blurred_mid = pil_image.filter(ImageFilter.GaussianBlur(10))
    blurred_near = pil_image.filter(ImageFilter.GaussianBlur(5))

    background = Image.composite(blurred_far, blurred_mid, depth_pil)  
    background = Image.composite(background, blurred_near, depth_pil)  
    final_image = Image.composite(pil_image, background, mask_pil)

    return final_image

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if file.filename == "":  # Check if a file was selected
        return "No selected file", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Process image
    image = cv2.imread(filepath)
    if image is None:
        return "Error loading image", 400

    depth_map = estimate_depth(image, midas, device)
    mask = segment_person(image, deeplab, device)
    output_image = apply_depth_blur(image, depth_map, mask)

    output_path = os.path.join(OUTPUT_FOLDER, file.filename)
    output_image.save(output_path)

    return send_from_directory(OUTPUT_FOLDER, file.filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render assigns a port dynamically
    app.run(host='0.0.0.0', port=port)
