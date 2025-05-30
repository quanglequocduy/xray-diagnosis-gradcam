import torch
import numpy as np
import cv2
import gc
import uuid
from PIL import Image
import cloudinary.uploader
import os
from torchvision import transforms

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# --- Cache mô hình PyTorch ---
MODELS_CACHE = {}

def get_pytorch_model(model_name):
    """
    Load PyTorch model (.pth) from cache or file.
    """
    if model_name not in MODELS_CACHE:
        model_path = {
            "resnet50_v1": "models/resnet50_v1.pth",
            "resnet50_v2": "models/resnet50_v2.pth",
            "densenet121": "models/densenet121.pth",
        }.get(model_name)
        if model_path is None:
            raise ValueError(f"Unknown model name for PyTorch: {model_name}")
        
        # Khởi tạo mô hình
        if model_name == "densenet121":
            model = models.densenet121(weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(num_ftrs, 5)  # Số lớp đầu ra
            )
        elif model_name == "resnet50_v1" or model_name == "resnet50_v2":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 5)  # Số lớp đầu ra
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Tải state_dict vào mô hình
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()  # Đặt mô hình ở chế độ đánh giá
        MODELS_CACHE[model_name] = model
    return MODELS_CACHE[model_name]

# --- Tiền xử lý ảnh ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize ảnh về kích thước 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Chuẩn hóa ảnh
])

def preprocess_image_pytorch(image_path):
    """
    Preprocess input image for PyTorch model.
    """
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    return img_tensor

# --- Tính ApproxCAM ---
def generate_approxcam_pytorch(image_path, model_name, target_class_idx=None):
    """
    Generate ApproxCAM heatmap using PyTorch model (.pth) directly.
    """
    # Preprocess input
    input_tensor = preprocess_image_pytorch(image_path)  # PyTorch tensor (1, 3, 224, 224)
    
    # Load PyTorch model từ cache
    model = get_pytorch_model(model_name)
    
    # Forward pass
    with torch.no_grad():
        feature_maps = model.features(input_tensor)  # Feature maps từ lớp convolution cuối cùng
        logits = model.classifier(feature_maps.mean(dim=(2, 3)))  # Global Average Pooling

    # Lấy lớp mục tiêu (target_class_idx)
    if target_class_idx is None:
        target_class_idx = torch.argmax(logits, dim=1).item()

    # Tính ApproxCAM
    weights = logits[:, target_class_idx].unsqueeze(-1).unsqueeze(-1)  # Lấy trọng số của lớp mục tiêu
    approxcam = torch.sum(weights * feature_maps, dim=1).squeeze(0)  # Tính heatmap

    # Chuẩn hóa heatmap
    approxcam = approxcam.cpu().numpy()
    approxcam = np.maximum(approxcam, 0)  # ReLU
    approxcam = approxcam / approxcam.max()  # Chuẩn hóa về [0, 1]

    # Đọc ảnh gốc, resize heatmap về kích thước ảnh
    img = cv2.imread(image_path)
    heatmap = cv2.resize(approxcam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"approxcam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    # Dọn dẹp bộ nhớ
    del input_tensor, logits, feature_maps, approxcam
    gc.collect()

    return result_path

def upload_to_cloudinary(file_path):
    upload_result = cloudinary.uploader.upload(file_path)
    os.remove(file_path)
    return upload_result["secure_url"]

# --- Hàm chính để gọi ApproxCAM ---
def generate_approxcam_and_upload(image_path, model_name):
    """
    Generate ApproxCAM heatmap and upload to Cloudinary.
    """
    result_path = generate_approxcam_pytorch(image_path, model_name)
    return upload_to_cloudinary(result_path)
