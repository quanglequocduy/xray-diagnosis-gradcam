import onnxruntime
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import cloudinary.uploader
import os
import uuid
import gc
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.densenet import DenseNet121_Weights

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Model paths
MODEL_PATHS = {
    "resnet50_v1": "models/resnet50_v1.pth",
    "resnet50_v2": "models/resnet50_v2.pth",
    "densenet121": "models/densenet121.pth",
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_model(model_name):
    if model_name == "resnet50_v1" or model_name == "resnet50_v2":
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.fc.in_features, 2)
        )
    elif model_name == "densenet121":
        weights = DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.classifier.in_features, 5)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location="cpu"))
    model.eval()
    return model

def generate_gradcam(image_path, model_name):
    """
    Generate Grad-CAM heatmap for the given image and model (optimized for CPU).
    """
    device = torch.device("cpu")
    model = get_model(model_name).to(device)
    model.eval()

    target_layer = model.layer4[-1] if "resnet" in model_name else model.features[-1]

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, class_idx]
    class_score.backward()

    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    result_path = f"gradcam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)

    # Clean up memory
    del model, input_tensor, grads, acts, heatmap
    gc.collect()

    return result_path


def upload_to_cloudinary(file_path):
    upload_result = cloudinary.uploader.upload(file_path)
    os.remove(file_path)
    return upload_result["secure_url"]

def generate_gradcam_and_upload(image_path, model_name):
    result_path = generate_gradcam(image_path, model_name)
    return upload_to_cloudinary(result_path)

# --- Phần ApproxCAM với ONNX Runtime ---

ONNX_SESSIONS = {}

def get_onnx_session(model_name):
    if model_name not in ONNX_SESSIONS:
        model_path = {
            "resnet50_v1": "models/resnet50_v1.onnx",
            "resnet50_v2": "models/resnet50_v2.onnx",
            "densenet121": "models/densenet121.onnx",
        }.get(model_name)
        if model_path is None:
            raise ValueError(f"Unknown model name for ONNX: {model_name}")
        ONNX_SESSIONS[model_name] = onnxruntime.InferenceSession(model_path)
    return ONNX_SESSIONS[model_name]

def preprocess_image_onnx(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 160, 160)
    return img_tensor.numpy()

def generate_approxcam_onnx(image_path, model_name, target_class_idx=None):
    """
    Generate ApproxCAM heatmap using ONNX Runtime for faster CPU inference and no backward pass.
    """
    # Preprocess input
    input_tensor = preprocess_image_onnx(image_path)  # numpy float32 tensor (1,3,160,160)
    
    # Load ONNX model session từ cache
    sess = get_onnx_session(model_name)
    
    # Get input/output names
    input_name = sess.get_inputs()[0].name
    
    # Run forward pass
    outputs = sess.run(None, {input_name: input_tensor})
    logits = outputs[0]  # assuming output shape (1, num_classes)
    
    if target_class_idx is None:
        target_class_idx = np.argmax(logits)
    
    # Kiểm tra có output thứ 2 chứa feature maps không
    if len(outputs) < 2:
        raise RuntimeError("ONNX model phải export thêm feature maps của last conv layer để tính ApproxCAM")
    
    feature_maps = outputs[1]  # numpy array shape (1, C, H, W)
    
    # Lấy logits class target
    class_score = logits[0, target_class_idx]
    
    # Tính weights theo ApproxCAM: weight_k = score * mean(feature_map_k)
    weights = class_score * feature_maps.mean(axis=(2, 3))  # shape (1, C)
    
    # Tính heatmap
    weighted_maps = weights[:, :, None, None] * feature_maps  # broadcast multiply
    cam = weighted_maps.sum(axis=1).squeeze()  # shape (H, W)
    
    # ReLU và chuẩn hóa heatmap
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-6
    
    # Đọc ảnh gốc, resize heatmap về kích thước ảnh
    img = cv2.imread(image_path)
    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    
    result_path = f"approxcam_{uuid.uuid4().hex}.jpeg"
    cv2.imwrite(result_path, superimposed_img)
    
    # Dọn dẹp bộ nhớ
    del input_tensor, outputs, logits, feature_maps, weighted_maps, cam
    gc.collect()
    
    return result_path

def generate_approxcam_and_upload(image_path, model_name):
    result_path = generate_approxcam_onnx(image_path, model_name)
    return upload_to_cloudinary(result_path)