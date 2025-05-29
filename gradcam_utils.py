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
    transforms.Resize((160, 160)),
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
