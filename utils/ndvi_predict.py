import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Define model class
class NDVIResNet(nn.Module):
    def __init__(self):
        super(NDVIResNet, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)

    def forward(self, x):
        return self.model(x)

# Load model
MODEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\hackthon\mango_diagnostics_app\models\ndvi_cnn_model.pth"
IMG_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NDVIResNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Prediction function
def predict_ndvi(img_path):
    image = Image.open(img_path).convert('L')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()

    class_names = {0: "Poor", 1: "Moderate", 2: "Healthy"}
    return f"NDVI Prediction: {class_names[pred]}"
