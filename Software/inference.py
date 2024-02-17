import torch
import torchvision.models as models
import torchvision.transforms as transforms
from model import CovNet


from PIL import Image

# Step 1: Define the Model Architecture
model = CovNet()

# Step 2: Load the Pre-trained Model
model.load_state_dict(torch.load('model_1.pth'))  # Replace 'resnet18.pth' with the path to your pre-trained model file

# Step 3: Set the Model to Evaluation Mode
model.eval()

# Step 4: Define Preprocessing and Postprocessing Steps (if needed)
transformer = transforms.Compose([
    transforms.Grayscale(),  # to properly handle loading of .png/.jpeg images
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.4465],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.2410])
])

# Step 5: Run Inference
# Load and preprocess input image
image = Image.open('DataRaw/single/img.png')  # Replace 'input_image.jpg' with the path to your input image
input_tensor = transformer(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# If GPU is available, move the input batch to GPU
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')

model.eval()
# Run inference

output = model(input_batch)
# Get predicted class
prediction = 0
if(output.item()>0.5):
    prediction =1

# Print predicted class
print('Predicted class:', prediction)