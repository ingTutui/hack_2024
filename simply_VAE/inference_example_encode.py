import torch
import yaml
import time
from models import *
from models.vanilla_vae import VanillaVAE
from torchvision.transforms import transforms
from PIL import Image
import cv2

# Load configuration
config_path = 'configs/vae.yaml'  # Update with your config file path if different
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Parameters
latent_dim = config['model_params']['latent_dim']
checkpoint_path = r'D:\Python\Progetti\15_hackaton_2024\simply_VAE\checkpoints\last.ckpt'  # Replace with your checkpoint path

# Initialize the model
model_name = config['model_params']['name']
model_class = vae_models[model_name]
model = model_class(**config['model_params'])

# Load the model's weights
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Handle PyTorch Lightning checkpoint format
state_dict = checkpoint.get('state_dict', checkpoint)
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model.eval()
device = torch.device('cpu')
model.to(device)

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize camera
cap = cv2.VideoCapture(0)
start_time = time.time()
frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Increment frame count
        frame_count += 1

        # Convert frame to PIL Image and preprocess
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tensor = transform(image).unsqueeze(0).to(device)

        # Encode the frame
        with torch.no_grad():
            encoded = model.encode(frame_tensor)[0]  # Assuming 'encode' method returns latent representation

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        print(f"Latent Encoding Shape: {encoded.shape}")
        print(f"FPS: {fps:.2f}")

        # Display the camera feed with FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Camera Feed", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
