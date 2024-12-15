import torch
import yaml
import time
from models import *
from models.vanilla_vae import VanillaVAE
from torchvision.utils import save_image

# Load configuration
config_path = 'configs/vae.yaml'  # Update with your config file path if different
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Parameters
latent_dim = config['model_params']['latent_dim']
checkpoint_path = r'C:\Users\StefanoPellegrini\Desktop\Projects\hackaton_2024\poc\PyTorch-VAE\logs\VanillaVAE\version_22\checkpoints\last.ckpt'  # Replace with your checkpoint path

# Initialize the model
model_name = config['model_params']['name']
model_class = vae_models[model_name]
model = model_class(**config['model_params'])

# Load the model's weights
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Handle PyTorch Lightning checkpoint format
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Remove 'model.' prefix from parameter names if present
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('model.'):
        new_key = k[len('model.'):]
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model.eval()

# Move model to CPU
device = torch.device('cpu')
model.to(device)

# Start time measurement
start_time = time.time()
elapsed_time = 0
num_generated = 0

# Generate images for 10 seconds
with torch.no_grad():
    while elapsed_time < 10:
        # Generate random latent vector
        z = torch.randn(1, latent_dim).to(device)
        # Generate image
        generated = model.decode(z)
        # Increment counter
        num_generated += 1
        # Update elapsed time
        elapsed_time = time.time() - start_time

save_image(generated, 'generated_images/image_k.png')
# Print the results
print(f"Generated {num_generated} images in {elapsed_time:.2f} seconds.")
print(f"Average time per image: {elapsed_time / num_generated:.4f} seconds.")
print(f"Average frame rate: {num_generated / elapsed_time:.2f} FPS.")
