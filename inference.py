import os
import sys

# Aggiungi il percorso della cartella 'pytorch_GAN_zoo' al sistema di percorsi
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_GAN_zoo'))

from pytorch_GAN_zoo.inference_custom import PlantGen
import torch
import numpy as np
from PIL import Image

# Initialize the PlantGen class with the configuration and checkpoint paths
plant = PlantGen(config_path='assets/default_train_config.json',
                 checkpoint_path='assets/default_s6_i400000.pt')

# Generate a random latent vector
latent_vector = torch.randn(1, 128).to(plant.pgan.device)

# Generate an image from the latent vector
generated_image = plant.generate_image(latent_vector)

print("Generated image shape:", generated_image.shape)
# Convert the generated image tensor to a numpy array
generated_image_np = generated_image.squeeze().cpu().numpy()
# Normalize the image to the range [0, 255]
generated_image_np = (generated_image_np - generated_image_np.min()) / (generated_image_np.max() - generated_image_np.min()) * 255
generated_image_np = generated_image_np.astype(np.uint8)
# Convert the numpy array to a PIL Image
image = Image.fromarray(generated_image_np.transpose(1, 2, 0))
# Display the image
image.show()
