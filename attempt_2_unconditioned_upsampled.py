import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from collections import deque
from cv2 import dnn_superres

import os
import sys

import time

# Aggiungi il percorso della cartella 'pytorch_GAN_zoo' al sistema di percorsi
sys.path.append(os.path.join(os.path.dirname(__file__), 'pytorch_GAN_zoo'))

from pytorch_GAN_zoo.inference_custom import PlantGen
from upsampler.upsampler import UpsamplerFSRCNN

# Initialize the PlantGen class with the configuration and checkpoint paths
plant = PlantGen(config_path='assets/default_train_config.json',
                 checkpoint_path='assets/default_s6_i400000.pt')

sr = UpsamplerFSRCNN()


# Parameter: Moving average window length
window_length = 10  # Adjust as needed
# Initialize a deque to store the tensors for moving average
tensor_queue = deque(maxlen=window_length)
alpha = 0.9
pooled_output_smoothed = torch.zeros(1, 128)



# Function to generate a random latent vector
def random_latent_vector(size=128):
    return torch.randn(size)

# Function to interpolate between two latent vectors
def interpolate_vectors(vector1, vector2, steps):
    return [vector1 + (vector2 - vector1) * (i / steps) for i in range(steps + 1)]

# Main loop
current_vector = random_latent_vector()
next_vector = random_latent_vector()

steps = 500  # Number of interpolation steps

while True:
    # Interpolate between the current and next vector
    interpolated_vectors = interpolate_vectors(current_vector, next_vector, steps)

    for step, vector in enumerate(interpolated_vectors):
        start_time = time.time()

        # Generate an image using the GAN
        vector = vector.unsqueeze(0)  # Add batch dimension if necessary for the GAN
        generated_image = plant.generate_image(vector)

        # Resize image tensor to 256x256
        generated_image = nn.functional.interpolate(generated_image, size=(350, 200), mode='bicubic', align_corners=False)

        # Convert the generated image tensor to a numpy array
        generated_image_np = generated_image.squeeze().cpu().numpy()
        generated_image_np = (generated_image_np - generated_image_np.min()) / (generated_image_np.max() - generated_image_np.min()) * 255
        generated_image_np = generated_image_np.astype(np.uint8)
        generated_image_np = np.transpose(generated_image_np, (1, 2, 0))  # Rearrange to HWC format

        # # Upsample the generated image
        # generated_image_np = sr.upsample(generated_image_np)

        # Display the generated image
        cv2.imshow('Generated Image', cv2.cvtColor(generated_image_np, cv2.COLOR_RGB2BGR))

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}", end='\r')

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

    # Update current and next vectors for the next interpolation
    current_vector = next_vector
    next_vector = random_latent_vector()


