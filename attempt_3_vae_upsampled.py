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

# Aggiungi il percorso della cartella 'pytorch_GAN_zoo' al sistema di percorsi
sys.path.append(os.path.join(os.path.dirname(__file__), 'simply_VAE'))


from pytorch_GAN_zoo.inference_custom import PlantGen
from upsampler.upsampler import UpsamplerFSRCNN
from simply_VAE.inference_custom import VaeCameraEncoder

# Initialize the PlantGen class with the configuration and checkpoint paths
plant = PlantGen(config_path='asset/default_train_config.json',
                 checkpoint_path='asset/default_s6_i200000.pt')
vae_encoder = VaeCameraEncoder(config_path=r'D:\Python\Progetti\15_hackaton_2024\simply_VAE\configs/vae.yaml',
                       checkpoint_path=r'D:\Python\Progetti\15_hackaton_2024\simply_VAE\checkpoints\last.ckpt')

sr = UpsamplerFSRCNN()


# Parameter: Moving average window length
window_length = 10  # Adjust as needed
# Initialize a deque to store the tensors for moving average
tensor_queue = deque(maxlen=window_length)
alpha = 0.9
pooled_output_smoothed = torch.zeros(1, vae_encoder.latent_dim)

# Initialize the camera
cap = cv2.VideoCapture(0)


while True:
    start_time = time.time()  # Start the timer for FPS calculation

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Generate latent space using VAE encoder
    latent_space = vae_encoder.encode_camera(frame)

    # # Smooth the latent space representation using exponential moving average
    # latent_space_smoothed = alpha * latent_space_smoothed + (1 - alpha) * latent_space

    # Generate an image using the GAN
    generated_image = plant.generate_image(latent_space)

    # Resize image tensor to 256x256
    generated_image = nn.functional.interpolate(generated_image, size=(700, 500), mode='bicubic', align_corners=False)

    # Convert the generated image tensor to a numpy array
    generated_image_np = generated_image.squeeze().cpu().numpy()
    generated_image_np = (generated_image_np - generated_image_np.min()) / (generated_image_np.max() - generated_image_np.min()) * 255
    generated_image_np = generated_image_np.astype(np.uint8)
    generated_image_np = np.transpose(generated_image_np, (1, 2, 0))  # Rearrange to HWC format

    # # Upsample the generated image
    # generated_image_np = sr.upsample(generated_image_np)

    # Display the generated image
    cv2.imshow('Generated Image', cv2.cvtColor(generated_image_np, cv2.COLOR_RGB2BGR))

    # Calculate and print the FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    print(f"FPS: {fps:.2f}", end='\r')  # Print FPS in real time

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()


