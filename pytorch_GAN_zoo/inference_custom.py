import importlib
from PIL import Image
import numpy as np
import torch
import json
from models.progressive_gan_custom import ProgressiveGAN

class PlantGen():
    """
    PlantGen is a class responsible for generating images using a pre-trained Progressive GAN model.
    
    Attributes:
        config_path (str): Path to the JSON configuration file.
        checkpoint_path (str): Path to the model checkpoint file.
        pgan (ProgressiveGAN): Instance of the ProgressiveGAN model initialized with the given configuration and checkpoint.
    """

    def __init__(self, 
                 config_path: str, 
                 checkpoint_path: str):
        """
        Initializes the PlantGen class with the provided configuration and checkpoint paths.

        Args:
            config_path (str): Path to the JSON configuration file.
            checkpoint_path (str): Path to the model checkpoint file.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.pgan = self.setup_pgan()

    def setup_pgan(self):
        """
        Sets up the ProgressiveGAN model by loading the configuration and checkpoint.

        Returns:
            ProgressiveGAN: An instance of the ProgressiveGAN model initialized with the loaded configuration and checkpoint.
        """
        # Load the configuration from JSON
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        # Initialize the ProgressiveGAN model with specific parameters
        pgan = ProgressiveGAN(useGPU=False,                 
                              dimLatentVector=config["dimLatentVector"],
                              depthScale0=config["depthScales"][0],
                              initBiasToZero=True,
                              leakyness=0.2,
                              perChannelNormalization=True,
                              miniBatchStdDev=False,
                              equalizedlR=True,
                              )
        # Load the checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=pgan.device)
        # Prepare the state dictionary for loading
        state_dict = {
            'netG': checkpoint['netG'],
            'config': config  # Add the config to match the expected structure
        }
        # Load the state dictionary into the model
        pgan.load_state_dict(state_dict)
        
        return pgan
    
    def generate_image(self, latent_vector):
        """
        Generates an image from a given latent vector using the ProgressiveGAN model.

        Args:
            latent_vector (torch.Tensor): A tensor representing the latent vector input to the GAN.

        Returns:
            torch.Tensor: The generated image as a tensor.
        """
        with torch.no_grad():
            # Generate the image using the model's test method
            generated_image = self.pgan.test(input=latent_vector, getAvG=True)
        return generated_image


if __name__ == "__main__":

# Hardcoded values
    config_path = r"D:\Python\Progetti\15_hackaton_2024\default_train_config.json"
    checkpoint_path = r"D:\Python\Progetti\15_hackaton_2024\default_s6_i400000.pt"

    plant_gen = PlantGen(config_path, checkpoint_path)
    latent_vector = torch.randn(1, 128).to(plant_gen.pgan.device)
    generated_image = plant_gen.generate_image(latent_vector)
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