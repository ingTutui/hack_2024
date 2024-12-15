import torch
import yaml
from torchvision.transforms import transforms
from PIL import Image
import cv2
# from models_vae import *
from models_vae import vae_models

class VaeCameraEncoder:
    def __init__(self, config_path, checkpoint_path, device='cpu'):
        """
        Initialize the VAE Camera Encoder class by loading the model and its weights.

        Args:
            config_path (str): Path to the YAML configuration file.
            checkpoint_path (str): Path to the model checkpoint file.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)

        # Load configuration
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Model parameters
        self.latent_dim = config['model_params']['latent_dim']
        model_name = config['model_params']['name']
        model_class = vae_models[model_name]

        # Initialize the model
        self.model = model_class(**config['model_params'])

        # Load the model's weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle PyTorch Lightning checkpoint format
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        # Set the model to evaluation mode
        self.model.eval()
        self.model.to(self.device)

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode_camera(self, img):
        """
        Process a camera image and return the latent space representation.

        Args:
            img (numpy.ndarray): Camera image in BGR format.

        Returns:
            torch.Tensor: Latent space representation of the image.
        """
        # Convert the image to PIL format and apply transformations
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        frame_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Encode the frame
        with torch.no_grad():
            encoded = self.model.encode(frame_tensor)[0]  # Assuming 'encode' method returns latent representation

        return encoded

# Example usage
if __name__ == "__main__":
    # Define paths
    config_path = 'configs/vae.yaml'
    checkpoint_path = r'D:\Python\Progetti\15_hackaton_2024\simply_VAE\checkpoints\last.ckpt'

    # Initialize the encoder
    vae_encoder = VaeCameraEncoder(config_path, checkpoint_path, device='cpu')

    # Initialize camera
    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Get latent representation
            latent_space = vae_encoder.encode_camera(frame)

            # Display latent space shape
            print(f"Latent Encoding Shape: {latent_space.shape}")

            # Display the camera feed
            cv2.imshow("Camera Feed", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()