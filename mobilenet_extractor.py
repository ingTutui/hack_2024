import torch
import timm

class MobileNetV4Extractor:
    def __init__(self):
        """
        Initialize the MobileNetV4Extractor with the model and prepare for efficient inference.
        """
        # Load the pretrained MobileNetV4 model
        self.model = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True)
        self.model.eval()

        # Prepare to intercept the output of the specified layer
        self.intercepted_output = None

        # Register the hook for the target layer
        self.target_layer = self.model.get_submodule('blocks.3')
        self.hook = self.target_layer.register_forward_hook(self._intercept_layer)

    def _intercept_layer(self, module, input, output):
        """
        Internal method to capture the output of the target layer during a forward pass.

        Args:
            module (torch.nn.Module): The layer being hooked.
            input (torch.Tensor): Input to the layer.
            output (torch.Tensor): Output from the layer.
        """
        self.intercepted_output = output

    def extract_features(self, input_tensor):
        """
        Extract features from the target layer for a given input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor to pass through the model.

        Returns:
            torch.Tensor: Output tensor of the specified layer.
        """
        self.intercepted_output = None  # Reset the intercepted output
        with torch.no_grad():
            self.model(input_tensor)
        return self.intercepted_output

    def cleanup(self):
        """
        Clean up the registered hook to avoid memory leaks.
        """
        self.hook.remove()


# Example usage for real-time processing
if __name__ == "__main__":
    import cv2

    # Initialize the extractor
    extractor = MobileNetV4Extractor()

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Loop to process frames in real time
    while True:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (resize to match model input and normalize)
        frame_resized = cv2.resize(frame, (224, 224))
        input_tensor = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = (input_tensor - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Extract features
        features = extractor.extract_features(input_tensor)

        # Process features or pass to GAN
        print("Extracted Features Shape:", features.shape)

        # Display the original frame
        cv2.imshow("Camera Feed", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    extractor.cleanup()