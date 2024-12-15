from cv2 import dnn_superres, dnn

class UpsamplerFSRCNN:
    def __init__(self):
        self.path = r"D:\Python\Progetti\15_hackaton_2024\upsampler\models\FSRCNN-small_x2.pb"
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(self.path)
        self.sr.setModel("FSRCNN".lower(), 2)

        # Imposta il backend e il target per utilizzare CUDA
        self.sr.setPreferableBackend(dnn.DNN_BACKEND_CUDA)
        self.sr.setPreferableTarget(dnn.DNN_TARGET_CUDA)

    def upsample(self, input_image):
        return self.sr.upsample(input_image)
