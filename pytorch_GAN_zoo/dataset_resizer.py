import os
from PIL import Image
from tqdm import tqdm

# Imposta il percorso delle immagini di input e di output
input_folder = r"D:\Python\Progetti\Artificial_Botanic\test_14_style3_runpod\data_raw"  # Modifica con il percorso corretto
output_folder = r"D:\Python\Progetti\hackaton_progan\v1\pytorch_GAN_zoo\plants"  # Modifica con il percorso corretto

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Dimensioni a cui ridimensionare le immagini
target_size = (256, 256)

# Funzione per ridimensionare e salvare le immagini
def resize_images(input_folder, output_folder, target_size):
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Filtra i formati immagine
            try:
                # Carica l'immagine
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)

                # Ridimensiona l'immagine
                img_resized = img.resize(target_size)

                # Salva l'immagine ridimensionata
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
            except Exception as e:
                print(f"Errore con {filename}: {e}")

# Esegui la funzione
resize_images(input_folder, output_folder, target_size)
