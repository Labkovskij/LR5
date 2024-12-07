import cv2
import os

def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Зображення за шляхом {image_path} не знайдено.")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не вдалося завантажити зображення з {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
