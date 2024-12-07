import argparse
from src.image_processing import load_and_preprocess_image, enhance_contrast
from src.clustering import kmeans_clustering
from src.object_counting import count_objects
from src.image_comparison import compare_images
from src.video_identification import identify_objects_in_video

def parse_args():
    parser = argparse.ArgumentParser(description="Обробка зображень та відео")
    parser.add_argument("image_path", type=str, help="Шлях до вхідного зображення")
    parser.add_argument("video_path", type=str, help="Шлях до вхідного відео")
    return parser.parse_args()

def main():
    args = parse_args()

    # Завантаження та обробка зображення
    image = load_and_preprocess_image(args.image_path)
    enhanced_image = enhance_contrast(image)

    # Кластеризація
    pixels = enhanced_image.reshape(-1, 3)
    labels, centers = kmeans_clustering(pixels, n_clusters=3)

    # Підрахунок об'єктів
    object_count, contours = count_objects(enhanced_image)

    # Порівняння зображень
    image1 = load_and_preprocess_image('data/image1.jpg')
    image2 = load_and_preprocess_image('data/image2.jpg')
    difference = compare_images(image1, image2)

    # Ідентифікація об'єктів у відео
    identify_objects_in_video(args.video_path)

if __name__ == "__main__":
    main()
