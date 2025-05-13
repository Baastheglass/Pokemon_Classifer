import os
from tqdm import tqdm
import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, 
    ShiftScaleRotate, RandomBrightnessContrast,
    HueSaturationValue, Blur, GaussNoise,
    ElasticTransform, Compose
)
def create_augmentations(input_dir, output_dir, augmentations_per_image=20):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define augmentation pipeline
    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.2),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.8),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        Blur(blur_limit=3, p=0.3),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3)
    ])
    
    # Process each image
    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load original image
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get base filename without extension
            base_name = os.path.splitext(filename)[0]
            
            # Save original image as augmentation 0
            output_path = os.path.join(output_dir, f"{base_name}_0.png")
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Create augmentations
            for i in range(1, augmentations_per_image + 1):
                augmented = augmentation_pipeline(image=image)['image']
                output_path = os.path.join(output_dir, f"{base_name}_{i}.png")
                cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    input_directory = './images'  # Folder with original images
    output_directory = './augmented_images'  # Where augmented images will be saved

    create_augmentations(input_directory, output_directory, augmentations_per_image=20)
