import os
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

def create_dummy_data(output_dir, num_train=20, num_val=5, size=(256, 256)):
    """
    Generates dummy dataset with random circles.
    Structure:
    output_dir/
        train/
            images/
            masks/
        val/
            images/
            masks/
    """
    root = Path(output_dir)
    
    for split, count in [('train', num_train), ('val', num_val)]:
        img_dir = root / split / 'images'
        mask_dir = root / split / 'masks'
        
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {count} images for {split}...")
        
        for i in tqdm(range(count)):
            # Create black image
            image = np.zeros(size, dtype=np.uint8)
            mask = np.zeros(size, dtype=np.uint8)
            
            # Add random noise to image
            noise = np.random.randint(0, 50, size, dtype=np.uint8)
            image = image + noise
            
            # Draw a random white circle (simulating TMJ)
            center_x = np.random.randint(50, size[1] - 50)
            center_y = np.random.randint(50, size[0] - 50)
            radius = np.random.randint(20, 60)
            
            y, x = np.ogrid[:size[0], :size[1]]
            dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            circle_mask = dist_from_center <= radius
            
            # Mask is purely the circle
            mask[circle_mask] = 255
            
            # Image is circle with some texture/intensity
            image[circle_mask] = np.random.randint(150, 255, size=image[circle_mask].shape)
            
            # Save
            Image.fromarray(image).save(img_dir / f"sample_{i:03d}.png")
            Image.fromarray(mask).save(mask_dir / f"sample_{i:03d}.png")
            
    print(f"✅ Dummy dataset generated at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/dummy_dataset", help="Output directory")
    parser.add_argument("--train", type=int, default=20, help="Number of training images")
    parser.add_argument("--val", type=int, default=5, help="Number of validation images")
    args = parser.parse_args()
    
    create_dummy_data(args.output, args.train, args.val)

