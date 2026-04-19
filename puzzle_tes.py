import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image_PIL
import os

def main():
    # 1. Setup Transforms for 1024x1024
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Images
    image_path_a = 'dataset/test_images/DALL-E.jpg'
    image_path_b = 'dataset/test_images/f169da0a598814ec38ccb0e25920a011.jpg'
    
    img_a = transform(Image_PIL.open(image_path_a).convert('RGB')).unsqueeze(0)
    img_b = transform(Image_PIL.open(image_path_b).convert('RGB')).unsqueeze(0)
    images = torch.cat([img_a, img_b], dim=0) 

    # Create the Grid Mask
    grid_size = 64 
    low_res_saliency = torch.randn((images.size(0), 1, grid_size, grid_size))
    
    # Simple swap indices
    indices = torch.tensor([1, 0])
    mask = (low_res_saliency < low_res_saliency[indices]).float()

    # Upscale mask to 1024x1024
    full_mask = F.interpolate(mask, size=(1024, 1024), mode='nearest')

    # Apply the Mix
    images_shuffled = images[indices]
    mixed_images = images * (1 - full_mask) + images_shuffled * full_mask

    # De-normalize Function
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)

    # Save each image individually
    output_dir = 'dataset/test_images/'
    os.makedirs(output_dir, exist_ok=True)
    
    final_images = denormalize(mixed_images)
    
    # Loop through the batch
    
    save_path = os.path.join(output_dir, f'puzzle_mixed.jpg')
    save_image(final_images[0], save_path)

if __name__ == "__main__":
    main()