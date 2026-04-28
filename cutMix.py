import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image_PIL
import os
from tqdm import tqdm


def grid_swap_generator():
    # --- Nastavenia ---
    path_real = 'testingPicsMix/real'
    path_fake = 'testingPicsMix/fake'
    output_dir = 'testingPicsMix/cutMixed'

    num_images = 1
    img_size = 1024
    grid_count = 8  # 8x8 mriežka (spolu 64 blokov)
    block_size = img_size // grid_count  # 1024 / 8 = 128 pixelov na blok

    os.makedirs(output_dir, exist_ok=True)

    # Transformácia (iba základná, aby sme mali tenzory)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    real_files = sorted([f for f in os.listdir(path_real) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[
        :num_images]
    fake_files = sorted([f for f in os.listdir(path_fake) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[
        :num_images]

    print(f"Generujem {len(real_files)} Swap-Mix obrázkov...")

    for i in tqdm(range(len(real_files))):
        try:
            # 1. Načítanie obrázkov
            img_r = transform(Image_PIL.open(os.path.join(path_real, real_files[i])).convert('RGB'))
            img_f = transform(Image_PIL.open(os.path.join(path_fake, fake_files[i])).convert('RGB'))

            # 2. Vytvorenie výsledného plátna (začneme s kópiou reálneho obrázka)
            mixed_img = img_r.clone()

            # 3. Prechádzame mriežku blok po bloku
            for row in range(grid_count):
                for col in range(grid_count):
                    # Náhodné rozhodnutie: 0 = nechať Real, 1 = nahradiť za Fake
                    if torch.rand(1).item() > 0.5:
                        y_start = row * block_size
                        y_end = y_start + block_size
                        x_start = col * block_size
                        x_end = x_start + block_size

                        # Nahradíme celý blok pixelmi z fake obrázka
                        mixed_img[:, y_start:y_end, x_start:x_end] = img_f[:, y_start:y_end, x_start:x_end]

            # 4. Uloženie
            save_path = os.path.join(output_dir, f'mixed_{i:04d}.jpg')
            save_image(mixed_img, save_path)

        except Exception as e:
            print(f"Chyba pri indexe {i}: {e}")


if __name__ == "__main__":
    grid_swap_generator()