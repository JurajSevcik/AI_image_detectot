import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image as Image_PIL
import os
from tqdm import tqdm


def puzzle_mix_folder():
    # --- Nastavenia ---
    path_real = 'mixDataset/real'
    path_fake = 'mixDataset/fake'
    output_dir = 'mixDataset/mixed'

    num_images = 2000  # Počet obrázkov na spracovanie
    grid_size = 8  # Hustota "puzzle" mriežky
    img_size = 1024  # Rozlíšenie

    os.makedirs(output_dir, exist_ok=True)

    # 1. Príprava transformácií (bez normalizácie pre jednoduchšie ukladanie)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 2. Načítanie zoznamov súborov
    real_files = sorted([f for f in os.listdir(path_real) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[
        :num_images]
    fake_files = sorted([f for f in os.listdir(path_fake) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[
        :num_images]

    if len(real_files) < num_images or len(fake_files) < num_images:
        print(f"Upozornenie: Nájdených menej obrázkov než požadovaných {num_images}.")
        num_images = min(len(real_files), len(fake_files))

    print(f"Generujem {num_images} mixovaných obrázkov...")

    # 3. Cyklus na spracovanie
    for i in tqdm(range(num_images)):
        try:
            # Načítanie dvojice (Real a Fake)
            img_r = transform(Image_PIL.open(os.path.join(path_real, real_files[i])).convert('RGB')).unsqueeze(0)
            img_f = transform(Image_PIL.open(os.path.join(path_fake, fake_files[i])).convert('RGB')).unsqueeze(0)

            # Spojenie do batchu [2, 3, 1024, 1024]
            batch = torch.cat([img_r, img_f], dim=0)

            # Vytvorenie Puzzle masky (rovnaká logika ako vo tvojom skripte)
            # Vygenerujeme náhodné hodnoty pre mriežku
            low_res_saliency = torch.randn((1, 1, grid_size, grid_size))
            low_res_saliency_shuffled = torch.randn((1, 1, grid_size, grid_size))

            # Maska určí, kde zostane Real (0) a kde pôjde Fake (1)
            mask = (low_res_saliency < low_res_saliency_shuffled).float()

            # Zväčšenie masky na plné rozlíšenie
            full_mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')

            # Aplikácia mixu: Real * (Inverzná Maska) + Fake * Maska
            # Ak maska=1, berie sa pixel z Fake. Ak maska=0, berie sa z Real.
            mixed_img = img_r * (1 - full_mask) + img_f * full_mask

            # Uloženie výsledku
            save_path = os.path.join(output_dir, f'mixed_{i:04d}.jpg')
            save_image(mixed_img[0], save_path)

        except Exception as e:
            print(f"Chyba pri indexe {i}: {e}")

    print(f"Hotovo! Obrázky sú v: {output_dir}")


if __name__ == "__main__":
    puzzle_mix_folder()