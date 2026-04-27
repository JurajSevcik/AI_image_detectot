import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image as Image_PIL
from tqdm import tqdm


# --- Modelová architektúra (musí byť identická s tou pri trénovaní) ---

def get_fft_features(img_tensor):
    gray = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.log(torch.abs(fft_shift) + 1e-8)
    min_v = magnitude.view(magnitude.size(0), -1).min(1, keepdim=True)[0].unsqueeze(2)
    max_v = magnitude.view(magnitude.size(0), -1).max(1, keepdim=True)[0].unsqueeze(2)
    magnitude = (magnitude - min_v) / (max_v - min_v + 1e-8)
    return magnitude.unsqueeze(1)


def get_pixel_residuals(img_tensor):
    kernel = torch.tensor([[-1., -1., -1.],
                           [-1., 8., -1.],
                           [-1., -1., -1.]],
                          device=img_tensor.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)
    residual = torch.nn.functional.conv2d(img_tensor, kernel, padding=1, groups=3)
    residual = torch.clamp(residual, -1.0, 1.0)
    return residual


class ArtifactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # Pri načítaní vlastných váh netreba defaultné
        self.backbone.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        freq = get_fft_features(x)
        resid = get_pixel_residuals(x)
        combined = torch.cat([x, freq, resid], dim=1)
        return self.sigmoid(self.backbone(combined))


# --- Pomocné funkcie ---

def prepare_input(img_path):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image_PIL.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)


def evaluate_folder(folder_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Používam zariadenie: {device}")

    # Načítanie modelu
    model = ArtifactDetector().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        print(f"Chyba: Súbor s váhami '{model_path}' neexistuje!")
        return

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not images:
        print("V priečinku sa nenašli žiadne obrázky.")
        return

    probabilities = []
    ai_count = 0

    print(f"Spracovávam {len(images)} obrázkov...")

    with torch.no_grad():
        for img_name in tqdm(images):
            img_path = os.path.join(folder_path, img_name)
            try:
                input_tensor = prepare_input(img_path).to(device)
                prob = model(input_tensor).item()
                probabilities.append(prob)

                if prob > 0.5:
                    ai_count += 1
            except Exception as e:
                print(f"Chyba pri spracovaní {img_name}: {e}")

    # Výpočty
    if probabilities:
        avg_prob = sum(probabilities) / len(probabilities)
        accuracy_percent = (avg_prob * 100) if avg_prob > 0.5 else ((1 - avg_prob) * 100)

        print("\n" + "=" * 30)
        print(f"VÝSLEDKY PRE: {folder_path}")
        print(f"Celkový počet obrázkov: {len(images)}")
        print(f"Označené ako AI: {ai_count}")
        print(f"Označené ako Real: {len(images) - ai_count}")
        print(f"Priemerná pravdepodobnost AI (istota modelu): {avg_prob * 100:.2f}%")
        print("=" * 30)
    else:
        print("Nepodarilo sa vyhodnotiť žiadne obrázky.")


if __name__ == "__main__":
    # TU NASTAV SVOJE CESTY
    FOLDER_TO_TEST = "testingPicsAI"
    MODEL_WEIGHTS = "ai_detector_weights_new.pth"

    evaluate_folder(FOLDER_TO_TEST, MODEL_WEIGHTS)