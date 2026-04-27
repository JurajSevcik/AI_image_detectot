import sys
sys.path.append('/users/asus/appdata/local/programs/python/python38/lib/site-packages')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image as Image_PIL
from PIL import ImageTk
import os
from tqdm import tqdm
import tkinter as tk
from tkinter import *
from tkinter import filedialog 

# define dataset class 
class AIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, class_name in enumerate(['real', 'fake']):
            #TODO: pouzit 3 trieedu 'upravené' -- zmena real obrazku cez AI
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in files:
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image_PIL.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# --- frature extracion 
#TODO: add features ()
#increse piyel residue matric size ?  
def get_fft_features(img_tensor):
    gray = 0.299 * img_tensor[:, 0] + 0.587 * img_tensor[:, 1] + 0.114 * img_tensor[:, 2]
    # FFT na ziskanie spektra 
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    # normalizacia 
    magnitude = torch.log(torch.abs(fft_shift) + 1e-8) 
    min_v = magnitude.view(magnitude.size(0), -1).min(1, keepdim=True)[0].unsqueeze(2)
    max_v = magnitude.view(magnitude.size(0), -1).max(1, keepdim=True)[0].unsqueeze(2)
    magnitude = (magnitude - min_v) / (max_v - min_v + 1e-8)
    return magnitude.unsqueeze(1) # [Batch, 1, 512, 512]

def get_pixel_residuals(img_tensor):
    # kontrola nekonzistencie medzi susednimi pixelmi
    # This matric specifically targets the "Checkerboard" artifact.
    #TODO: change matrix 
    kernel = torch.tensor([[-1., -1., -1.],
                           [-1.,  8., -1.],
                           [-1., -1., -1.]], 
                           device=img_tensor.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(3, 1, 1, 1)
    residual = torch.nn.functional.conv2d(img_tensor, kernel, padding=1, groups=3)
    residual = torch.clamp(residual, -1.0, 1.0) # normalizacia 
    return residual


# --- zostavenie modelu ---

class ArtifactDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        freq = get_fft_features(x) 
        resid = get_pixel_residuals(x)
        combined = torch.cat([x, freq, resid], dim=1)
        return self.sigmoid(self.backbone(combined))

# --- Trenovanie ---

# this is coppyed funcion I am not shure how exactli does it work 
def run_epoch(model, loader, criterion, optimizer, device, phase):
    if phase == 'train': #trenovanie
        model.train()
    else: # evaluacia 
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(loader, desc=f"{phase.capitalize()}", leave=False)
    #progres bar 

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()

            if phase == 'train':
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc

# --- Main funkcia --- train
def main():
    #data na trenovanie a validaciu
    TRAINING_DATASET = 'dataset/train'
    VALIDATION_DATASET = 'dataset/val'
    # priprava dat
    data_transforacia = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # pracujeme s 1024x1024, moze odstranovavt informacie,
    # TODO: mozno pouzit ine rozlisenie 
    # normalizacia podla pouziteho modelu 'resnet18'

    trenovaci_dataset = AIDataset(TRAINING_DATASET, transform=data_transforacia)
    loader = DataLoader(trenovaci_dataset, batch_size=32, shuffle=True)
    pouzivam = "cpu"
    epoch_num = 5
    model = ArtifactDetector().to(torch.device(pouzivam)) 
    # bude sa trenovať na CPU alebo GPU, ak je "cuda" zmen 
    #TODO: change to "cuda" if available 
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()

    print(f"training on {len(trenovaci_dataset)} images... \nThis may take a minute")
    for epoch in range(epoch_num):
        trening_loss, trening_accuracy = run_epoch(model, loader, criterion,  optimizer, torch.device(pouzivam), 'train')
        print(f"Epoch {epoch+1}/{epoch_num} | Loss: {trening_loss:.4f} | Acc: {trening_accuracy:.4f}")

    torch.save(model.state_dict(), "ai_detector_weights_new.pth")
    print("Model ulozeny ako ai_detector_weights_new.pth")

def prepare_input(img_path):
    # Standard transforms
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image_PIL.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0) # Vráti [1, 3, 512, 512]

    return x


def test_prediction(image_path, model_path):
    device = torch.device("cpu") # zmen na cuda ak dostupna
    
    # Load Model
    model = ArtifactDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prepare Image
    input_tensor = prepare_input(image_path).to(device)

    with torch.no_grad():
        probability = model(input_tensor).item()
    
    print(f"\n--- vysledok pre: {image_path} ---")
    print(f"AI Probability: {probability * 100:.2f}%")
    global dispaly_verdict
    outcome = ""
    if probability > 0.5: #rozhodovacia hranica
        outcome = "Verdict: AI GENERATED"
        dispaly_verdict = tk.Label(window, text=outcome, fg="red", font=15)
    else:
        outcome = "Verdict: REAL PHOTOGRAPH"
        dispaly_verdict = tk.Label(window, text=outcome, fg="green", font=15)
    print(outcome)
    text_out = str(f"AI Probability: {probability * 100:.2f}%")
    
    global dispaly_probability 
    
    dispaly_probability = tk.Label(window, text=text_out)
    dispaly_verdict.pack()
    dispaly_probability.pack()
    

    window.mainloop()


test_image = ""

def openFile():
    file = (filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")]
                                      ))
    print(file)
    global test_image 
    test_image = file

    im = Image_PIL.open(test_image)
    im = im.resize((400, 300))
    tkimage = ImageTk.PhotoImage(im)
    global myvar
    myvar=Label(window,image = tkimage)
    myvar.image = tkimage
    myvar.pack()
    window.mainloop()
    
    return file

def print_test_path():
    print("test path:" + test_image)

def test_new_image():
    myvar.destroy()
    dispaly_probability.destroy()
    dispaly_verdict.destroy()

if __name__ in {"__main__", "__mp_main__"}:
    TRAIN = 0 # 1 = trenovanie, 0 = testovanie
    model_path="ai_detector_weights_new.pth"
    #test_image = "C:/Users/Asus/Documents/Leto25_26/Nový priečinok/AI_image_detectot/dataset/test_images/not_raw.jpg" # Update this!
    

    if TRAIN == 1:
        main()
    else: 
        path = "C:/Users/Asus/Documents/Leto25_26/Nový priečinok/"
        window = Tk()
        window.title("AI tester")
        window.geometry('500x500')
        button_open = Button(text="Open image", command=openFile, height = 2, width = 15).pack()
        print(test_image)
        #button_path = Button(text="path", command=lambda : print_test_path()).pack()
        button_test = Button(text="Test Image", command=lambda : test_prediction(test_image, model_path), height = 2, width = 15).pack()
        button_restart = Button(text="restart", command= lambda : test_new_image(), height = 2, width = 15).pack()
        window.mainloop()
        