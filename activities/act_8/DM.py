import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# --- 1. Hyperparameters & Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_epochs = 100      # Your chosen sweet spot
batch_size = 128
T = 1000               
beta_start = 0.0001
beta_end = 0.02

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mnist_diffusion.pth")

betas = torch.linspace(beta_start, beta_end, T).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# --- 2. Model Architecture (U-Net) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64))
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(256, 64, 3, padding=1) 
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float().view(-1, 1))
        t_emb_64 = t_emb.view(-1, 64, 1, 1)
        t_emb_128 = t_emb.repeat(1, 2).view(-1, 128, 1, 1)
        
        x1 = F.relu(self.conv1(x) + t_emb_64)
        x1_pool = self.pool1(x1)
        x2 = F.relu(self.conv2(x1_pool) + t_emb_128)
        x2_pool = self.pool2(x2)
        x3 = F.relu(self.conv3(x2_pool))
        x_up1 = self.up1(x3)
        x_concat1 = torch.cat([x_up1, x2], dim=1) 
        x4 = F.relu(self.conv4(x_concat1))
        x_up2 = self.up2(x4)
        x_concat2 = torch.cat([x_up2, x1], dim=1)
        x5 = F.relu(self.conv5(x_concat2))
        return self.conv_out(x5)

def get_noisy_image(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# --- 3. Training Logic (Fresh Start) ---
def train_fresh(total_epochs):
    print(f"--- Starting FRESH Training on {device} ---")
    
    # IMPORTANT: Delete existing model if you want a truly clean slate
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("Old weights deleted. Starting from zero.")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4) # Slightly higher for fresh start
    
    best_loss = float('inf')

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, T, (images.shape[0],)).to(device)
            x_noisy, target_noise = get_noisy_image(images, t)
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(target_noise, predicted_noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{total_epochs} | Avg Loss: {avg_loss:.6f}")

        # Save only the best version
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_PATH)
            # print("  [Saved Best Model]")

    print(f"Final training complete. Best loss achieved: {best_loss:.6f}")

# --- 4. Loading & Sampling Logic ---
@torch.no_grad()
def load_and_sample(num_samples=6):
    model = SimpleUNet().to(device)
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print("Error: No model found to sample from!")
        return

    print("Generating digits from noise...")
    plt.figure(figsize=(15, 3))
    for s in range(num_samples):
        img = torch.randn((1, 1, 28, 28)).to(device)
        for i in reversed(range(T)):
            t = torch.tensor([i]).to(device)
            pred_noise = model(img, t)
            alpha, alpha_cumprod, beta = alphas[i], alphas_cumprod[i], betas[i]
            z = torch.randn_like(img) if i > 0 else 0
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_cumprod).sqrt()) * pred_noise) + beta.sqrt() * z

        plt.subplot(1, num_samples, s + 1)
        plt.imshow(img.cpu().squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- 5. Execution ---
if __name__ == "__main__":
    train_fresh(target_epochs)
    load_and_sample(num_samples=6)