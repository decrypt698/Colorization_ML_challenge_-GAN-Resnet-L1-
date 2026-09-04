import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import lab_to_rgb_image, ColorizationDataset
from model import ResUNet, Discriminator          # importe les DEUX
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
import os

BATCH_SIZE = 16
NUM_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "jpg"
SAVE_DIR = "results"

bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()
L1_LAMBDA = 100

def train_gan_epoch(gen, disc, loader, opt_gen, opt_disc, epoch):
    gen.train(); disc.train()
    total_g, total_d = 0.0, 0.0
    loop = tqdm(loader, leave=True)
    for batch in loop:
        L  = batch['L'].to(DEVICE)
        ab = batch['ab'].to(DEVICE)
        # --- Policier (D) ---
        fake_ab = gen(L)
        D_real = disc(L, ab)
        D_fake = disc(L, fake_ab.detach())
        D_loss = (bce(D_real, torch.ones_like(D_real)) +
                  bce(D_fake, torch.zeros_like(D_fake))) / 2
        opt_disc.zero_grad(); D_loss.backward(); opt_disc.step()
        # --- Faussaire (G) ---
        D_fake = disc(L, fake_ab)
        G_loss = bce(D_fake, torch.ones_like(D_fake)) + l1(fake_ab, ab) * L1_LAMBDA
        opt_gen.zero_grad(); G_loss.backward(); opt_gen.step()
        total_g += G_loss.item(); total_d += D_loss.item()
        loop.set_postfix(G=G_loss.item(), D=D_loss.item(), epoch=epoch)
    return total_g / len(loader), total_d / len(loader)

@torch.no_grad()
def validate(gen, loader):
    gen.eval()
    total = 0.0
    for batch in loader:
        L = batch['L'].to(DEVICE); ab = batch['ab'].to(DEVICE)
        total += l1(gen(L), ab).item()
    return total / len(loader)

def save_some_examples(model, val_loader, epoch, folder):
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        l_channel, ab_channels = batch['L'].to(DEVICE), batch['ab'].to(DEVICE)
        predicted_ab = model(l_channel)
        for i in range(min(4, l_channel.size(0))):
            true_rgb = lab_to_rgb_image(l_channel[i], ab_channels[i])
            pred_rgb = lab_to_rgb_image(l_channel[i], predicted_ab[i])
            comparison = torch.cat([torch.from_numpy(true_rgb).permute(2, 0, 1),
                                    torch.from_numpy(pred_rgb).permute(2, 0, 1)], dim=2)
            save_image(comparison.float() / 255.0,
                       os.path.join(folder, f"epoch_{epoch}_sample_{i}.png"))

def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    full_dataset = ColorizationDataset(DATA_DIR)
    n = len(full_dataset)
    n_test = int(0.1 * n); n_val = int(0.1 * n); n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42))
    print(f"{n} images -> train {n_train} | val {n_val} | test {n_test}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False)

    gen  = ResUNet().to(DEVICE)                     # le faussaire
    disc = Discriminator(in_channels=5).to(DEVICE)  # 5 = L(3) + ab(2) avec ResNet
    opt_gen  = optim.Adam(gen.parameters(),  lr=2e-4, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        g_loss, d_loss = train_gan_epoch(gen, disc, train_loader, opt_gen, opt_disc, epoch)
        val_loss = validate(gen, val_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | G {g_loss:.4f} | D {d_loss:.4f} | val L1 {val_loss:.4f}")
        save_some_examples(gen, val_loader, epoch, SAVE_DIR)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(gen.state_dict(), "best_gan_generator.pth")
            print(f"    -> nouveau meilleur générateur (val {best_loss:.4f})")

if __name__ == "__main__":
    main()