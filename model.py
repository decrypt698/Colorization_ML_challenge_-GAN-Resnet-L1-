import torch
import torch.nn as nn
import torchvision


# ---- Bloc de MONTÉE (décodeur) : agrandit + recolle le skip + LeakyReLU ----
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)  # x2 la taille
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),   # 👈 LeakyReLU ici
        )

    def forward(self, x, skip):
        x = self.up(x)                      # on agrandit
        x = torch.cat([x, skip], dim=1)     # on colle le souvenir (skip)
        return self.conv(x)

# policier (GAN) : il regarde l'image et dit vrai/faux
class Discriminator(nn.Module):
    def __init__(self,in_channels=3):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,kernel_size=4,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512,1,kernel_size=4,stride=1,padding=1) #sortie :carte vrai/faux
        )
    def forward(self,L, ab):
        x=torch.cat([L, ab], dim=1) #concaténation des canaux gris+couleur
        return self.net(x)
    
# ---- Le réseau complet ----
class ResUNet(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        # On charge un ResNet18 DÉJÀ entraîné (il sait déjà voir contours/textures)
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        # --- ENCODEUR = les morceaux du ResNet (la descente) ---
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 canaux, taille/2
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1   # 64 canaux
        self.enc2 = resnet.layer2   # 128 canaux
        self.enc3 = resnet.layer3   # 256 canaux
        self.enc4 = resnet.layer4   # 512 canaux (le plus profond)

        # --- DÉCODEUR = la montée (fait maison), avec skip-connections ---
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(128, 64, 64)
        self.up4 = Up(64, 64, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # revient à la taille de départ
            nn.Conv2d(32, out_channels, kernel_size=1),           # sort 2 canaux (a, b)
            nn.Tanh(),                                            # sortie entre -1 et 1
        )

    def forward(self, x):
        x0 = self.enc0(x)              # descente...
        x1 = self.enc1(self.pool(x0))
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)             # point le plus profond
        u = self.up1(x4, x3)           # montée avec skips
        u = self.up2(u, x2)
        u = self.up3(u, x1)
        u = self.up4(u, x0)
        return self.final(u)           # image couleur (a, b)

    def freeze_encoder(self):
        """Gèle le ResNet : on ne ré-entraîne QUE le décodeur (comme dit le sujet)."""
        for enc in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for p in enc.parameters():
                p.requires_grad = False