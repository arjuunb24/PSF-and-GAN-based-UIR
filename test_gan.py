import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
import cv2
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ================= UIQM IMPLEMENTATION =================
def _uicm(img):
    img = img.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    alpha_L, alpha_R = 0.1, 0.1

    RG = R - G
    YB = (R + G) / 2 - B

    RG_sort = np.sort(RG.flatten())
    YB_sort = np.sort(YB.flatten())

    num = len(RG_sort)
    RG_sort = RG_sort[int(num*alpha_L):int(num*(1-alpha_R))]
    YB_sort = YB_sort[int(num*alpha_L):int(num*(1-alpha_R))]

    u_RG, u_YB = np.mean(RG_sort), np.mean(YB_sort)
    sigma_RG, sigma_YB = np.std(RG_sort), np.std(YB_sort)

    return -0.0268 * np.sqrt(u_RG**2 + u_YB**2) + 0.1586 * np.sqrt(sigma_RG**2 + sigma_YB**2)

def _uism(img, window_size=5):
    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    def sobel(channel):
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    Rs, Gs, Bs = sobel(R), sobel(G), sobel(B)

    lambda_R, lambda_G, lambda_B = 0.299, 0.587, 0.114
    return lambda_R * np.mean(Rs) + lambda_G * np.mean(Gs) + lambda_B * np.mean(Bs)

def _uiconm(img, window_size=5):
    img = img.astype(np.float32)
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]

    def contrast(channel):
        num_windows_h = channel.shape[0] // window_size
        num_windows_w = channel.shape[1] // window_size
        c = 0
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                patch = channel[i*window_size:(i+1)*window_size, j*window_size:(j+1)*window_size]
                max_val = np.max(patch)
                min_val = np.min(patch)
                if max_val + min_val > 0:
                    c += (max_val - min_val) / (max_val + min_val)
        total_windows = num_windows_h * num_windows_w
        if total_windows == 0:
            return 0
        return c / total_windows

    lambda_R, lambda_G, lambda_B = 0.299, 0.587, 0.114
    return lambda_R * contrast(R) + lambda_G * contrast(G) + lambda_B * contrast(B)

def compute_uiqm(img_pil):
    img = np.array(img_pil)
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return c1 * _uicm(img) + c2 * _uism(img) + c3 * _uiconm(img)

# ================= PHYSICS-INFORMED POST-GAN RESTORATION =================
# Goal: complement pix2pix GAN outputs while protecting PSNR/SSIM:
# - Model residual forward-scattering blur as convolution with Gaussian PSF.
# - Use regularized Wiener deconvolution (stable vs Richardson–Lucy).
# - Use gating: if GAN output already sharp, skip deconv and only denoise.
# - Apply edge-preserving bilateral filtering to suppress ringing/grain.

def _estimate_sigma_from_blur(img_pil, sigma_min=0.7, sigma_max=1.6):
    gray = np.asarray(img_pil.convert("L")).astype(np.float32) / 255.0
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
    )
    var = float(lap.var())
    v_low, v_high = 1e-4, 5e-3
    v = max(v_low, min(v_high, var))
    sharp = (v - v_low) / (v_high - v_low)
    sigma = sigma_max - sharp * (sigma_max - sigma_min)
    return float(sigma), var

def _gaussian_psf(ksize, sigma):
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    psf /= (psf.sum() + 1e-8)
    return psf

def _wiener_deconv_channel(img, psf, K):
    H, W = img.shape
    psf_pad = np.zeros((H, W), dtype=np.float32)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    psf_pad = np.roll(psf_pad, -kh // 2, axis=0)
    psf_pad = np.roll(psf_pad, -kw // 2, axis=1)

    G = np.fft.fft2(img)
    Hf = np.fft.fft2(psf_pad)

    H_conj = np.conj(Hf)
    denom = (np.abs(Hf) ** 2 + K)
    F_hat = (H_conj / denom) * G
    out = np.real(np.fft.ifft2(F_hat))
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def physics_informed_restore_postgan(
    gan_output_pil,
    ksize=21,
    sigma_min=0.7,
    sigma_max=1.6,
    K_min=0.02,
    K_max=0.08,
    sharp_var_threshold=0.0020,
):
    sigma, var = _estimate_sigma_from_blur(gan_output_pil, sigma_min, sigma_max)

    img_bgr = cv2.cvtColor(np.array(gan_output_pil), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    # If already sharp, do mild denoise only
    if var >= sharp_var_threshold:
        den = cv2.bilateralFilter((img_bgr * 255).astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
        out_rgb = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)

    t = (sigma - sigma_min) / (sigma_max - sigma_min + 1e-8)
    K = K_max - t * (K_max - K_min)

    if ksize % 2 == 0:
        ksize += 1
    psf = _gaussian_psf(ksize, sigma)

    out = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        out[..., c] = _wiener_deconv_channel(img_bgr[..., c], psf, K)

    out_u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    out_dn = cv2.bilateralFilter(out_u8, d=5, sigmaColor=35, sigmaSpace=35)

    out_rgb = cv2.cvtColor(out_dn, cv2.COLOR_BGR2RGB)
    print(f"[PhysicsPost] sigma={sigma:.3f}, var={var:.6f}, K={K:.4f} (Wiener+bilateral)")
    return Image.fromarray(out_rgb)

# --- 1. DEFINING THE GENERATOR ARCHITECTURE ---
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

# --- 2. INFERENCE SCRIPT ---
def test_directory(input_dir, ref_dir, model_path, output_dir="output_results", max_images=9999, filter_str=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")

    generator = GeneratorUNet().to(device)
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg')) and (filter_str in f)][:max_images]

    # Separate lists for GAN vs Physics
    psnr_gan, ssim_gan, uiqm_gan = [], [], []
    psnr_phy, ssim_phy, uiqm_phy = [], [], []

    for i, img_name in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {img_name}")
        image_path = os.path.join(input_dir, img_name)
        ref_path = os.path.join(ref_dir, img_name)

        try:
            img = Image.open(image_path).convert("RGB")
            has_ref = os.path.exists(ref_path)
            if has_ref:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_img = ref_img.resize(img.size, Image.BICUBIC)
                ref_np = np.array(ref_img)
        except Exception as e:
            print(f"Could not read image: {e}")
            continue

        original_size = img.size

        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = generator(input_tensor)

        output_tensor = output_tensor.squeeze(0).cpu()
        output_tensor = (output_tensor * 0.5) + 0.5

        output_img = transforms.ToPILImage()(output_tensor)
        output_img = output_img.resize(original_size, Image.BICUBIC)

        # ------------------ (1) SAVE GAN OUTPUT ------------------
        gan_out_path = os.path.join(output_dir, f"restored_{img_name}")
        output_img.save(gan_out_path)
        print(f"Saved GAN to: {gan_out_path}")

        # ------------------ (2) PHYSICS POST-PROCESS ------------------
        physics_img = physics_informed_restore_postgan(output_img)

        phy_out_path = os.path.join(output_dir, f"physics_{img_name}")
        physics_img.save(phy_out_path)
        print(f"Saved Physics to: {phy_out_path}")

        # ------------------ METRICS: GAN ------------------
        res_gan = np.array(output_img)
        uiqm_val_g = compute_uiqm(output_img)
        uiqm_gan.append(uiqm_val_g)
        print(f"[GAN] UIQM: {uiqm_val_g:.4f}")

        if has_ref and res_gan.shape == ref_np.shape:
            try:
                psnr_val_g = compute_psnr(ref_np, res_gan)
                ssim_val_g = compute_ssim(ref_np, res_gan, channel_axis=-1)
                psnr_gan.append(psnr_val_g)
                ssim_gan.append(ssim_val_g)
                print(f"[GAN] PSNR: {psnr_val_g:.4f} | SSIM: {ssim_val_g:.4f}")
            except ValueError as e:
                print(f"[GAN] Skipped PSNR/SSIM due to error: {e}")

        # ------------------ METRICS: GAN + PSF ------------------
        res_phy = np.array(physics_img)
        uiqm_val_p = compute_uiqm(physics_img)
        uiqm_phy.append(uiqm_val_p)
        print(f"[PSF] UIQM: {uiqm_val_p:.4f}")

        if has_ref and res_phy.shape == ref_np.shape:
            try:
                psnr_val_p = compute_psnr(ref_np, res_phy)
                ssim_val_p = compute_ssim(ref_np, res_phy, channel_axis=-1)
                psnr_phy.append(psnr_val_p)
                ssim_phy.append(ssim_val_p)
                print(f"[PSF] PSNR: {psnr_val_p:.4f} | SSIM: {ssim_val_p:.4f}")
            except ValueError as e:
                print(f"[PSF] Skipped PSNR/SSIM due to error: {e}")

    print("\n" + "="*50)
    print("FINAL AVERAGE METRICS (GAN vs GAN + PSF)")
    print("="*50)

    if psnr_gan:
        print(f"[PLAIN GAN] Avg PSNR: {np.mean(psnr_gan):.4f} | Avg SSIM: {np.mean(ssim_gan):.4f}")
    if uiqm_gan:
        print(f"[PLAIN GAN] Avg UIQM: {np.mean(uiqm_gan):.4f}")

    print("-"*50)

    if psnr_phy:
        print(f"[GAN + PSF] Avg PSNR: {np.mean(psnr_phy):.4f} | Avg SSIM: {np.mean(ssim_phy):.4f}")
    if uiqm_phy:
        print(f"[GAN + PSF] Avg UIQM: {np.mean(uiqm_phy):.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python test_gan.py <input_dir> <reference_dir> <path_to_generator.pth> [filter_string]")
    else:
        filter_str = sys.argv[4] if len(sys.argv) > 4 else ""
        test_directory(sys.argv[1], sys.argv[2], sys.argv[3], filter_str=filter_str)