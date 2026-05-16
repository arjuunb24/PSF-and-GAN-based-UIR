import torch
import os
import cv2
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# Import all your core tools from test_gan
from test_gan import (
    GeneratorUNet,
    compute_uiqm,
    _estimate_sigma_from_blur,
    _gaussian_psf,
    _wiener_deconv_channel
)

def run_ablation_variants(gan_output_pil):
    """
    Takes the GAN output and returns 5 variants corresponding exactly to your table:
    (a) GAN-only
    (b) GAN + Wiener (Fixed K)
    (c) GAN + Wiener (Adaptive K)
    (d) GAN + Wiener + Gating (Adaptive K but skipped if already sharp)
    (e) GAN + Wiener + Gating + Bilateral (The full physics approach)
    """
    # Base Constants
    ksize = 21
    sigma_min, sigma_max = 0.7, 1.6
    K_min, K_max = 0.02, 0.08
    sharp_var_threshold = 0.0020

    # Convert GAN output for processing
    img_bgr = cv2.cvtColor(np.array(gan_output_pil), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0

    # (a) GAN-only
    img_a = gan_output_pil.copy()

    # Shared Calculations for physics steps
    sigma, var = _estimate_sigma_from_blur(gan_output_pil, sigma_min, sigma_max)
    if ksize % 2 == 0: ksize += 1
    psf = _gaussian_psf(ksize, sigma)

    # Calculate Adaptive K value
    t = (sigma - sigma_min) / (sigma_max - sigma_min + 1e-8)
    adaptive_K = K_max - t * (K_max - K_min)

    # ----------------------------------------------------
    # (b) GAN + Wiener (Fixed K = 0.05) - NO Bilateral, NO Gating
    # ----------------------------------------------------
    out_b = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        out_b[..., c] = _wiener_deconv_channel(img_bgr[..., c], psf, 0.05)
    img_b = Image.fromarray(cv2.cvtColor(np.clip(out_b * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    # ----------------------------------------------------
    # (c) GAN + Wiener (Adaptive K) - NO Bilateral, NO Gating
    # ----------------------------------------------------
    out_c = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        out_c[..., c] = _wiener_deconv_channel(img_bgr[..., c], psf, adaptive_K)
    out_c_u8 = np.clip(out_c * 255.0, 0, 255).astype(np.uint8)
    img_c = Image.fromarray(cv2.cvtColor(out_c_u8, cv2.COLOR_BGR2RGB))

    # ----------------------------------------------------
    # (d) GAN + Wiener + Gating - NO Bilateral
    # ----------------------------------------------------
    if var >= sharp_var_threshold:
        img_d = img_a.copy() # Skip deconvolution entirely
    else:
        img_d = img_c.copy() # Use adaptive K deconvolution

    # ----------------------------------------------------
    # (e) GAN + Wiener + Gating + Bilateral (Full Current Pipeline)
    # ----------------------------------------------------
    if var >= sharp_var_threshold:
        # Gating says it's sharp -> only apply denoising (Bilateral)
        den = cv2.bilateralFilter((img_bgr * 255).astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
        img_e = Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    else:
        # Apply bilateral to the de-convolved adaptive output (out_c_u8)
        out_dn = cv2.bilateralFilter(out_c_u8, d=5, sigmaColor=35, sigmaSpace=35)
        img_e = Image.fromarray(cv2.cvtColor(out_dn, cv2.COLOR_BGR2RGB))

    return {'a': img_a, 'b': img_b, 'c': img_c, 'd': img_d, 'e': img_e}

def run_ablation_study(input_dir, ref_dir, model_path, max_images=9999):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Component Ablation on: {device}")

    # Load Model
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:max_images]
    print(f"Testing on {len(image_files)} images...")

    # Dictionary to hold all metrics
    metrics = {
        'a': {'psnr': [], 'ssim': [], 'uiqm': []},
        'b': {'psnr': [], 'ssim': [], 'uiqm': []},
        'c': {'psnr': [], 'ssim': [], 'uiqm': []},
        'd': {'psnr': [], 'ssim': [], 'uiqm': []},
        'e': {'psnr': [], 'ssim': [], 'uiqm': []}
    }

    for i, img_name in enumerate(image_files):
        print(f"\rProcessing image {i+1}/{len(image_files)}: {img_name}...   ", end="")
        sys.stdout.flush()

        image_path = os.path.join(input_dir, img_name)
        ref_path = os.path.join(ref_dir, img_name)
        
        has_ref = os.path.exists(ref_path)

        try:
            img = Image.open(image_path).convert("RGB")
            if has_ref:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_img = ref_img.resize((256, 256), Image.BICUBIC)
                ref_np = np.array(ref_img)
        except Exception as e:
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor = generator(input_tensor).squeeze(0).cpu()
            
        output_tensor = (output_tensor * 0.5) + 0.5
        gan_output_pil = transforms.ToPILImage()(output_tensor)
        gan_output_pil = gan_output_pil.resize((256,256), Image.BICUBIC)

        # GET ALL 5 VARIANTS
        variants = run_ablation_variants(gan_output_pil)

        # CALCULATE METRICS FOR EACH
        for key in ['a', 'b', 'c', 'd', 'e']:
            img_variant_pil = variants[key]
            
            # UIQM
            metrics[key]['uiqm'].append(compute_uiqm(img_variant_pil))
            
            # PSNR / SSIM
            if has_ref:
                var_np = np.array(img_variant_pil)
                if var_np.shape == ref_np.shape:
                    try:
                        metrics[key]['psnr'].append(compute_psnr(ref_np, var_np))
                        metrics[key]['ssim'].append(compute_ssim(ref_np, var_np, channel_axis=-1))
                    except ValueError:
                        pass # Ignore if image too small for window
    
    print("\n\n=======================================================")
    print("FINAL ABLATION TABLE RESULTS")
    print("=======================================================\n")
    print(f"{'Configuration':<35} | {'PSNR':<7} | {'SSIM':<7} | {'UIQM':<7}")
    print("-" * 65)

    names = {
        'a': "(a) GAN-only",
        'b': "(b) GAN + Wiener (Fixed K)",
        'c': "(c) GAN + Wiener (Adaptive K)",
        'd': "(d) GAN + Wiener + Gating",
        'e': "(e) GAN + Wiener + Gating + Bilateral"
    }

    for key in ['a', 'b', 'c', 'd', 'e']:
        psnr_avg = np.mean(metrics[key]['psnr']) if metrics[key]['psnr'] else 0.0
        ssim_avg = np.mean(metrics[key]['ssim']) if metrics[key]['ssim'] else 0.0
        uiqm_avg = np.mean(metrics[key]['uiqm']) if metrics[key]['uiqm'] else 0.0
        print(f"{names[key]:<35} | {psnr_avg:<7.4f} | {ssim_avg:<7.4f} | {uiqm_avg:<7.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python ablation_table.py <input_dir> <reference_dir> <path_to_generator.pth> [max_images]")
    else:
        max_imgs = int(sys.argv[4]) if len(sys.argv) > 4 else 9999
        run_ablation_study(sys.argv[1], sys.argv[2], sys.argv[3], max_images=max_imgs)
