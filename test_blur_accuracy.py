import os
import cv2
import numpy as np
from PIL import Image
import sys

# Import our estimation function from your existing script
from test_gan import _estimate_sigma_from_blur

def apply_synthetic_blur(img_pil, sigma):
    """Applies mathematically precise Gaussian blur using OpenCV."""
    img_np = np.array(img_pil)
    # ksize=(0,0) forces cv2 to compute the kernel size based directly on our precise sigma
    blurred_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred_np)

def test_blur_accuracy(ref_dir, sigmas_to_test, max_images=500):
    image_files = [f for f in os.listdir(ref_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:max_images]
    
    if not image_files:
        print(f"Error: No images found in {ref_dir}")
        return

    print(f"Evaluating Blur Estimation Accuracy on {len(image_files)} Reference Images...")
    
    overall_errors = []
    
    print("\n" + "="*50)
    print("BLUR ESTIMATION ACCURACY RESULTS")
    print("="*50)
    
    for idx, true_sigma in enumerate(sigmas_to_test):
        predictions = []
        errors = []
        
        for img_name in image_files:
            img_path = os.path.join(ref_dir, img_name)
            try:
                # Load sharp reference image
                img_pil = Image.open(img_path).convert("RGB")
                
                # Resize to match network handling size (variance depends heavily on resolution!)
                img_pil = img_pil.resize((256, 256), Image.BICUBIC) 
                
                # 1. Apply true synthetic blur
                blurred_pil = apply_synthetic_blur(img_pil, true_sigma)
                
                # 2. Predict blur using our algorithm 
                # (Notice we unpack the tuple because your function returns (sigma, var))
                pred_sigma, variance = _estimate_sigma_from_blur(blurred_pil, sigma_min=0.7, sigma_max=1.6)
                
                predictions.append(pred_sigma)
                errors.append(abs(true_sigma - pred_sigma))
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        
        if not predictions:
            print(f"Failed to process images for sigma {true_sigma}")
            continue
            
        # 3. Calculate Metrics
        avg_pred = np.mean(predictions)
        mae = np.mean(errors)
        std_dev = np.std(predictions)
        
        overall_errors.extend(errors)
        
        print(f"\n[TEST {idx+1}] Artificial True Blur (Sigma) = {true_sigma:.2f}")
        print(f"  - Average Predicted Sigma : {avg_pred:.2f}")
        print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
        print(f"  - Consistency (Std Dev)   : {std_dev:.2f}")

    # Final Overall Summary
    if overall_errors:
        overall_mae = np.mean(overall_errors)
        overall_rmse = np.sqrt(np.mean(np.array(overall_errors)**2))
        print("\n" + "="*50)
        print("SUMMARY OVERALL:")
        print(f"  - Overall Mean Absolute Error (MAE): {overall_mae:.3f}")
        print(f"  - Overall Root Mean Square Error (RMSE): {overall_rmse:.3f}")
        print("="*50)

if __name__ == "__main__":
    reference_directory = r"data\test\reference"
    
    # The true mathematical sigmas we want to test
    test_sigmas = [1.0, 1.5, 2.0]
    
    # Run the test on a sample of 250 images for speed (you can change max_images to 9999 for all)
    test_blur_accuracy(reference_directory, test_sigmas, max_images=9999)
