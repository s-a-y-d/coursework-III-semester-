import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from preprocessing import ImagePreprocessor
from radon_engine import RadonEngine
from wavelet_stage import WaveletStage
import pywt

def add_noise(image, sigma=0.15):
    noise = np.random.normal(0, sigma, image.shape)
    return np.clip(image + noise, 0, 1)

def wavelet_2d_denoising(image, wavelet='db1', threshold=0.15):
    coeffs = pywt.wavedec2(image, wavelet, mode='periodization')
    coeffs_filt = [coeffs[0]]
    for detail_level in coeffs[1:]:
        coeffs_filt.append(tuple(pywt.threshold(c, threshold, mode='hard') for c in detail_level))
    return pywt.waverec2(coeffs_filt, wavelet, mode='periodization')

def run_robustness_test():
    prep = ImagePreprocessor()
    original = prep.process()
    H, W = original.shape
    p = prep.p
    noisy_img = add_noise(original, sigma=0.2)
    radon = RadonEngine(p)
    wavelet_1d = WaveletStage(wavelet_name='db1')
    wavelet_1d.p_size_placeholder = p
    sinogram = radon.transform(noisy_img)
    ridge_coeffs = wavelet_1d.transform_sinogram(sinogram)
    filtered_coeffs = []
    for row_coeffs in ridge_coeffs:
        filtered_coeffs.append([pywt.threshold(level, 0.12, mode='soft') for level in row_coeffs])
    rec_sinogram = wavelet_1d.inverse_transform_sinogram(filtered_coeffs)
    ridge_denoised_full = radon.inverse_transform(rec_sinogram)
    ridge_denoised = ridge_denoised_full[:H, :W]
    wave2d_denoised_full = wavelet_2d_denoising(noisy_img, threshold=0.15)
    wave2d_denoised = wave2d_denoised_full[:H, :W]
    noisy_crop = noisy_img[:H, :W]
    methods = {
        "Ввод с шумом": noisy_crop,
        "2D-вейвлет": wave2d_denoised,
        "Риджлет": ridge_denoised
    }
    print(f"\nСравнение устойчивости к шуму (p={p}):")
    print(f"{'Метод':<16} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-" * 40)
    results_img = []
    for name, img in methods.items():
        p_val = psnr(original, img, data_range=1.0)
        s_val = ssim(original, img, data_range=1.0)
        print(f"{name:<16} | {p_val:<10.2f} | {s_val:<6.3f}")
        results_img.append((name, img))
    plt.figure(figsize=(16, 4))
    titles = ["Исходное", "С шумом", "Обесшумленное 2D-вейвлетом", "Обесшумленное риджлетом"]
    images = [original, noisy_crop, wave2d_denoised, ridge_denoised]
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_robustness_test()
