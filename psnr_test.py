import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from preprocessing import ImagePreprocessor
from radon_engine import RadonEngine
from wavelet_stage import WaveletStage

def check_reconstruction():
    prep = ImagePreprocessor()
    original = prep.process()
    p = prep.p
    radon = RadonEngine(p)
    wavelet = WaveletStage(wavelet_name='db1')
    wavelet.p_size_placeholder = p
    sinogram = radon.transform(original)
    coeffs = wavelet.transform_sinogram(sinogram)
    rec_sinogram = wavelet.inverse_transform_sinogram(coeffs)
    reconstructed = radon.inverse_transform(rec_sinogram)
    val_psnr = psnr(original, reconstructed, data_range=original.max() - original.min())
    print(f"Реконструкция завершена.")
    print(f"PSNR: {val_psnr:.2f} dB")
    if val_psnr > 30:
        print("Результат отличный: потери минимальны.")
    else:
        print("Внимание: проверьте нормализацию в обратном Радоне.")

if __name__ == "__main__":
    check_reconstruction()
