import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor
from radon_engine import RadonEngine
from wavelet_stage import WaveletStage

def run_experiment():
    preprocessor = ImagePreprocessor()
    wavelet_worker = WaveletStage(wavelet_name='db1')
    print("--- Этап 1: Предобработка ---")
    img_prime = preprocessor.process()
    p = preprocessor.p
    print(f"--- Этап 2: Преобразование Радона (p={p}) ---")
    radon_engine = RadonEngine(p=p)
    sinogram = radon_engine.transform(img_prime)
    print("--- Этап 3: Вейвлет ---")
    ridgelet_map = wavelet_worker.get_feature_map(sinogram)
    full_coeffs = wavelet_worker.transform_sinogram(sinogram)
    print("--- Визуализация ---")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_prime, cmap='gray')
    ax[0].set_title("Исходное (p x p)")
    ax[1].imshow(sinogram, cmap='hot')
    ax[1].set_title("Синограмма Радона")
    ax[2].imshow(ridgelet_map, cmap='inferno')
    ax[2].set_title("Риджлет-коэффициенты")
    plt.tight_layout()
    plt.show()
    return full_coeffs

coeffs = run_experiment()
print("\nЦепочка успешно завершена. Данные готовы для нейросети.")
