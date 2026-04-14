import pywt
import numpy as np

class WaveletStage:
    def __init__(self, wavelet_name: str = 'db1'):
        self.wavelet_name = wavelet_name
    def transform_sinogram(self, sinogram: np.ndarray) -> list:
        ridgelet_coeffs = []
        for row in sinogram:
            coeffs = pywt.wavedec(row, self.wavelet_name, mode='periodization')
            ridgelet_coeffs.append(coeffs)
        return ridgelet_coeffs
    def get_feature_map(self, sinogram: np.ndarray) -> np.ndarray:
        rows, cols = sinogram.shape
        feature_map = np.zeros_like(sinogram)
        for i, row in enumerate(sinogram):
            _, cD = pywt.dwt(row, self.wavelet_name, mode='periodization')
            feature_map[i, :len(cD)] = cD
        return feature_map
    def inverse_transform_sinogram(self, ridgelet_coeffs_list):
        reconstructed_sinogram = []
        for coeffs in ridgelet_coeffs_list:
            row = pywt.waverec(coeffs, self.wavelet_name, mode='periodization')
            reconstructed_sinogram.append(row[:self.p_size_placeholder])
        return np.array(reconstructed_sinogram)
