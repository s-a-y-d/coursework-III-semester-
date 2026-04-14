import numpy as np
import pywt

class SparsityAnalyzer:
    @staticmethod
    def get_energy_distribution(coeffs: np.ndarray):
        energy = np.sort(np.abs(coeffs.flatten())**2)[::-1]
        cumulative_energy = np.cumsum(energy)
        return cumulative_energy / cumulative_energy[-1] * 100
    def compare_methods(self, image, ridgelet_coeffs_list, p):
        ridg_flat = []
        for row_coeffs in ridgelet_coeffs_list:
            for level in row_coeffs:
                ridg_flat.extend(level.flatten())
        ridg_flat = np.array(ridg_flat)
        coeffs2d = pywt.wavedec2(image, 'db1')
        wave2d_flat = []
        for level in coeffs2d:
            if isinstance(level, tuple):
                for sub in level:
                    wave2d_flat.extend(sub.flatten())
            else:
                wave2d_flat.extend(level.flatten())
        wave2d_flat = np.array(wave2d_flat)
        energy_ridg = self.get_energy_distribution(ridg_flat)
        energy_wave = self.get_energy_distribution(wave2d_flat)
        return energy_ridg, energy_wave
    @staticmethod
    def find_n90(energy_curve):
        return np.where(energy_curve >= 90)[0][0] + 1
