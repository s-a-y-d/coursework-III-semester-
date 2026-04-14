import numpy as np

class RadonEngine:
    def __init__(self, p: int):
        self.p = p
    def transform(self, image: np.ndarray) -> np.ndarray:
        if image.shape != (self.p, self.p):
            raise ValueError(f"Размер изображения должен быть {self.p}x{self.p}")
        sinogram = np.zeros((self.p + 1, self.p), dtype=np.float32)
        i_coords, j_coords = np.indices((self.p, self.p))
        for k in range(self.p):
            l_map = (j_coords - k * i_coords) % self.p
            for l in range(self.p):
                sinogram[k, l] = np.sum(image[l_map == l])
        for l in range(self.p):
            sinogram[self.p, l] = np.sum(image[:, l])
        return sinogram
    def inverse_transform(self, sinogram: np.ndarray) -> np.ndarray:
        p = self.p
        img = np.zeros((p, p), dtype=np.float32)
        i, j = np.indices((p, p))
        for k in range(p):
            l = (j - k * i) % p
            img += sinogram[k, l]
        for l_vert in range(p):
            img[l_vert, :] += sinogram[p, l_vert]
        total_sum = np.sum(sinogram[0, :])
        img = (img - total_sum) / p
        return img
