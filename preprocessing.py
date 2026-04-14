import numpy as np
import cv2
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, target_p: int = None):
        self.p = target_p
    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2: return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0: return False
        return True
    def get_nearest_prime(self, n: int) -> int:
        while not self.is_prime(n):
            n += 1
        return n
    def find_test_image(self, base_name: str = "test image") -> Path:
        current_dir = Path(__file__).parent
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for file in current_dir.iterdir():
            if file.stem == base_name and file.suffix.lower() in valid_extensions:
                return file
        raise FileNotFoundError(f"Файл с названием '{base_name}' не найден в {current_dir}")
    def load_image(self, file_path: Path) -> np.ndarray:
        try:
            with open(file_path, "rb") as f:
                chunk = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(chunk, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            raise IOError(f"Не удалось прочитать файл через байтовый поток: {e}")
        if image is None:
            raise ValueError(f"Не удалось декодировать изображение. Проверьте файл: {file_path}")
        return image.astype(np.float32)
    def adjust_to_prime_size(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        max_side = max(h, w)
        if self.p is None:
            self.p = self.get_nearest_prime(max_side)
            print(f"Подобрано простое число p = {self.p}")
        padded_img = np.zeros((self.p, self.p), dtype=np.float32)
        top = (self.p - h) // 2
        left = (self.p - w) // 2
        padded_img[top:top + h, left:left + w] = image
        return padded_img
    def normalize(self, image: np.ndarray) -> np.ndarray:
        img_min, img_max = image.min(), image.max()
        if img_max - img_min == 0: return image
        return (image - img_min) / (img_max - img_min)
    def process(self) -> np.ndarray:
        file_path = self.find_test_image()
        print(f"Найден файл: {file_path.name}")
        raw_image = self.load_image(file_path)
        prime_image = self.adjust_to_prime_size(raw_image)
        return self.normalize(prime_image)
