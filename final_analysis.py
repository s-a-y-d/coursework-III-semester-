import torch
import numpy as np
import matplotlib.pyplot as plt
from model import RidgeletInterpretNet
from preprocessing import ImagePreprocessor
from radon_engine import RadonEngine

def get_ridgelet_atom(p, k, l):
    engine = RadonEngine(p)
    sinogram = np.zeros((p + 1, p), dtype=np.float32)
    sinogram[k, l] = 1.0
    atom = engine.inverse_transform(sinogram)
    atom = (atom - atom.mean()) / (atom.std() + 1e-8)
    return atom

def find_best_match(nn_atom, p, k):
    best_sim = -1.0
    best_l = 0
    best_math_atom = None
    for l in range(p):
        math_atom = get_ridgelet_atom(p, k, l)
        sim = np.abs(np.dot(nn_atom.flatten(), math_atom.flatten()) / (
                np.linalg.norm(nn_atom) * np.linalg.norm(math_atom) + 1e-8
        ))
        if sim > best_sim:
            best_sim = sim
            best_l = l
            best_math_atom = math_atom
    return best_sim, best_l, best_math_atom

def run_final_analysis():
    prep = ImagePreprocessor()
    img_array = prep.process()
    p = prep.p
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    model = RidgeletInterpretNet(p=p)
    try:
        model.load_state_dict(torch.load("ridgelet_net.pth"))
        model.eval()
        print(f"Модель для p={p} успешно загружена.")
    except FileNotFoundError:
        print("Ошибка: Файл ridgelet_net.pth не найден. Сначала запустите train.py.")
        return
    with torch.no_grad():
        logits = model(img_tensor)
        predicted_k = torch.argmax(logits, dim=1).item()
    print(f"--- Результаты анализа ---")
    print(f"Доминирующий наклон в 'test image': k = {predicted_k}")
    nn_atoms = model.get_weights_as_images().numpy()
    nn_atom_raw = nn_atoms[predicted_k]
    nn_atom = (nn_atom_raw - nn_atom_raw.mean()) / (nn_atom_raw.std() + 1e-8)
    sim, l_opt, math_atom = find_best_match(nn_atom, p, predicted_k)
    print(f"Максимальное косинусное сходство: {sim:.4f}")
    print(f"Оптимальное смещение l: {l_opt}")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Исходная тестовая картинка\n(обнаружено k={predicted_k})")
    plt.subplot(1, 3, 2)
    plt.imshow(nn_atom_raw, cmap='RdBu_r')
    plt.title(f"Веса нейронов")
    plt.subplot(1, 3, 3)
    plt.imshow(math_atom, cmap='RdBu_r')
    plt.title(f"Риджлет-атом\n(k={predicted_k}, l={l_opt})\nсходство: {sim:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_final_analysis()
