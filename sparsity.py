import matplotlib.pyplot as plt
from preprocessing import ImagePreprocessor
from radon_engine import RadonEngine
from wavelet_stage import WaveletStage
from sparsity_analyzer import SparsityAnalyzer
from test_generator import SyntheticTests

def run_comparison():
    prep = ImagePreprocessor()
    try:
        _ = prep.process()
        p = prep.p
        print(f"На основе 'test image' установлено p = {p}")
    except Exception as e:
        print(f"Ошибка при определении p: {e}")
        return
    gen = SyntheticTests(p)
    radon = RadonEngine(p)
    wavelet_1d = WaveletStage(wavelet_name='db1')
    analyzer = SparsityAnalyzer()
    tests = {
        "Полоса": gen.generate_stripe("diagonal"),
        "Крест": gen.generate_cross(),
        "Окно": gen.generate_window(),
        "V-образный": gen.generate_v_shape()
    }
    print(f"\n{'Тест':<20} | {'N90 Риджлет':<12} | {'N90 2D-вейвлет':<10} | {'Отношение'}")
    print("-" * 65)
    results = {}
    for name, img in tests.items():
        sinogram = radon.transform(img)
        ridg_coeffs = wavelet_1d.transform_sinogram(sinogram)
        e_ridg, e_wave = analyzer.compare_methods(img, ridg_coeffs, p)
        n90_r = analyzer.find_n90(e_ridg)
        n90_w = analyzer.find_n90(e_wave)
        ratio = n90_w / n90_r
        results[name] = (e_ridg, e_wave)
        print(f"{name:<20} | {n90_r:<12} | {n90_w:<10} | {ratio:.1f}x")
    if "V-образный" in results:
        e_ridg, e_wave = results["V-образный"]
        plt.figure(figsize=(10, 6))
        limit = max(100, int(len(e_ridg) * 0.1))
        plt.plot(e_ridg[:limit], label='Энергия риджлета', color='red', linewidth=2)
        plt.plot(e_wave[:limit], label='Энергия 2D-вейвлета', color='blue', linestyle='--')
        plt.axhline(y=90, color='green', linestyle=':', label='Энергетический порог (90%)')
        plt.title(f"Анализ разрежённости (p={p}): V-образный тест")
        plt.ylabel("Энергия, %")
        plt.xlabel("Число коэффициентов")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

run_comparison()
