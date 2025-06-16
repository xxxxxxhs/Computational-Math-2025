import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# helpers ------------------------------------------------

def read_points_interactive():
    print("Введите пары x y (пустая строка — конец ввода):")
    pts = []
    while True:
        line = input().strip()
        if not line:
            break
        parts = line.replace(",", ".").split()
        if len(parts) != 2:
            print("Ожидалось два числа через пробел")
            continue
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def read_points_from_file() -> list:
    fname = input("Имя файла с данными: ").strip()
    path = Path(fname)
    if not path.exists():
        print("Файл не найден"); sys.exit(1)
    pts = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", ".").split()
        if len(parts) != 2:
            continue
        pts.append((float(parts[0]), float(parts[1])))
    return pts


def request_points():
    mode = input("Читать точки из файла? (y/n): ").strip().lower()
    pts = read_points_from_file() if mode == "y" else read_points_interactive()
    if not (8 <= len(pts) <= 12):
        print("Нужно 8–12 точек"); sys.exit(1)
    xs, ys = map(np.array, zip(*pts))
    return xs, ys


# -----------------------------------------------------
# метрики ------------------------------------------------

# Сигма - среднеквадратичное отклонение
def rms(pred, real):
    return math.sqrt(np.mean((pred - real) ** 2))
# R^2 -
def r2_score(pred, real):
    ss_res = np.sum((real - pred) ** 2)
    ss_tot = np.sum((real - real.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else 0.0


# -----------------------------------------------------
# модели ------------------------------------------------

def model_linear(xs, ys):
    c = np.polyfit(xs, ys, 1)
    return np.polyval(c, xs), c

def model_poly2(xs, ys):
    c = np.polyfit(xs, ys, 2)
    return np.polyval(c, xs), c

def model_poly3(xs, ys):
    c = np.polyfit(xs, ys, 3)
    return np.polyval(c, xs), c

def model_expo(xs, ys):
    if np.any(ys <= 0):
        return None, None
    b, ln_a = np.polyfit(xs, np.log(ys), 1)
    a = math.exp(ln_a)
    return a * np.exp(b * xs), (a, b)

def model_log(xs, ys):
    if np.any(xs <= 0):
        return None, None
    b, a = np.polyfit(np.log(xs), ys, 1)
    return a + b * np.log(xs), (a, b)

def model_power(xs, ys):
    if np.any(xs <= 0) or np.any(ys <= 0):
        return None, None
    b, ln_a = np.polyfit(np.log(xs), np.log(ys), 1)
    a = math.exp(ln_a)
    return a * xs ** b, (a, b)

MODELS = {
    "linear":   (model_linear,    "y = a₀ + a₁ x"),
    "poly2":    (model_poly2,     "y = a₂ x² + a₁ x + a₀"),
    "poly3":    (model_poly3,     "y = a₃ x³ + a₂ x² + a₁ x + a₀"),
    "expo":     (model_expo,      "y = A · e^{Bx}"),
    "log":      (model_log,       "y = A + B · ln x"),
    "power":    (model_power,     "y = A · x^{B}"),
}

# -----------------------------------------------------
# main --------------------------------------

def main():
    print("Будут исследоваться следующие формы зависимостей:")
    for name, (_, formula) in MODELS.items():
        print(f"  {name:6s}: {formula}")
    print()

    xs, ys = request_points()

    results = {}
    for name, (fn, _) in MODELS.items():
        pred, coeffs = fn(xs, ys)
        if pred is None:
            continue
        results[name] = {
            "sigma": rms(pred, ys),
            "coeffs": coeffs,
            "pred": pred,
            "r2": r2_score(pred, ys),
        }

    if not results:
        print("Не удалось построить ни одной модели"); sys.exit(1)

    best_name = min(results, key=lambda n: results[n]["sigma"])

    out_mode = input("Сохранить результаты в файл? (y/n): ").strip().lower()
    if out_mode == "y":
        out_path = Path(input("Имя выходного файла: ").strip())
        with out_path.open("w", encoding="utf-8") as f:
            for name, res in results.items():
                f.write(f"{name}: σ={res['sigma']:.3f}  R²={res['r2']:.3f}  coeffs={res['coeffs']}\n")
            f.write(f"Лучшее приближение: {best_name} (σ={results[best_name]['sigma']:.3f})\n")
            if "linear" in results:
                r = np.corrcoef(xs, ys)[0, 1]
                f.write(f"Корреляция Пирсона (linear): r={r:.3f}\n")
        print(f"Результаты сохранены в {out_path}")
    else:
        for name, res in results.items():
            print(f"{name:6s}: σ={res['sigma']:.3f}  R²={res['r2']:.3f}  coeffs={res['coeffs']}")
        print(f"Лучшее приближение => {best_name} (σ={results[best_name]['sigma']:.3f})")
        if "linear" in results:
            r = np.corrcoef(xs, ys)[0, 1]
            print(f"Корреляция Пирсона (linear): r={r:.3f}")

    plt.scatter(xs, ys, label="data", zorder=3)
    pu = np.linspace(xs.min(), xs.max(), 400)
    for name, res in results.items():
        if name.startswith("poly"):
            y_plot = np.polyval(res["coeffs"], pu)
        elif name == "linear":
            y_plot = res["coeffs"][0] * pu + res["coeffs"][1]
        elif name == "expo":
            y_plot = res["coeffs"][0] * np.exp(res["coeffs"][1] * pu)
        elif name == "log":
            y_plot = res["coeffs"][0] + res["coeffs"][1] * np.log(pu)
        elif name == "power":
            y_plot = res["coeffs"][0] * pu ** res["coeffs"][1]
        else:
            continue
        plt.plot(pu, y_plot, label=name)
    plt.title("Аппроксимация точек")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
