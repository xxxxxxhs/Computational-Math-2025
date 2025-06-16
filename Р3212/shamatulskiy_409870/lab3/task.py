import math
import sys


def f1(x):
    """f(x) = x^2"""
    return x**2

def f2(x):
    """f(x) = sin(x)"""
    return math.sin(x)

def f3(x):
    """f(x) = e^x"""
    return math.exp(x)

def f4(x):
    """f(x) = 1/x (при x ≠ 0)"""
    return 1.0 / x

def f5(x):
    """f(x) = x^3 + 2x"""
    return x**3 + 2*x

functions = [f1, f2, f3, f4, f5]

def rectangle_method(f, a, b, n, mode="left"):
    h = (b - a) / n
    total = 0.0

    if mode == "left":
        for i in range(n):
            x = a + i * h
            total += f(x)

    elif mode == "right":
        for i in range(1, n+1):
            x = a + i * h
            total += f(x)

    elif mode == "middle":
        for i in range(n):
            x = a + (i + 0.5) * h
            total += f(x)
    else:
        raise ValueError("Неизвестный режим прямоугольников.")

    return total * h

def trapezoid_method(f, a, b, n):
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    return total * h

def simpson_method(f, a, b, n):
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    total = f(a) + f(b)
    odd_sum = 0.0
    even_sum = 0.0
    for k in range(1, n):
        x = a + k * h
        if k % 2 == 0:
            even_sum += f(x)
        else:
            odd_sum += f(x)
    total += 4 * odd_sum + 2 * even_sum
    return total * h / 3.0

def integrate_with_precision(f, a, b, eps, method_name="rectangle", mode="left", n_init=4):
    if method_name == "rectangle":
        method_func = lambda f,a,b,n: rectangle_method(f,a,b,n,mode=mode)
        p = 2 if mode == "middle" else 1
        if p == 2:
            runge_factor = 1.0/3.0
        else:
            runge_factor = 1.0  # Если p=1

    elif method_name == "trapezoid":
        method_func = trapezoid_method
        runge_factor = 1.0/3.0

    elif method_name == "simpson":
        method_func = simpson_method
        runge_factor = 1.0/15.0

    else:
        raise ValueError("Неизвестный метод: " + method_name)

    n = n_init
    I_old = method_func(f, a, b, n)
    while True:
        n *= 2
        I_new = method_func(f, a, b, n)
        # оценка погрешности по правилу Рунге:
        error_est = abs(I_new - I_old) * runge_factor

        if error_est < eps:
            return I_new, n
        I_old = I_new

def main():
    print("Выберите функцию для интегрирования:")
    for idx, func in enumerate(functions, start=1):
        print(f"{idx}) {func.__doc__}")

    choice = int(input("Введите номер функции (1..5): "))
    if choice < 1 or choice > len(functions):
        print("Некорректный выбор функции. Завершение.")
        return
    f = functions[choice - 1]

    a = float(input("Введите левую границу интегрирования a = "))
    b = float(input("Введите правую границу интегрирования b = "))
    if a == b:
        print("Границы интегрирования совпадают, интеграл будет 0.")
        return

    eps = float(input("Введите требуемую точность (например, 1e-6): "))

    n_init = int(input("Введите начальное число разбиений (n >= 2), по умолчанию 4: ") or 4)
    if n_init < 2:
        n_init = 4

    print("\nВыберите метод интегрирования:")
    print("1) Прямоугольники (левые)")
    print("2) Прямоугольники (правые)")
    print("3) Прямоугольники (средние)")
    print("4) Трапеции")
    print("5) Симпсон")

    method_choice = int(input("Номер метода: "))

    if method_choice == 1:
        method_name = "rectangle"
        mode = "left"
    elif method_choice == 2:
        method_name = "rectangle"
        mode = "right"
    elif method_choice == 3:
        method_name = "rectangle"
        mode = "middle"
    elif method_choice == 4:
        method_name = "trapezoid"
        mode = ""
    elif method_choice == 5:
        method_name = "simpson"
        mode = ""
    else:
        print("Некорректный выбор метода. Завершение.", file=sys.stderr)
        return

    I, N = integrate_with_precision(f, min(a, b), max(a, b), eps, method_name, mode, n_init)

    # вывод рез-тов
    print(f"\nРезультаты:")
    print(f"Метод: {method_name}, вариант = {mode if mode else '—'}")
    print(f"Число разбиений, достигнутое для заданной точности: {N}")
    print(f"Приблизительное значение интеграла: {I:.8f}")
    print(f"Точность (заданная): {eps}")

if __name__ == "__main__":
    main()
