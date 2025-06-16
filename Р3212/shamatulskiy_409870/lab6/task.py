import math
import matplotlib.pyplot as plt

def eq1_f(x, y):
    return x + y

def eq1_exact(x, x0, y0):
    return (y0 + 1) * math.exp(x - x0) - x - 1 + x0

def eq2_f(x, y):
    return y * x * x

def eq2_exact(x, x0, y0):
    return y0 * math.exp((x**3 - x0**3) / 3)

def eq3_f(x, y):
    return y * x

def eq3_exact(x, x0, y0):
    return y0 * math.exp((x**2 - x0**2) / 2)

def euler(f, x0, y0, xn, h):
    xs = []
    ys = []
    x = x0
    y = y0
    while x <= xn + 1e-12:
        xs.append(x)
        ys.append(y)
        y = y + h * f(x, y)
        x = x + h
    return xs, ys

def rk4(f, x0, y0, xn, h):
    xs = []
    ys = []
    x = x0
    y = y0
    while x <= xn + 1e-12:
        xs.append(x)
        ys.append(y)
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1 / 2)
        k3 = f(x + h/2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        y = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
    return xs, ys

def milne(f, x0, y0, xn, h):
    xs_rk, ys_rk = rk4(f, x0, y0, x0 + 3*h, h)
    xs = xs_rk[:]
    ys = ys_rk[:]
    fs = [f(xs[i], ys[i]) for i in range(4)]
    i = 3
    while xs[i] + h <= xn + 1e-12:
        x_next = xs[i] + h
        y_pred = ys[i-3] + (4*h/3) * (2*fs[i-2] - fs[i-1] + 2*fs[i])
        f_pred = f(x_next, y_pred)
        y_corr = ys[i-1] + (h/3) * (fs[i-1] + 4*fs[i] + f_pred)
        xs.append(x_next)
        ys.append(y_corr)
        fs.append(f(x_next, y_corr))
        i += 1
    return xs, ys

def runge_error(method, f, x0, y0, xn, h, p):
    _, y_h = method(f, x0, y0, xn, h)
    _, y_h2 = method(f, x0, y0, xn, h/2)
    y_end_h = y_h[-1]
    y_end_h2 = y_h2[-1]
    return abs(y_end_h2 - y_end_h) / (2**p - 1)

def main():
    print("Выберите уравнение:")
    print("1) y' = x + y")
    print("2) y' = y * x^2")
    print("3) y' = y * x")
    choice = int(input())
    if choice == 1:
        f = eq1_f
        exact = eq1_exact
    elif choice == 2:
        f = eq2_f
        exact = eq2_exact
    else:
        f = eq3_f
        exact = eq3_exact
    x0 = float(input("x0 = "))
    y0 = float(input("y0 = "))
    xn = float(input("xn = "))
    h = float(input("h = "))
    eps = float(input("epsilon = "))
    xs_e, ys_e = euler(f, x0, y0, xn, h)
    xs_rk, ys_rk = rk4(f, x0, y0, xn, h)
    xs_m, ys_m = milne(f, x0, y0, xn, h)
    print("\n i\t   x\t   Euler\t\t   RK4\t\t   Milne\t\t   Exact")
    N = len(xs_e)
    for i in range(N):
        x_val = xs_e[i]
        y_ex = exact(x_val, x0, y0)
        y_e = ys_e[i] if i < len(ys_e) else float('nan')
        y_r = ys_rk[i] if i < len(ys_rk) else float('nan')
        y_m = ys_m[i] if i < len(ys_m) else float('nan')
        print(f"{i:2d}\t{x_val:8.4f}\t{y_e:10.6f}\t{y_r:10.6f}\t{y_m:10.6f}\t{y_ex:10.6f}")
    err_euler = runge_error(euler, f, x0, y0, xn, h, 1)
    err_rk4 = runge_error(rk4, f, x0, y0, xn, h, 4)
    max_err_m = 0.0
    for i, xm in enumerate(xs_m):
        ye = exact(xm, x0, y0)
        ym = ys_m[i]
        max_err_m = max(max_err_m, abs(ye - ym))
    print(f"\nОценка погрешности Эйлера (правило Рунге): {err_euler:.6e}")
    print(f"Оценка погрешности RK4 (правило Рунге): {err_rk4:.6e}")
    print(f"Максимальная погрешность Милна: {max_err_m:.6e}")
    xs_exact = [x0 + i*h/10 for i in range(int((xn - x0)/(h/10)) + 1)]
    ys_exact = [exact(x, x0, y0) for x in xs_exact]
    plt.plot(xs_exact, ys_exact, label="Exact", color="black")
    plt.plot(xs_e, ys_e, label="Euler", linestyle="--")
    plt.plot(xs_rk, ys_rk, label="RK4", linestyle="-.")
    plt.plot(xs_m, ys_m, label="Milne", linestyle=":")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Numerical vs Exact Solution")
    plt.show()

if __name__ == "__main__":
    main()
