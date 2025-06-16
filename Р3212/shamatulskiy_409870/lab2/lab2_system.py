import math
import sys
import numpy as np
import matplotlib.pyplot as plt

def sys1(x):
    u,v = x
    return np.array([u**2 + v**2 - 4, u*v - 1])

def phi1(x):
    u,v = x
    return np.array([math.sqrt(max(0,4 - v**2)), 1/u])

def sys2(x):
    u,v = x
    return np.array([math.sin(u) + v - 1, u + math.cos(v) - 1])

def phi2(x):
    u,v = x
    return np.array([math.asin(1 - v), math.acos(1 - u)])

systems = {
    '1': (sys1, phi1),
    '2': (sys2, phi2),
}

def jacobian(phi, x, h=1e-6):
    n = len(x)
    J = np.zeros((n,n))
    for i in range(n):
        dx = np.zeros(n); dx[i] = h
        J[:,i] = (phi(x+dx) - phi(x-dx))/(2*h)
    return J

def simple_iter(sysf, phi, x0, eps, maxit=1000):
    J = jacobian(phi, x0)
    if max(abs(np.linalg.eigvals(J))) >= 1:
        print('Условие сходимости не выполнено'); sys.exit()
    x_prev = np.array(x0)
    errors = []
    for k in range(1, maxit+1):
        x_new = phi(x_prev)
        err = np.abs(x_new - x_prev)
        errors.append(err)
        if np.all(err < eps):
            return x_new, k, errors
        x_prev = x_new
    print('Не сошлось за max итераций'); sys.exit()

def get_input():
    print('Системы:')
    print('1: x^2 + y^2 = 4; x*y = 1')
    print('2: sin(x) + y = 1; x + cos(y) = 1')
    sysn = input('Выберите номер системы: ')
    x0 = float(input('x0= ')), float(input('y0= '))
    eps = float(input('ε= '))
    return sysn, x0, eps

def output(x, it, err):
    if input('Вывод в файл? (y/n) ').lower()=='y':
        fn = input('Имя файла: ')
        with open(fn, 'w') as g:
            g.write(f'x = {x}\n')
            g.write(f'Итераций = {it}\n')
            g.write(f'Ошибка последней итерации = {err}\n')
    else:
        print('x =', x)
        print('Итераций =', it)
        print('Ошибка последней итерации =', err)

def plot(sysf, a=-2, b=2):
    xs = np.linspace(a, b, 400)
    ys = np.linspace(a, b, 400)
    X, Y = np.meshgrid(xs, ys)
    Z1 = sysf([X, Y])[0]
    Z2 = sysf([X, Y])[1]
    plt.contour(X, Y, Z1, levels=[0])
    plt.contour(X, Y, Z2, levels=[0])
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.show()

def main():
    sysn, x0, eps = get_input()
    sysf, phi = systems.get(sysn, (None,None))
    if not sysf:
        print('Неизвестная система'); sys.exit()
    x, it, errors = simple_iter(sysf, phi, x0, eps)
    output(x, it, list(errors[-1]))
    plot(sysf)

if __name__=='__main__':
    main()
