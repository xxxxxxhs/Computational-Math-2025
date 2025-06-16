import numpy as np

def gauss_seidel(A, b, tol, max_iter=10000):
    n = len(b)
    x = np.zeros(n)
    for k in range(1, max_iter+1):
        x_old = x.copy()
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s)/A[i, i]
        err = np.abs(x - x_old)
        if np.all(err < tol):
            return x, k, err
    return x, k, err

def make_diagonally_dominant(A, b):
    n = A.shape[0]
    for i in range(n):
        for j in range(i, n):
            if abs(A[j, i]) >= np.sum(np.abs(A[j, :])) - abs(A[j, i]):
                if j != i:
                    A[[i, j]], b[[i, j]] = A[[j, i]].copy(), b[[j, i]].copy()
                break
    for i in range(n):
        if abs(A[i, i]) < np.sum(np.abs(A[i, :])) - abs(A[i, i]):
            return False
    return True

def read_matrix():
    mode = input("Чтение матрицы из файла? (y/n): ")
    if mode.lower() == "y":
        fn = input("Имя файла: ")
        data = np.loadtxt(fn)
        A = data[:, :-1]
        b = data[:, -1]
    else:
        n = int(input("n: "))
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            row = list(map(float, input(f"Строка {i+1} (n+1 чисел): ").split()))
            A[i, :] = row[:-1]
            b[i] = row[-1]
    return A, b

def read_tol():
    mode = input("Чтение точности из файла? (y/n): ")
    if mode.lower() == "y":
        fn = input("Имя файла с точностью: ")
        tol = float(open(fn).read().strip())
    else:
        tol = float(input("Точность: "))
    return tol

def main():
    A, b = read_matrix()
    tol = read_tol()
    if not make_diagonally_dominant(A, b):
        print("Невозможно добиться диагонального преобладания")
        return
    x, iters, err = gauss_seidel(A, b, tol)
    norm = np.linalg.norm(A, ord=np.inf)
    print("Норма матрицы (сумма по строкам, бесконечная):", norm)
    print("Решение x:", x)
    print("Число итераций:", iters)
    print("Вектор погрешностей:", err)

if __name__ == "__main__":
    main()
