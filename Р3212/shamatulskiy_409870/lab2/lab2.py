import math
import sys
import matplotlib.pyplot as plt

def f1(x):
    return x**3 - 2*x - 5

def f2(x):
    return math.sin(x) - 0.5*x

def f3(x):
    return math.exp(-x) - x

functions = {
    '1': ('x^3 - 2x - 5', f1),
    '2': ('sin(x) - 0.5x', f2),
    '3': ('e^(-x) - x', f3),
}

def phi3(x):
    return math.exp(-x)

phis = {
    '3': phi3,
}

def bisection(f,a,b,eps):
    fa,fb=f(a),f(b)
    if fa*fb>0: return None,None,None
    n=0
    while (b-a)/2>eps:
        c=(a+b)/2
        if f(a)*f(c)<=0:
            b=c
        else:
            a=c
        n+=1
    return (a+b)/2,f((a+b)/2),n

def secant(f,a,b,eps):
    fa,fb=f(a),f(b)
    if fa==fb: return None,None,None
    n=0
    while abs(b-a)>eps:
        c=b - fb*(b-a)/(fb-fa)
        a,b,fa,fb=b,c,fb,f(c)
        n+=1
    return b,f(b),n

def simple_iter(f,phi,a,b,eps):
    mp = max(abs((phi(a+1e-6)-phi(a))/1e-6),abs((phi(b)-phi(b-1e-6))/1e-6))
    if mp>=1: return 'bad',None,None
    x0=(a+b)/2
    n=0
    while True:
        x1=phi(x0)
        n+=1
        if abs(x1-x0)<eps:
            return x1,f(x1),n
        x0=x1

def get_input():
    if input('Ввод из файла? (y/n) ').lower()=='y':
        fn=input('Имя файла: ')
        with open(fn) as g:
            data=g.read().split()
        it=iter(data)
        eq=next(it)
        method=next(it)
        a,b,eps=map(float,(next(it),next(it),next(it)))
    else:
        print('Уравнения:')
        for k,v in functions.items():
            print(k,v[0])
        eq=input('Выберите номер функции: ')
        print('Методы: 1-бисекция, 2-секущие, 3-простая итерация')
        method=input('Выберите метод: ')
        a=float(input('a= '))
        b=float(input('b= '))
        eps=float(input('ε= '))
    return eq,method,a,b,eps

def output(root,val,iter_count):
    if input('Вывод в файл? (y/n) ').lower()=='y':
        fn=input('Имя файла: ')
        with open(fn,'w') as g:
            g.write(f'Корень: {root}\n')
            g.write(f'f(root): {val}\n')
            g.write(f'Итераций: {iter_count}\n')
    else:
        print('Корень:',root)
        print('f(root)=',val)
        print('Итераций:',iter_count)

def plot(f,a,b):
    xs=[a+i*(b-a)/1000 for i in range(1001)]
    ys=[f(x) for x in xs]
    plt.plot(xs,ys)
    plt.axhline(0, color='black')
    plt.show()

def main():
    eq,method,a,b,eps=get_input()
    desc,f=functions.get(eq,(None,None))
    if f is None:
        print('Неизвестная функция'); sys.exit()
    fa,fb=f(a),f(b)
    if method in ('1','2') and fa*fb>0:
        print('На интервале нет единственного корня'); sys.exit()
    if method=='1':
        root,val,it=bisection(f,a,b,eps)
    elif method=='2':
        root,val,it=secant(f,a,b,eps)
    else:
        phi=phis.get(eq)
        if not phi:
            print('Итерационная форма не задана'); sys.exit()
        root,val,it=simple_iter(f,phi,a,b,eps)
        if root=='bad':
            print('Условие сходимости не выполнено'); sys.exit()
    output(root,val,it)
    plot(f,a,b)

if __name__=='__main__':
    main()
