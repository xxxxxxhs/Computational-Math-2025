import sys, math, numpy as np
import matplotlib.pyplot as plt

def finite_diffs(y):
    table=[y.copy()]
    while len(table[-1])>1:
        table.append([table[-1][i+1]-table[-1][i] for i in range(len(table[-1])-1)])
    return table

def lagrange(xv,yv,x):
    s=0.0
    for i in range(len(xv)):
        p=1.0
        for j in range(len(xv)):
            if i!=j:
                p*=(x-xv[j])/(xv[i]-xv[j])
        s+=yv[i]*p
    return s

def newton_divided(xv,yv,x):
    div=yv.copy()
    n=len(xv)
    for k in range(1,n):
        for i in range(n-1,k-1,-1):
            div[i]=(div[i]-div[i-1])/(xv[i]-xv[i-k])
    s=div[-1]
    for k in range(n-2,-1,-1):
        s=s*(x-xv[k])+div[k]
    return s

def newton_fd(xv,yv,x):
    h=xv[1]-xv[0]
    table=finite_diffs(yv)
    n=len(xv)
    if abs(x-xv[0])<abs(x-xv[-1]):
        t=(x-xv[0])/h
        s,fact=table[0][0],1.0
        for k in range(1,n):
            fact*=(t-(k-1))/k
            s+=fact*table[k][0]
    else:
        t=(x-xv[-1])/h
        s,fact=table[0][-1],1.0
        for k in range(1,n):
            fact*=(t+(k-1))/k
            s+=fact*table[k][-1]
    return s

def read_points():
    m=int(input("Количество точек: "))
    pts=[tuple(map(float,input("x y: ").split())) for _ in range(m)]
    pts.sort()
    xv,yv=zip(*pts)
    return list(xv),list(yv)

def func_choice():
    print("Доступные функции:\ny = sin(x) -- sin\ny = cos(x) -- cos\ny = exp(x) -- exp")
    fx=input("Выберите функцию: ").strip()
    a,b=map(float,input("Интервал a b: ").split())
    n=int(input("Число точек: "))
    xv=[a+i*(b-a)/(n-1) for i in range(n)]
    if fx=="sin":
        yv=[math.sin(v) for v in xv]
    elif fx=="cos":
        yv=[math.cos(v) for v in xv]
    else:
        yv=[math.exp(v) for v in xv]
    return xv,yv

def load_file(path):
    with open(path,"r",encoding="utf-8") as f:
        lines=[l.strip() for l in f if l.strip()]
    try:
        count=int(lines[0])
        data=lines[1:]
    except ValueError:
        data=lines
    xv,yv=[],[]
    for l in data:
        parts=l.split()
        if len(parts)>=2:
            a,b=map(float,parts[:2])
            xv.append(a)
            yv.append(b)
    return xv,yv

def plot_method(title,xv,yv,ys_func):
    xs=np.linspace(min(xv),max(xv),400)
    plt.figure()
    plt.plot(xs,[ys_func(xx) for xx in xs])
    plt.scatter(xv,yv,color="black",zorder=5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)


def main():
    if input("Ввод из файла? (y/n): ").lower().startswith("y"):
        xv,yv=load_file(input("Файл: "))
    else:
        if input("Сгенерировать по функции? (y/n): ").lower().startswith("y"):
            xv,yv=func_choice()
        else:
            xv,yv=read_points()
    steps=[round(xv[i+1]-xv[i],10) for i in range(len(xv)-1)]
    if len(set(steps))!=1:
        print("Шаг по x должен быть равномерным")
        sys.exit()
    table=finite_diffs(yv)
    print("Таблица конечных разностей:")
    for i in range(len(xv)):
        row=[f"{xv[i]:g}",f"{yv[i]:g}"]
        for k in range(1,len(table)):
            if i<len(table[k]):
                row.append(f"{table[k][i]:g}")
        print("\t".join(row))
    xq=float(input("Введите X для интерполяции: "))
    print("Лагранж:",lagrange(xv,yv,xq))
    print("Ньютон разделённые:",newton_divided(xv,yv,xq))
    print("Ньютон конечные:",newton_fd(xv,yv,xq))

    plot_method("Lagrange interpolation",xv,yv,lambda x: lagrange(xv,yv,x))
    plot_method("Newton (divided) interpolation",xv,yv,lambda x: newton_divided(xv,yv,x))
    plot_method("Newton (finite diffs) interpolation",xv,yv,lambda x: newton_fd(xv,yv,x))
    plt.show()

if __name__=="__main__":
    main()