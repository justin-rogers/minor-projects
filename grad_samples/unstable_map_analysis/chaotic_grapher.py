import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
style.use('fivethirtyeight')
# Colorblind-friendly colors
cbf = [[0, 158 / 255, 115 / 255], [86 / 255, 180 / 255, 233 / 255],
       [213 / 255, 94 / 255, 0], [0, 114 / 255, 178 / 255],
       [204 / 255, 121 / 255, 167 / 255], [230 / 255, 159 / 255, 0]]


def f(x):
    """Hand-constructed function to meet given chaos requirement"""
    return 1 - 1.9935450866059057 * x**2


def fn(x, n):
    for _ in range(n):
        x = f(x)
    return x


def data_snatcher(n=1):
    """n=number of iterates, returns a dict (n,fn(x,n)) for x near 0"""
    foo = {}
    for x in range(-250, 250):
        foo[x / 5000] = fn(x / 5000, n)
    return foo


def data_snatcher_hd(n=1):
    """n=number of iterates, returns a dict (n,fn(x,n)) for x near 0"""
    foo = {}
    for x in range(-2500, 2500):
        foo[x / 50000] = fn(x / 50000, n)
    return foo


print(data_snatcher(5))


def data_plotter(n=1):
    """n=# iterates, plots output dict from data_snatcher"""
    data = sorted(data_snatcher(n).items())  #sorted by key, list of tuples
    x, y = zip(*data)
    plt.figure(figsize=(12, 12), edgecolor='k')
    plt.scatter(x, y, c=[0, 114 / 255, 178 / 255], s=5)
    plt.title('Iterates of f: n={}'.format(n))
    plt.xlabel('x')
    plt.ylabel('f^n(x)')
    if n < 10:
        n = '0' + str(n)
    plt.yticks(np.arange(-1, 1, .25))
    #plt.savefig(r".\243frames\{}".format(n))
    plt.show()


def axis_getter(n=50):
    """n=# iterates, plots output dict from data_snatcher"""
    data = sorted(data_snatcher(n).items())  #sorted by key, list of tuples
    x, y = zip(*data)
    plt.figure(figsize=(12, 12), edgecolor='k')
    plt.scatter(x, y, c=[0, 114 / 255, 178 / 255], s=5)
    plt.title('Iterates of f: n={}'.format(n))
    plt.xlabel('x')
    plt.ylabel('f^n(x)')

def data_plotter_hd(n=1):
    """n=# iterates, plots output dict from data_snatcher"""
    data = sorted(data_snatcher_hd(n).items())  #sorted by key, list of tuples
    x, y = zip(*data)
    plt.figure(figsize=(12, 12), edgecolor='k')
    plt.scatter(x, y, c=[0, 114 / 255, 178 / 255], s=5)
    plt.title('Iterates of f: n={}. (10x sampling)'.format(n))
    plt.xlabel('x')
    plt.ylabel('f^n(x)')
    if n < 10:
        n = '0' + str(n)
    plt.yticks(np.arange(-1, 1, .25))
    #plt.savefig(r".\243frames\{}".format(n))
    plt.show()


data_plotter(15)
data_plotter_hd(15)
