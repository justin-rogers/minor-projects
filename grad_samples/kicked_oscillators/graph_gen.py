# (PKO): https://arxiv.org/pdf/1004.0565.pdf
# (SIC): https://arxiv.org/pdf/0705.3294.pdf
# Replicating some work on periodically kicked oscillators.
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.cm as cm

style.use('fivethirtyeight')
font = {'weight': 'bold', 'size': 10}
plt.rc('font', **font)


def get_model(sig, lam, A, T):
    """return the map, the jacobian, and a "simulate N kicks and return data" convenience map

    Inputs: (all floats)
    sig: amount of shear.
    lam: damping, or rate of contraction to the limit cycle (y=0)
    A: amplitude of kicks
    T: time interval between kicks

    Outputs:
    psi_T: the time T map associated to given parameters.
    Jpsi_T: a function sending [x,y] to the product J*[x,y], where J is the jacobian of the above map.
    trace_orbit: applies psi_T N times to the given initial point, returns the data.
    """
    # precompute a few things we'll need repeatedly
    elt = math.exp(-lam * T)
    sol = sig / lam  # sigma over lambda
    J12 = sol * (1 - elt)

    def psi_T(th0, y0):
        """time T map (flow + kick), returns [th_T,y1_T]. (PKO) eqn (2), pg 5"""

        # math.modf(...)[0] takes mod1
        th = math.modf(th0 + T + sol * (y0 + A * math.sin(2 * math.pi * th0)) *
                       (1 - elt))[0]
        y = elt * (y0 + A * math.sin(2 * math.pi * th0))
        return [th, y]

    def Jpsi_T(A, v):
        """Jacobian of psi_T applied to A=[v1,v2]. (PKO) eqn (3), pg 8"""
        th, y = A[0], A[1]
        v1, v2 = v[0], v[1]

        foo = 2 * math.pi * A * math.cos(2 * math.pi * th)
        J11 = 1 + (sol * foo) * (1 - elt)
        # J12 computed in enclosing function, constant
        J21 = elt * foo
        J22 = elt
        return [v1 * J11 + v2 * J12, v1 * J21 + v2 * J22]

    def trace_orbit(th0, y0, N):
        """Inputs: (theta0, y0, number_of_steps). Return the result of N steps of simulation"""
        TH = [th0]
        Y = [y0]
        th, y = th0, y0
        for _ in range(N + 1):
            th, y = psi_T(th, y)
            Y.append(y)
            TH.append(th)
        return [TH, Y]

    return psi_T, Jpsi_T, trace_orbit


def get_fig1_data(sig, lam, A, T, N=1000):
    """Get phi_T(gamma): image of S1 x 0 under a kick.
    N = number of sample points.


    Output: 2 by N array, each column is a (theta,y) pair.
    Reference image: PKO pg4.
    """
    F = get_model(sig, lam, A, T)[0]
    sample_pts = np.linspace(0, 1, N)
    sample_vals = np.zeros([2, N])
    for i in range(N):
        sample_vals[:, i] = F(sample_pts[i], 0)
    return sample_vals


def fig1_color():
    plt.figure(figsize=(6, 6))
    sigma_vals = [0.05, 0.25, 0.5, 1, 2, 10]
    for i in range(1, 7):
        sigma = sigma_vals[i - 1]
        N = 1000
        data = get_fig1_data(sigma, 0.1, 0.1, 10, N=N)
        colors = cm.rainbow(np.linspace(0, 1, N))
        subplot_code = 320 + i
        ax = plt.subplot(subplot_code)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title('σ = {}.'.format(sigma))
        ax.scatter(data[0], data[1], marker='.', color=colors, linewidth=0)
        ax.plot([0, 1], [0, 0], color='grey', linewidth=0.8)
        ax.scatter([0, 1], [0, 0], marker='o', color='b')

    plt.suptitle('Fixed params: λ=0.1, A=0.1, τ=10')
    plt.savefig("./graphs/fig1_color1000_v2.png")
    #plt.show()
    plt.clf()


def generic_fig_l(N=1000):
    fig = plt.figure(figsize=(6, 6))
    A = [0.1]
    l_vals = [0.01, 0.05, 0.1, 0.2, 0.4, 2.0]
    T = [10]
    s = [1]
    for i in range(6):
        l = [l_vals[i]]
        T = [1 / l[0]]

        params = s + l + A + T
        data = get_fig1_data(*params, N=N)
        colors = cm.rainbow(np.linspace(0, 1, N))
        subplot_code = 320 + i + 1
        ax = plt.subplot(subplot_code)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title('λ={}, τ={}.'.format(l[0], T[0]))
        ax.scatter(data[0], data[1], marker='.', color=colors, linewidth=0)
        ax.plot([0, 1], [0, 0], color='grey', linewidth=0.8)
        ax.scatter([0, 1], [0, 0], marker='o', color='b')
    plt.suptitle('Fixed params: A=0.1, σ=1')
    plt.savefig('./graphs/lambda_test_v2.png')
    #plt.show()
    plt.clf()


def example_gen(N=1000):
    fig = plt.figure(figsize=(6, 6))
    l = [0.1]
    A_vals = [0, 0.001, 1, 10, 100, 1000]
    T = [10]
    s = [0.05]
    for i in range(6):
        A = [A_vals[i]]
        T = [1 / l[0]]

        params = s + l + A + T
        data = get_fig1_data(*params, N=N)
        colors = cm.rainbow(np.linspace(0, 1, N))
        subplot_code = 320 + i + 1
        ax = plt.subplot(subplot_code)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
        plt.title('A={}.'.format(A[0]))
        ax.scatter(data[0], data[1], marker='.', color=colors, linewidth=0)
        ax.plot([0, 1], [0, 0], color='grey', linewidth=0.8)
        ax.scatter([0, 1], [0, 0], marker='o', color='b')
    plt.suptitle('Fixed params: σ=1, τ=10, λ=0.1')
    plt.savefig('./graphs/A_test_v2.png')
    #plt.show()
    plt.clf()


if __name__ == "__main__":
    generic_fig_l()
    fig1_color()
    example_gen()
