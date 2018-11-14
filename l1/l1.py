import numpy as np
import matplotlib.pyplot as plt
import math as mth


def Ux1(t, a=1):
    return np.exp(-a * t)


def Ux2(t, a=1):
    return -np.exp(-a * t)


def U(x):
    return np.sin(x)


def Analitic(x, t, a=1):
    return np.exp(-a * t) * np.sin(x)


def progonka(a, b, c, d, s):
    P = np.zeros(s)
    Q = np.zeros(s)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    k = s - 1

    for i in range(1, s):
        P[i] = -c[i] / (b[i] + a[i] * P[i - 1])
        Q[i] = (d[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])
    P[k] = 0
    Q[k] = (d[k] - a[k] * Q[k - 1]) / (b[k] + a[k] * P[k - 1])

    x = np.zeros(s)
    x[k] = Q[k]

    for i in range(s - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x

# данные
# param_a = 1
# x0 = 0
# xl = mth.pi
# t = 2
# интервал с 21 по х и 2001 по t
# aprox = 2
# space_step = 0.157
# time_step = 0.001
#


def autofill(x0, space_step, m, n):
    Uarray = np.zeros([n, m])

    tmp_x = x0
    for j in range(m):
        Uarray[0][j] = U(tmp_x)
        tmp_x += space_step

    return Uarray


def explicit(param_t, param_a, space_step, time_step, m, n, aprox, ans_time):
    x0 = 0
    xl = mth.pi

    X = np.arange(x0, xl + space_step, space_step)

    Uarray = autofill(x0, space_step, m, n)

    sigma = param_a**2 * time_step / space_step**2

    tmp_time = 0
    for k in range(1, n):
        for j in range(1, m - 1):
            Uarray[k][j] = sigma * \
                (Uarray[k - 1][j + 1] + Uarray[k - 1][j - 1]) + \
                (1 - 2 * sigma) * Uarray[k - 1][j]
        if aprox == 1:
            Uarray[k][0] = Uarray[k][1] - space_step * Ux1(tmp_time, param_a)
            Uarray[k][m - 1] = Uarray[k][m - 2] + \
                space_step * Ux2(tmp_time, param_a)
        elif aprox == 2:
            Uarray[k][0] = (Ux1(tmp_time, param_a) +
                            Uarray[k][2] / (2 * space_step) -
                            2 * Uarray[k][1] / space_step) * \
                ((2 * space_step) / -3)
            Uarray[k][m - 1] = (Ux2(tmp_time, param_a) -
                                Uarray[k][m - 3] / (2 * space_step) +
                                2 * Uarray[k][m - 2] / space_step) * \
                ((2 * space_step) / 3)
        elif aprox == 3:
            Uarray[k][0] = (Uarray[k][1] -
                            space_step * Ux1(tmp_time, param_a) +
                            (space_step**2 / 2 / time_step) *
                            Uarray[k - 1][0]) / \
                (1 + space_step**2 / 2 / time_step)
            Uarray[k][m - 1] = (Uarray[k][m - 2] +
                                space_step * Ux2(tmp_time, param_a) +
                                (space_step**2 / 2 / time_step) *
                                Uarray[k - 1][m - 1]) / \
                (1 + space_step**2 / 2 / time_step)

        tmp_time += time_step

    in_array = int(ans_time / time_step)
    ans_t = in_array * time_step

    plt.subplot(211)
    plt.gcf().canvas.set_window_title('Лабораторная номер 1')
    plt.plot(X, Analitic(X, ans_t), color='red')
    plt.plot(X, Uarray[in_array])
    plt.grid()

    plt.subplot(212)
    T = np.arange(0, param_t + time_step, time_step)
    max_analitic_in_it_time = []
    for k in T:
        in_it_time = Analitic(X, k)
        in_arr = int(k / time_step)
        max_analitic_in_it_time.append(max(abs(in_it_time - Uarray[in_arr])))
    plt.plot(T, max_analitic_in_it_time, color='red', label='error')
    plt.ylabel('Error')
    plt.grid()

    plt.show()


def implicit(param_t, param_a, space_step, time_step, m, n, aprox, ans_time):
    x0 = 0
    xl = mth.pi

    X = np.arange(x0, xl + space_step, space_step)

    Uarray = autofill(x0, space_step, m, n)

    sigma = param_a**2 * time_step / space_step**2

    for k in range(n - 1):
        a = np.zeros(m)
        b = np.zeros(m)
        c = np.zeros(m)
        d = np.zeros(m)

        # alpha = 1
        # beta = 0
        # gamma = 1
        # tetta = 0
        for j in range(1, m - 1):
            a[j] = sigma
            b[j] = -(1 + 2 * sigma)
            c[j] = sigma
            d[j] = -Uarray[k][j]
        if aprox == 1:
            b[0] = -1 / space_step
            c[0] = 1 / space_step
            d[0] = Ux1((k + 1) * time_step)

            a[m - 1] = -1 / space_step
            b[m - 1] = 1 / space_step
            d[m - 1] = Ux2((k + 1) * time_step)
        elif aprox == 2:
            k0 = 1 / (2 * space_step) / c[1]
            b[0] = (-3 / (space_step * 2)) + a[1] * k0
            c[0] = 2 / space_step + b[1] * k0
            d[0] = Ux1((k + 1) * time_step) + d[1] * k0

            k1 = -(1 / (space_step * 2)) / a[m - 2]
            a[m - 1] = (-2 / space_step) + b[m - 2] * k1
            b[m - 1] = (3 / (space_step * 2)) + c[m - 2] * k1
            d[m - 1] = Ux2((k + 1) * time_step) + d[m - 2] * k1
        elif aprox == 3:
            b[0] = 2 * param_a**2 / space_step + space_step / time_step
            c[0] = - 2 * param_a**2 / space_step
            d[0] = (space_step / time_step) * Uarray[k - 1][0] - \
                Ux1((k + 1) * time_step) * 2 * param_a**2

            a[m - 1] = -2 * param_a**2 / space_step
            b[m - 1] = 2 * param_a**2 / space_step + space_step / time_step
            d[m - 1] = (space_step / time_step) * Uarray[k - 1][m - 1] + \
                Ux2((k + 1) * time_step) * 2 * param_a**2

        Y = progonka(a, b, c, d, m)
        Uarray[k + 1] = Y

    in_array = int(ans_time / time_step)
    ans_t = in_array * time_step

    plt.subplot(211)
    plt.gcf().canvas.set_window_title('Лабораторная номер 1')
    plt.plot(X, Analitic(X, ans_t), color='red')
    plt.plot(X, Uarray[in_array])
    plt.grid()

    plt.subplot(212)
    T = np.arange(0, param_t + time_step, time_step)
    max_analitic_in_it_time = []
    for k in T:
        in_it_time = Analitic(X, k)
        in_arr = int(k / time_step)
        max_analitic_in_it_time.append(max(abs(in_it_time - Uarray[in_arr])))
    plt.plot(T, max_analitic_in_it_time, color='red', label='error')
    plt.ylabel('Error')
    plt.grid()

    plt.show()


def KN(param_t, param_a, space_step, time_step, m, n, aprox, ans_time):
    x0 = 0
    xl = mth.pi

    X = np.arange(x0, xl + space_step, space_step)

    Uarray = autofill(x0, space_step, m, n)

    sigma = param_a**2 * time_step / space_step**2

    for k in range(n - 1):
        a = np.zeros(m)
        b = np.zeros(m)
        c = np.zeros(m)
        d = np.zeros(m)

        # alpha = 1
        # beta = 0
        # gamma = 1
        # tetta = 0

        for j in range(1, m - 1):
            a[j] = -sigma / 2
            b[j] = (1 + sigma)
            c[j] = -sigma / 2
            d[j] = (sigma / 2 * Uarray[k][j + 1] + (1 - sigma) *
                    Uarray[k][j] + (sigma / 2) * Uarray[k][j - 1])
        if aprox == 1:
            b[0] = -1 / space_step
            c[0] = 1 / space_step
            d[0] = Ux1((k + 1) * time_step)
            a[m - 1] = -1 / space_step
            b[m - 1] = 1 / space_step
            d[m - 1] = Ux2((k + 1) * time_step)
        if aprox == 2:
            k0 = 1 / (2 * space_step) / c[1]
            b[0] = (-3 / (space_step * 2)) + a[1] * k0
            c[0] = 2 / space_step + b[1] * k0
            d[0] = Ux1((k + 1) * time_step) + d[1] * k0

            k1 = -(1 / (space_step * 2)) / a[m - 2]
            a[m - 1] = (-2 / space_step) + b[m - 2] * k1
            b[m - 1] = (3 / (space_step * 2)) + c[m - 2] * k1
            d[m - 1] = Ux2((k + 1) * time_step) + d[m - 2] * k1
        elif aprox == 3:
            b[0] = 2 * param_a**2 / space_step + space_step / time_step
            c[0] = - 2 * param_a**2 / space_step
            d[0] = (space_step / time_step) * Uarray[k - 1][0] - \
                Ux1((k + 1) * time_step) * 2 * param_a**2

            a[m - 1] = -2 * param_a**2 / space_step
            b[m - 1] = 2 * param_a**2 / space_step + space_step / time_step
            d[m - 1] = (space_step / time_step) * Uarray[k - 1][m - 1] + \
                Ux2((k + 1) * time_step) * 2 * param_a**2

        Y = progonka(a, b, c, d, m)
        Uarray[k + 1] = Y

    in_array = int(ans_time / time_step)
    ans_t = in_array * time_step

    plt.subplot(211)
    plt.gcf().canvas.set_window_title('Лабораторная номер 1')
    plt.plot(X, Analitic(X, ans_t), color='red')
    plt.plot(X, Uarray[in_array])
    plt.grid()

    plt.subplot(212)
    T = np.arange(0, param_t + time_step, time_step)
    max_analitic_in_it_time = []
    for k in T:
        in_it_time = Analitic(X, k)
        in_arr = int(k / time_step)
        max_analitic_in_it_time.append(max(abs(in_it_time - Uarray[in_arr])))
    plt.plot(T, max_analitic_in_it_time, color='red', label='error')
    plt.ylabel('Error')
    plt.grid()

    plt.show()
