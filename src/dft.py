import math
import cmath
import random
import time

def dft(x):
    """
    A lame discrete Fourier transform implementation. A little too slow for usage.
    """
    n = len(x)
    two_pi_over_n = (2 * math.pi / n)
    y = []
    for j in range(n//2):  # by symmetry, only need first half
        j_two_pi_over_n = j * two_pi_over_n * 1.0j
        y.append(sum([x[k] * cmath.exp(-j_two_pi_over_n * k) for k in range(n)]))
    return y


def dft_power(x, n_j=None):
    n = len(x)
    if n_j is None:
        n_j = n // 2
    two_pi_over_n = (2 * math.pi / n)
    y = []
    for j in range(n_j):
        j_two_pi_over_n = j * two_pi_over_n
        yj_re = 0.0
        yj_im = 0.0
        for k in range(n):
            z = j_two_pi_over_n * k
            yj_re += x[k] * math.cos(z)
            yj_im += x[k] * math.sin(-z)
        y.append((yj_re, yj_im))
    # return [math.sqrt(re*re + im*im) for re, im in y]
    return y


data = [random.gauss(0.0, 1.0) for _ in range(3600)]

times = []
for n in range(3):
    ts = time.time()
    ft_data = dft_power(data)
    times.append(time.time() - ts)

print(times)
