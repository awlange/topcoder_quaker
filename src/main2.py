"""
TopCoder Earthquake Challenge
awgl
"""

import time
import math
import cmath
import random

random.seed(12345)


class MetaData:
    def __init__(self, sampleRate, S, sitesData):
        self.sampleRate = sampleRate
        self.S = S
        self.sitesData = sitesData
        self.gtf_Site = None

global metaData
global doTraining


def valOr(val, x):
    return val if val > 0.0 else x


def compute_product_integrals(x, y, z=None):
    integral = 0.0
    integral_sq = 0.0
    if z is None:
        for valx, valy in zip(x, y):
            p = valx * valy
            integral += p
            integral_sq += p*p
    else:
        for valx, valy, valz in zip(x, y, z):
            p = valx * valy * valz
            integral += p
            integral_sq += p*p
    integral /= len(x)
    integral_sq /= len(y)
    return integral, integral_sq**0.5


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
    """
    Also lame DFT, but faster b/c we don't use the complex number stuff. Also, just return the power spectrum here.
    """
    n = len(x)
    if n_j is None:
        n_j = n // 2
    two_pi_over_n = (2 * math.pi / n)
    y = []
    for j in range(n_j+1):
        j_two_pi_over_n = j * two_pi_over_n
        yj_re = 0.0
        yj_im = 0.0
        for k in range(n):
            z = j_two_pi_over_n * k
            yj_re += x[k] * math.cos(z)
            yj_im += x[k] * math.sin(-z)
        y.append((yj_re, yj_im))
    return [math.sqrt(re*re + im*im) for re, im in y]


def forecast(hour, data, K, globalQuakes, writeToDisk=False):
    """
    The method to predict probability of earthquake happening next hour
    """
    global metaData

    # Compute average data across hour, ignoring -1 missing values
    H = metaData.sampleRate
    S = metaData.S

    H3600 = H*3600

    # Scale to be on range 0-1
    max_val = float(2**24)
    data = [float(d) / max_val for d in data]
    K /= 10.0

    smoothed_vals = [0.0 for _ in range(H3600)]

    # Data transforms
    site_channel_stats = []
    site_second_vals = []
    site_combination_stats = []
    site_sorted_second_vals = []
    for j in range(S):
        channel_stats = []
        second_vals = []
        for c in range(3):

            # Compute average
            n = 0
            avg = 0.0
            offset = j*(H3600*3) + c*H3600
            for i in range(H3600):
                val = data[offset + i]
                if val >= 0:  # skip missing values
                    n += 1
                    avg += val
            if n > 0:
                avg /= float(n)

            # Mean fill missing values
            mean_filled_vals = [valOr(data[offset + i], avg) for i in range(H3600)]
            mean = sum(mean_filled_vals) / float(H3600)

            # Compute seconds means, mean center
            i = 0
            second_mean = 0.0
            second_means = []
            for val in mean_filled_vals:
                i += 1
                second_mean += val - mean
                if i == H:
                    second_means.append(second_mean / H)
                    second_mean = 0.0
                    i = 0

            # Smoothing
            smooth_stdev = 0.0
            for i in range(3600):
                # Apply simple smoothing
                val = second_means[((i-2) % 3600)] * 0.05 \
                    + second_means[((i-1) % 3600)] * 0.20 \
                    + second_means[i] * 0.5 \
                    + second_means[((i+1) % 3600)] * 0.20 \
                    + second_means[((i+2) % 3600)] * 0.05
                smooth_stdev += val*val
                smoothed_vals[i] = val
            smooth_stdev = math.sqrt(smooth_stdev / float(3600))

            # First derivative of smoothed vals
            second_1_derivs = []
            for i in range(3600):
                fp1 = smoothed_vals[((i+1) % 3600)]
                fm1 = smoothed_vals[((i-1) % 3600)]
                second_1_derivs.append(0.5 * (fp1 - fm1))

            # Fourier transform vals and deriv, get power spectrum
            # ft_second_vals = dft_power(smoothed_vals, 100)
            # ft_second_derivs = dft_power(second_1_derivs)

            # Tuple-ize for sorting
            tupled_data = [(smoothed_vals[i], second_1_derivs[i]) for i in range(3600)]
            # tupled_data = [(ft_second_vals[i], ft_second_derivs[i]) for i in range(1800)]

            # Collect up the data
            channel_stats.append(mean)
            channel_stats.append(smooth_stdev)
            second_vals.append(tupled_data)
            # second_vals.append(ft_second_vals)

        site_channel_stats.append(channel_stats)
        site_second_vals.append(second_vals)

        # # Sort tuple data to keep value associated with derivatives. Sort by value.
        # tmp = []
        # for m in second_vals:
        #     m.sort()
        #     # Keep em together... not great for viz but convenient for moving around
        #     tmp.append(m)
        # site_sorted_second_vals.append(tmp)

        # Take product of channel second means
        # dim_01 = compute_product_integrals(means[0], means[1])
        # dim_02 = compute_product_integrals(means[0], means[2])
        # dim_12 = compute_product_integrals(means[1], means[2])
        # tri_012 = compute_product_integrals(means[0], means[1], means[2])
        #
        # combination_stats = []
        # combination_stats.append(dim_01[0])
        # combination_stats.append(dim_02[0])
        # combination_stats.append(dim_12[0])
        # combination_stats.append(tri_012[0])
        # combination_stats.append(dim_01[1])
        # combination_stats.append(dim_02[1])
        # combination_stats.append(dim_12[1])
        # combination_stats.append(tri_012[1])
        # site_combination_stats.append(combination_stats)


    # Global quakes per site
    # Turn into a score
    site_global_quakes = []
    for j in range(S):
        # For simplicity to start, only keep the global quake with the minimum distance to this site.
        # If no global quakes, fill with dummy data.
        global_quakes_score = 0.0
        if len(globalQuakes) > 0:
            global_quakes_score = 0.0
            site_loc = (metaData.sitesData[j*2], metaData.sitesData[j*2 + 1])
            for i in range(len(globalQuakes) // 5):
                global_quakes_score += globalQuakes[i*5+3] / distance(site_loc, (globalQuakes[i*5], globalQuakes[i*5+1]))

        site_global_quakes.append(global_quakes_score)

    # Writing to disk
    if writeToDisk:
        with open("bar_" + str(hour) + ".csv", "w") as f:
            # Print data per site
            for s in range(S):
                # is this the site where an earthquake happens next hour?
                happens = 1 if metaData.gtf_Site is not None and s == metaData.gtf_Site else 0
                if happens:
                    # Compute difference of this hour and when it happens
                    happens = metaData.gtf_Hour - hour
                else:
                    continue  # skip that shit! we need more positive data
                f.write(str(happens) + ",")
                f.write(str(K) + ",")
                f.write(str(site_global_quakes[s]) + ",")
                f.write(",".join([str(scs) for scs in site_channel_stats[s]]) + ",")

                # For printing out the values and the derivatives for viz
                for c in range(3):
                    vals = []
                    d1vals = []
                    for val, d1val in site_second_vals[s][c]:
                        vals.append(val)
                        d1vals.append(d1val)
                    f.write(",".join(["{0:.6g}".format(val) for val in vals]) + ",")
                    f.write(",".join(["{0:.6g}".format(val) for val in d1vals]))

                    # f.write(",".join([str(val) for val in site_second_vals[s][c]]))
                    if c == 2:
                        f.write("\n")
                    else:
                        f.write(",")

                # # Print out sorted values
                # for c in range(3):
                #     vals = []
                #     d1vals = []
                #     d2vals = []
                #     for val, d1val, d2val in site_sorted_second_vals[s][c]:
                #         vals.append(val)
                #         d1vals.append(d1val)
                #         d2vals.append(d2val)
                #     f.write(",".join([str(val) for val in vals]) + ",")
                #     f.write(",".join([str(val) for val in d1vals]) + ",")
                #     f.write(",".join([str(val) for val in d2vals]))
                #     if c == 2:
                #         f.write("\n")
                #     else:
                #         f.write(",")

                # f.write(",".join([str(cs) for cs in site_combination_stats[s]]) + ",")
                # f.write(",".join([str(sm) for sm in site_sorted_means[s][0]]) + ",")
                # f.write(",".join([str(sm) for sm in site_sorted_means[s][1]]) + ",")
                # f.write(",".join([str(sm) for sm in site_sorted_means[s][2]]) + "\n")


    # Nothing right now
    return [0.5 for _ in range(S * 2160)]


def readInt():
    return int(input())


def readDouble():
    return float(input())


def distance(loc1, loc2):
    """
    loc1 and loc2 are tuples of the latitude and longitude
    """
    earthRadius = 6371.01

    deltaLon = abs(loc1[1] - loc2[1])
    if deltaLon > 180:
        deltaLon = 360 - deltaLon

    # Convert to radians
    deg2rad = math.pi / 180.0

    deltaLon *= deg2rad
    loc1 = (loc1[0] * deg2rad, loc1[1] * deg2rad)
    loc2 = (loc2[0] * deg2rad, loc2[1] * deg2rad)

    cl11 = math.cos(loc1[0])
    sl11 = math.sin(loc1[0])
    cl21 = math.cos(loc2[0])
    sl21 = math.sin(loc2[0])

    cdeltaLon = math.cos(deltaLon)
    sdeltaLon = math.sin(deltaLon)

    x = math.sqrt((cl11 * sdeltaLon)**2 + (cl21 * sl11 - sl21 * cl11 * cdeltaLon)**2)
    y = sl21 * sl11 + cl21 * cl11 * cdeltaLon
    dist = earthRadius * math.atan2(x, y)
    return dist



def init(sampleRate, S, sitesData):
    global metaData
    metaData = MetaData(sampleRate, S, sitesData)

    # TODO: Initialize network(s)

    return 0


def answer(gtf_Site, gtf_Hour, gtf_Latitude, gtf_Longitude, gtf_Magnitude, gtf_DistToQuake):
    """
    For training, probably want to store these somewhere
    """
    with open("foo_" + str(gtf_Hour) + ".csv", "w") as f:
        f.write(str(gtf_Site) + ",")  # Note: index is 1 based, whereas it is 0 based elsewhere
        f.write(str(gtf_Latitude) + ",")
        f.write(str(gtf_Longitude) + ",")
        f.write(str(gtf_Magnitude) + ",")
        f.write(str(gtf_DistToQuake) + ",")
        f.write(str(metaData.sitesData) + "\n")


def training():
    """
    Read input for training data
    """
    global metaData
    gtf_Site = readInt()  # 1 based index
    gtf_Hour = readInt()
    gtf_Latitude = readDouble()
    gtf_Longitude = readDouble()
    gtf_Magnitude = readDouble()
    gtf_DistToQuake = readDouble()
    answer(gtf_Site, gtf_Hour, gtf_Latitude, gtf_Longitude, gtf_Magnitude, gtf_DistToQuake)

    metaData.gtf_Site = gtf_Site - 1  # Convert to 0 based index
    metaData.gtf_Hour = gtf_Hour


def readInput():
    with open("foo", "w") as f:
        sampleRate = readInt()
        f.write(str(sampleRate) + "\n")

        S = readInt()
        f.write(str(S) + "\n")

        SLEN = readInt()
        f.write(str(SLEN) + "\n")

        sitesData = [readDouble() for _ in range(SLEN)]
        f.write(str(sitesData) + "\n")

        return sampleRate, S, sitesData

def main():
    """
    The main method, following pseudocode
    """
    global doTraining

    sampleRate, S, sitesData = readInput()
    ret = init(sampleRate, S, sitesData)
    print(ret)

    doTraining = readInt()
    if doTraining == 1:
        training()

    # Repeat until quake happens
    while 1:
        hour = readInt()
        if hour == -1:
            break

        DLEN = readInt()
        data = [readInt() for _ in range(DLEN)]
        K = readDouble()
        QLEN = readInt()
        globalQuakes = [readDouble() for _ in range(QLEN)]

        retM = forecast(hour, data, K, globalQuakes, writeToDisk=True)
        print(len(retM))
        for val in retM:
            print(val)


if __name__ == "__main__":
    main()
    # fit()
