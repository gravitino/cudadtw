import numpy as np
import pylab as pl
import matplotlib

Ms = [128, 256, 512, 1024, 2048, 4096]


fig = pl.figure()
ax = fig.add_subplot(111)
pl.rcParams.update({'font.size': 30})

gpu10avg = np.array([0.3338633, 0.6289851, 1.211321, 
                     2.370729, 4.705315, 12.31788])
gpu10std = np.array([0.0051115233, 0.0075934506, 0.0088027224,
                     0.008916461, 0.0141416754, 0.1283775664])

ax.errorbar(Ms, gpu10avg, fmt="o", yerr=gpu10std, c="blue")
p1, = ax.plot(Ms, gpu10avg, color="blue", linestyle="-")

omp10avg = np.array([8.768149, 17.954, 36.82623,
                     75.34849, 152.4197, 305.3048])
omp10std = np.array([0.176690129, 0.1663725471, 0.129380971, 
                    0.2575330203, 0.9130701628, 3.7794122941])

ax.errorbar(Ms, omp10avg, fmt="s", yerr=omp10std, c="green")
p2, = ax.plot(Ms, omp10avg, color="green", linestyle="--")

cpu10avg = np.array([58.87119, 117.7952, 236.9547, 478.7272,
                     970.7637, 1964.442])
cpu10std = np.array([1.1404732765, 0.9450350728, 2.4774986063,
                     3.65465, 4.8321658831, 10.6585738049])

ax.errorbar(Ms, cpu10avg, fmt="v", yerr=cpu10std, c="red")
p3, = ax.plot(Ms, cpu10avg, color="red", linestyle="-.")

pl.legend([p1, p2, p3], ["GPU-10", "CPU-openmp-10", "CPU-single-10"])
    
ax.set_title("subsequence CLDTW (ecg dataset)")
ax.set_xlabel('query length')
ax.set_ylabel('execution time in seconds')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xticks(Ms)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter(2))

pl.tight_layout()
pl.show()
