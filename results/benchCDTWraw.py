import numpy as np
import pylab as pl
import matplotlib
from subprocess import check_output

datasets = {#"metal": {"M": 2**16, "N": 100000, "K": 10},}
            "ecg":   {"M": 2**16, "N": 10000, "K": 10}}

Ms = [128, 256, 512, 1024, 2048, 4096]


for dataset in datasets:
    N = datasets[dataset]["N"]
    K = datasets[dataset]["K"]
    SBLOCK = np.zeros((len(Ms), K))
    STHRD  = np.zeros((len(Ms), K))
    SOMP   = np.zeros((len(Ms), K))
    SCPU   = np.zeros((len(Ms), K))
    
    for im, M in enumerate(Ms):
        for k in range(K):
            args  = ["data/%s/single_query%s.bin" % (dataset, k)]
            args.append("data/%s/single_subject.bin" % (dataset,))
            args.extend([str(M), str(N), "10"])
            output = check_output(["./test_dtw"]+args)
            
            for line in output.split("\n"):
                if "GPU BLOCK:" in line:
                    SBLOCK[im, k] = float(line.split()[-1])/1000.0
                if "GPU THRD:" in line:
                    STHRD[im, k] = float(line.split()[-1])/1000.0
                if "CPU OMP:" in line:
                    SOMP[im, k] = float(line.split()[-1])/1000.0
                if "CPU SEQ:" in line:
                    SCPU[im, k] = float(line.split()[-1])/1000.0
            print output

    fig = pl.figure()
    ax = fig.add_subplot(111)
    pl.rcParams.update({'font.size': 28})

    blockavg, blockstd = np.mean(SBLOCK, axis=1), np.std(SBLOCK, axis=1)
    ax.errorbar(Ms, blockavg, fmt="o", yerr=blockstd, c="blue")
    p1, = ax.plot(Ms, blockavg, color="blue", linestyle="-")
    
    threadavg, threadstd = np.mean(STHRD, axis=1), np.std(STHRD, axis=1)
    ax.errorbar(Ms, threadavg, fmt="s", yerr=threadstd, c="green")
    p2, = ax.plot(Ms, threadavg, color="green", linestyle="--")

    ompavg, ompstd = np.mean(SOMP, axis=1), np.std(SOMP, axis=1)
    ax.errorbar(Ms, ompavg, fmt="v", yerr=ompstd, c="red")
    p3, = ax.plot(Ms, ompavg, color="red", linestyle="-.")

    cpuavg, cpustd = np.mean(SCPU, axis=1), np.std(SCPU, axis=1)
    ax.errorbar(Ms, cpuavg, fmt="p", yerr=cpustd, c="black")
    p4, = ax.plot(Ms, cpuavg, color="black", linestyle=":")

    pl.legend([p1, p2, p3, p4], ["GPU-block-10", "GPU-thread-10", 
                                 "CPU-openmp-10", "CPU-single-10"])
    
    ax.set_title("raw subsequence CDTW (%s dataset)" % dataset)
    ax.set_xlabel('query length')
    ax.set_ylabel('execution time in seconds')

    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(Ms)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter(2))
    
    print dataset, "summary"
    print "blockavg:", list(blockavg)
    print "blockstd:", list(blockstd)
    print "threadavg:", list(threadavg)
    print "threadstd:", list(threadstd)
    print "ompavg:", list(ompavg)
    print "ompstd:", list(ompstd)
    print "cpuavg:", list(cpuavg)
    print "cpustd:", list(cpustd)
    
    
    pl.tight_layout()
    pl.show()
