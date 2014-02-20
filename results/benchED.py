import numpy as np
import pylab as pl
import matplotlib
from subprocess import check_output

datasets = {"metal": {"M": 2**16, "N": 11348848, "K": 10},}
            #"ecg":   {"M": 2**16, "N": 20040000, "K": 10}}

Ms = [128, 256, 512, 1024, 2048, 4096, 8192]


for dataset in datasets:
    N = datasets[dataset]["N"]
    K = datasets[dataset]["K"]
    Sgpu = np.zeros((len(Ms), K))
    Smkl = np.zeros((len(Ms), K))
    Sucr = np.zeros((len(Ms), K))
    
    for im, M in enumerate(Ms):
        for k in range(K):
            args  = ["data/%s/double_query%s.bin" % (dataset, k)]
            args.append("data/%s/double_subject.bin" % (dataset,))
            args.extend([str(M), str(N)])
            output = check_output(["./GPU-ED"]+args)
            
            for line in output.split("\n"):
                if "best match:" in line:
                    Sgpu[im, k] = float(line.split()[-1])/1000.0
            
            print output
            
            args  = ["data/%s/double_query%s.bin" % (dataset, k)]
            args.append("data/%s/double_subject.bin" % (dataset,))
            args.extend([str(M), str(N)])
            output = check_output(["./MKL-ED"]+args)
            
            for line in output.split("\n"):
                if "best match:" in line:
                    Smkl[im, k] = float(line.split()[-1])/1000.0
            
            print output
            
            args  = ["data/%s/double_subject.bin" % (dataset,)]
            args.append("data/%s/double_query%s.bin" % (dataset, k))
            args.extend([str(M), str(N)])
            output = check_output(["./UCR Suite/UCR_ED_BIN"]+args)
            
            for line in output.split("\n"):
                if "Total Execution Time :" in line:
                    Sucr[im, k] = float(line.split()[-2])
            
            print output

            

    fig = pl.figure()
    ax = fig.add_subplot(111)
    pl.rcParams.update({'font.size': 28})

    gpuavg, gpustd = np.mean(Sgpu, axis=1), np.std(Sgpu, axis=1)
    ax.errorbar(Ms, gpuavg, fmt="o", yerr=gpustd, c="blue")
    p1, = ax.plot(Ms, gpuavg, color="blue", linestyle="--")
    mklavg, mklstd = np.mean(Smkl, axis=1), np.std(Smkl, axis=1)
    ax.errorbar(Ms, mklavg, fmt="v", yerr=mklstd, color="green")
    p2, = ax.plot(Ms, mklavg, c="green", linestyle="-.")
    ucravg, ucrstd = np.mean(Sucr, axis=1), np.std(Sucr, axis=1)
    ax.errorbar(Ms, ucravg, fmt="s", yerr=ucrstd, color="red")
    p3, = ax.plot(Ms, ucravg, c="red")
    

    pl.legend([p1, p2, p3], ["GPU-ED", "MKL-ED", "UCR-ED"])
    
    ax.set_title("z-normalized sED (%s dataset)" % dataset)
    ax.set_xlabel('query length')
    ax.set_ylabel('execution time in seconds')

    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(Ms)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter(2))
    
    print dataset, "summary"
    print "gpuavg:", list(gpuavg)
    print "gpustd:", list(gpustd)
    print "mklavg:", list(mklavg)
    print "mklstd:", list(mklstd)
    print "ucravg:", list(ucravg)
    print "ucrstd:", list(ucrstd)
    
    
    pl.tight_layout()
    pl.show()
