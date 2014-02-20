import numpy as np
import pylab as pl
import matplotlib
from subprocess import check_output

datasets = {#"metal": {"M": 2**16, "N": 100000, "K": 10},}
            "metal":   {"M": 2**16, "N": 100000, "K": 10}}

Ms = [128, 256, 512, 1024, 2048, 4096]


for dataset in datasets:
    N = datasets[dataset]["N"]
    K = datasets[dataset]["K"]
    Sgpu10 = np.zeros((len(Ms), K))
    Sgpu20 = np.zeros((len(Ms), K))
    Sucr10 = np.zeros((len(Ms), K))
    Sucr20 = np.zeros((len(Ms), K))
    
    for im, M in enumerate(Ms):
        for k in range(K):
            args  = ["data/%s/single_query%s.bin" % (dataset, k)]
            args.append("data/%s/single_subject.bin" % (dataset,))
            args.extend([str(M), str(N), "10"])
            output = check_output(["./GPU-DTW"]+args)
            
            for line in output.split("\n"):
                if "best match:" in line:
                    Sgpu10[im, k] = float(line.split()[-1])/1000.0
            
            print output
            
            args  = ["data/%s/single_query%s.bin" % (dataset, k)]
            args.append("data/%s/single_subject.bin" % (dataset,))
            args.extend([str(M), str(N), "20"])
            output = check_output(["./GPU-DTW"]+args)
            
            for line in output.split("\n"):
                if "best match:" in line:
                    Sgpu20[im, k] = float(line.split()[-1])/1000.0
            
            print output
          
            args  = ["data/%s/double_subject.bin" % (dataset,)]
            args.append("data/%s/double_query%s.bin" % (dataset, k))
            args.extend([str(M), "0.1", str(N)])
            output = check_output(["./UCR Suite/UCR_DTW_BIN"]+args)
            
            for line in output.split("\n"):
                if "Time :" in line:
                    Sucr10[im, k] = float(line.split()[-2])
            
            print output
            
            args  = ["data/%s/double_subject.bin" % (dataset,)]
            args.append("data/%s/double_query%s.bin" % (dataset, k))
            args.extend([str(M), "0.2", str(N)])
            output = check_output(["./UCR Suite/UCR_DTW_BIN"]+args)
            
            for line in output.split("\n"):
                if "Time :" in line:
                    Sucr20[im, k] = float(line.split()[-2])
            
            print output


    fig = pl.figure()
    ax = fig.add_subplot(111)
    pl.rcParams.update({'font.size': 28})

    gpu10avg, gpu10std = np.mean(Sgpu10, axis=1), np.std(Sgpu10, axis=1)
    ax.errorbar(Ms, gpu10avg, fmt="o", yerr=gpu10std, c="blue")
    p1, = ax.plot(Ms, gpu10avg, color="blue", linestyle="-")
    
    gpu20avg, gpu20std = np.mean(Sgpu20, axis=1), np.std(Sgpu20, axis=1)
    ax.errorbar(Ms, gpu20avg, fmt="s", yerr=gpu20std, c="green")
    p2, = ax.plot(Ms, gpu20avg, color="green", linestyle="--")

    ucr10avg, ucr10std = np.mean(Sucr10, axis=1), np.std(Sucr10, axis=1)
    ax.errorbar(Ms, ucr10avg, fmt="v", yerr=ucr10std, c="red")
    p3, = ax.plot(Ms, ucr10avg, color="red", linestyle="-.")

    ucr20avg, ucr20std = np.mean(Sucr20, axis=1), np.std(Sucr20, axis=1)
    ax.errorbar(Ms, ucr20avg, fmt="p", yerr=ucr20std, c="black")
    p4, = ax.plot(Ms, ucr20avg, color="black", linestyle=":")

    pl.legend([p1, p2, p3, p4], ["GPU-DTW-10", "GPU-DTW-20", 
                                 "UCR-DTW-10", "UCR-DTW-20"])
    
    ax.set_title("subsequence CDTW (%s dataset)" % dataset)
    ax.set_xlabel('query length')
    ax.set_ylabel('execution time in seconds')

    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks(Ms)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter(2))
    
    print dataset, "summary"
    print "gpu10avg:", list(gpu10avg)
    print "gpu10std:", list(gpu10std)
    print "ucr10avg:", list(ucr10avg)
    print "ucr10std:", list(ucr10std)
    print "gpu20avg:", list(gpu20avg)
    print "gpu20std:", list(gpu20std)
    print "ucr20avg:", list(ucr20avg)
    print "ucr20std:", list(ucr20std)
    
    
    pl.tight_layout()
    pl.show()
