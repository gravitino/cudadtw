import numpy as np
import pylab as pl
import array as ar
import scipy.io

query = scipy.io.loadmat('ECG_query.mat')["ecg_query"]
subject = scipy.io.loadmat('ECG.mat')["ECG"]

query, subject = subject[:100000], subject[100000:]

L, K = 2**16, 10
for k in range(K):
    lower = np.random.randint(len(query)-L)
    shape = query[lower:lower+L]
    
    with open("single_query%i.bin" % k, "wb") as f:
        f.write(ar.array("f", shape))
    with open("double_query%i.bin" % k, "wb") as f:
        f.write(ar.array("d", shape))
    
    pl.plot(shape)
    pl.show()

with open("single_subject.bin", "wb") as f:
    f.write(ar.array("f", subject))

with open("double_subject.bin", "wb") as f:
    f.write(ar.array("d", subject))

print len(query), len(subject)

pl.plot(subject)
pl.show()
