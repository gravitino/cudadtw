import numpy as np
import array as ar
import mlpy
import sys

def load_data(filename_query, filename_subject):

    with open(filename_query, "rb") as f:
        query = np.array(ar.array("d", f.read()))
    
    with open(filename_subject, "rb") as f:
        subject = np.array(ar.array("d", f.read()))

    return query, subject
    
def sDTW(query, subject):

    M, N = len(query), len(subject)
    
    for i in range(N-M+1):
        mlpy.dtw_std(query, subject[i:i+M], dist_only=True)


query, subject = load_data(sys.argv[1], sys.argv[2])
query, subject = query[:int(sys.argv[3])], subject[:int(sys.argv[4])]

#print len(query), len(subject)

import time
begin = time.clock()

sDTW(query, subject)

end = time.clock()

print end-begin

