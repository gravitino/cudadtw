import numpy as np
import pylab as pl
import array as ar

Subject = []
for i in range(1000):
    filename = "parent_"+"0"*(4-len(str(i)))+str(i)+".txt"
    with open("./parent/"+filename, "r") as f:
        for line in f:
            value = float(line)
            if not np.isnan(value):
                Subject.append(value)
    print "read", filename

subject = np.array(Subject)
#subject = (subject-np.mean(subject))/np.std(subject)
#subject = np.maximum(-5, np.minimum(5, subject))


Query = []
for i in range(1000):
    filename = "child_"+"0"*(4-len(str(i)))+str(i)+".txt"
    with open("./child/"+filename, "r") as f:
        for line in f:
            value = float(line)
            if not np.isnan(value):
                Query.append(value)
    print "read", filename

query = np.array(Query)
#query = (query-np.mean(query))/np.std(query)
#query = np.maximum(-5, np.minimum(5, query))


L, K = 8192, 10
for k in range(K):
    lower = np.random.randint(len(query)-L)
    shape = query[lower:lower+L]
    
    with open("single_query%i.bin" % k, "wb") as f:
        f.write(ar.array("f", shape))
    with open("double_query%i.bin" % k, "wb") as f:
        f.write(ar.array("d", shape))
    with open("query%i.txt" % k, "w") as f:
        for value in shape:
            f.write("%s " % value)
    pl.plot(shape)
    pl.show()

with open("single_subject.bin", "wb") as f:
    f.write(ar.array("f", subject))

with open("double_subject.bin", "wb") as f:
    f.write(ar.array("d", subject))

with open("subject.txt", "w") as f:
     for value in subject:
            f.write("%s " % value)
    

print len(query), len(subject)

pl.plot(subject)
pl.show()
