library(dtw);

args <- commandArgs(trailingOnly = TRUE)
M<-strtoi(args[3])
N<-strtoi(args[4])

print(args[1])
print(args[2])
print(M)
print(N)

query<-readBin(file(args[1], "rb"), numeric(), n=M, size=8)
subject<-readBin(file(args[2], "rb"), numeric(), n=N, size=8)

start.time <- Sys.time()

for (i in 1:(N-M+1)) {
    dtw(query, subject[i:(i+M-1)], distance_only=TRUE)
}

time.taken <- Sys.time()-start.time

print(time.taken)


