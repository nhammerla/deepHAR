numColsAtEndToIgnore=1

args <- commandArgs(trailingOnly = TRUE)
fileToRead<-(args[1])
minmeanF1score<-as.numeric(args[2])
x = read.csv(fileToRead, sep=',', header=TRUE)
x = x[,2:(ncol(x)-numColsAtEndToIgnore)]
x = subset(x, x[,ncol(x)]>minmeanF1score) 
print('First 6 rows of results being processed:')
head(x)
write.table(x, paste('noHeader',fileToRead,sep=''),row.names=FALSE, col.names=FALSE,sep=',')
