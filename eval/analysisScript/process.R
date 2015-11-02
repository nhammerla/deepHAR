args <- commandArgs(trailingOnly = TRUE)
fileToRead<-(args[1])
x = read.csv(fileToRead)
x = x[, ncol(x)>0.4]
write.table(x,paste('noHeader',fileToRead,sep=''),row.names=FALSE, col.names=FALSE,sep=',')
