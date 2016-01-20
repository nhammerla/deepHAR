#Example usage: Rscript find-and-replace.R /home/shane/deepHAR/eval/analysisScript/18janfANOVAresults/18janPAMAPRNNSBS4/variableImportances.txt /data/EXP/CSV/PAMAP2/rnnsbs.csv
#rm(list = ls())

args <- commandArgs(trailingOnly = TRUE)

textToProcessPath = args[1]#"/Users/shanework/Desktop/hulk/home/shane/deepHAR/eval/analysisScript/18janfANOVAresults/18janPAMAPRNNSBS4/variableImportances.txt"
csvWithVarNamesPath = args[2]#"/Users/shanework/Desktop/hulk/data/EXP/CSV/PAMAP2/rnnsbs.csv"


file1 = file(description = textToProcessPath, open='r' )
ignore = readLines(file1, 4)


mainEffectsString = readLines(file1, 1)
pairwiseInteractionEffectsString = readLines(file1,1)
restOfOutput = readLines(file1)
restOfOutput = restOfOutput[3:100]


varNames = names(read.csv(csvWithVarNamesPath)) 
varNames
varNames = varNames[-length(varNames)]
i=0:(length(varNames)-1)
Xs = paste('X',i,sep='')
length(varNames)
length(Xs)
varNames
Xs
for (i in 1:length(restOfOutput)){
	  restOfOutput[i] = paste0(restOfOutput[i],'\n')
}

#restOfOutput = paste(restOfOutput )
restOfOutput = gsub("NA","", restOfOutput)
for(i in 1:length(varNames)){
	  #print( paste(varNames[i], Xs[i]) )
	  restOfOutput<-gsub(Xs[i], varNames[i], restOfOutput)
}
for(i in 1:length(restOfOutput)){
	  if(restOfOutput[i]=='\n'){
		      restOfOutput[i]==''
  }
}
#cat(restOfOutput)
write((restOfOutput), file='ans.txt')
