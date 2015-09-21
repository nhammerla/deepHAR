#This R script will take as argument the GPU to run on. It will compute manyneural network runs with random sets of hyperparameter sets as detailed below:
#
#
#
# Layer size	   		Discrete log-uniform 	10..2000
# Number of layers		Random choice from 	1..10
# Learning rate			Cts log-uniform  	0.00001..0.9
# Dropout	 		cts log-uniform		0.01..0.99
# Momentum 			cts uniform 		0.01..0.99
# learningRateDecay 		cts log-uniform 	0.00001..0.9
# maxInNorm 			cts log-uniform		0.00001..10
#           
#
# Suggested use: create one screen bash instance per GPU available
# Then in each individual screen, do:
# > Rscript experimentsToRun.R 1
# to run on GPU number 1.  
 
logUniform <- function(a,b){
  exp(runif(1, log(a), log(b)))
}

newHyperparamComb <- function(datafile, gpu){
  numLayers = round(runif(1,1,10))
  layerSize = round(logUniform(10,2000))
  learningRate = logUniform(0.00001,0.9)
  dropout = logUniform(0.01, 0.99)
  momentum = logUniform(0.01,0.99)
  learningRateDecay = logUniform(0.00001, 0.9)
  maxInNorm = logUniform(0.00001,10)
  
  cmd=paste('th ../models/DNN/main.lua -gpu',gpu,
        ' -datafile',datafile,
        ' -numLayers',numLayers,
        ' -layerSize',layerSize,
        ' -learningRate',learningRate,
        ' -dropout',dropout,
        ' -momentum',momentum,
        ' -learningRateDecay',learningRateDecay,
        ' maxInNorm',maxInNorm)
print(cmd)  
system(cmd)
 
}


args <- commandArgs(trailingOnly = TRUE)
gpuToUse<-as.integer(args[1])

for (i in 1:5)  {
  newHyperparamComb('../data/oppChal/opportunityShane.dat',gpuToUse)
}
