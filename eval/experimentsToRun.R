# Make CSV file with combinations of hyperparameters for DNN, CNN and RNN neural networks           
# Example run: Rscript experimentsToRun.R 'DNN' 200 will create 200 combinations of hyperparameters for a DNN.
 
args <- commandArgs(trailingOnly = TRUE)
networkType<-(args[1])
n<-as.integer(args[2])

#Helper functions
#Should we use base e, base 2, or base 10?
Uniform 		<- function(n, a, b)	{ runif(n,a,b) }
logUniform 		<- function(n, a, b)	{ exp( Uniform(n,log(a),log(b))  ) }
discreteUniform		<- function(n, a, b)	{ ceiling(runif(n, a-1, b)) }
discreteLogUniform	<- function(n, a, b)	{ ceiling(logUniform(n, a-1, b)) }

newHyperparamComb <- function(networkType, n){
	if(networkType=="DNN"){
	  numLayers 		= discreteUniform(n,1,10)
	  layerSize 		= discreteLogUniform(n,128,2048)
	  learningRate 		= logUniform(n,0.00001,0.5)
	  dropout 		= logUniform(n,0.01, 0.5)
	  momentum 		= logUniform(n,0.01,0.99)
	  learningRateDecay 	= logUniform(n,10e-7, 10e-4)
	  maxInNorm 		= logUniform(n,0.5,5)

	  row = cbind( numLayers, layerSize, learningRate, dropout, momentum, learningRateDecay, maxInNorm)
	  }

	if(networkType=="CNN"){
	  numLayers 		= discreteUniform(n,1,10)
	  layerSize 		= discreteLogUniform(n,128,2048)
	  learningRate 		= logUniform(n,0.00001,0.5)
	  #dropout 		= logUniform(n,0.01, 0.5)
	  momentum 		= logUniform(n,0.01,0.99)
	  learningRateDecay 	= logUniform(n,10e-7, 10e-4)
	  maxInNorm 		= logUniform(n,0.5,5)

	kW1 = 
	kW2 = 
	kW3 = 
	nF1 = 
	nF2 =
	nF3 = 
	mW1 = 
	mW2 = 
	mW3 =
	dropout1 = 
	dropout2 = 
	dropout3 = 
	dropoutFull = 
	
	imbalanced = 

	row = 	
	}

	if(networkType="RNN"){
	  numLayers 		= discreteUniform(n,1,10)
	  layerSize 		= discreteLogUniform(n,128,2048)
	  learningRate 		= logUniform(n,0.00001,0.5)
	  dropout 		= logUniform(n,0.01, 0.5)
	  momentum 		= logUniform(n,0.01,0.99)
	  learningRateDecay 	= logUniform(n,10e-7, 10e-4)
	  maxInNorm 		= logUniform(n,0.5,5)




	}
}

outputTable = newHyperparamComb(networkType, n)
write.csv(outputTable, file = 'hyperparameterCombinations.csv', row.names=FALSE)
