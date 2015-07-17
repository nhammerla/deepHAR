
# no Weightings
# two layer, 64
mkdir 2L
mkdir 2L/64
th ../models/RNN/main.lua -logdir '2L/64/l0.01' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 64 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/64/l0.05' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 64 -learningRate 0.05 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/64/l0.1'  -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 64 -learningRate 0.1  -learningRateDecay 5e-6 -maxInNorm 3
# two layer, 128
mkdir 2L/128
th ../models/RNN/main.lua -logdir '2L/128/l0.01' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/128/l0.05' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 128 -learningRate 0.05 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/128/l0.1'  -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 128 -learningRate 0.1  -learningRateDecay 5e-6 -maxInNorm 3
# weighted
th ../models/RNN/main.lua -logdir '2L/64/l0.01w1'  -classWeights '0.5,1,0.7,1' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 64  -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/64/l0.01w2'  -classWeights '1,2.5,1,3'   -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 64  -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/128/l0.01w1' -classWeights '0.5,1,0.7,1' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '2L/128/l0.01w2' -classWeights '1,2.5,1,3'   -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 2 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3

# three layer, 64
mkdir 3L
mkdir 3L/64
th ../models/RNN/main.lua -logdir '3L/64/l0.01' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 64 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/64/l0.05' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 64 -learningRate 0.05 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/64/l0.1'  -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 64 -learningRate 0.1  -learningRateDecay 5e-6 -maxInNorm 3
# three layer, 128
mkdir 3L/128
th ../models/RNN/main.lua -logdir '3L/128/l0.01' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/128/l0.05' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 128 -learningRate 0.05 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/128/l0.1'  -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 128 -learningRate 0.1  -learningRateDecay 5e-6 -maxInNorm 3
# weighted
th ../models/RNN/main.lua -logdir '3L/64/l0.01w1'  -classWeights '0.5,1,0.7,1' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 64  -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/64/l0.01w2'  -classWeights '1,2.5,1,3'   -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 64  -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/128/l0.01w1' -classWeights '0.5,1,0.7,1' -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
th ../models/RNN/main.lua -logdir '3L/128/l0.01w2' -classWeights '1,2.5,1,3'   -data ../data/PD.data -ignore -ignoreClass 0 -numLayers 3 -layerSize 128 -learningRate 0.01 -learningRateDecay 5e-6 -maxInNorm 3
