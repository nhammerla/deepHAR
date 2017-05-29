## deepHAR
Code repository for experiments on deep architectures for HAR in ubicomp.
Using this code you will be able to replicate some of the experiments described in our IJCAI 2016 paper:
```
@article{hammerla2016deep,
  title={Deep, convolutional, and recurrent models for human activity recognition using wearables},
  author={Hammerla, Nils Y and Halloran, Shane and Ploetz, Thomas},
  journal={IJCAI 2016},
  year={2016}
}
```

## Disclaimer
This code is still incomplete. At the moment only the bi-directional RNN will work on the opportunity data-set.

## Installation
```
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh

# after installation, we need some additional packages
#HDF5 luarock
sudo apt-get install libhdf5-serial-dev hdf5-tools
git clone https://github.com/deepmind/torch-hdf5
cd torch-hdf5
luarocks make hdf5-0-0.rockspec LIBHDF5_LIBDIR="/usr/lib/x86_64-linux-gnu/"

# json
luarocks install json

# RNN support
luarocks install torch
luarocks install nn
luarocks install dpnn
luarocks install torchx
luarocks install rnn

# we use python3
pip3 install h5py
pip3 install simplejson
pip3 install numpy
```

## Usage
First download and extract the Opportunity dataset. Then use the provided python script in the `data` directory to prepare the training/validation/test sets.
```
cd data
python3 data_reader.py opportunity /path/to/OpportunityUCIDataset
```
This will generate two hdf5-files that are read by the lua scripts, `opportunity.h5` and `opportunity.h5.classes.json`.

To train the bi-directional RNN that we have found to work best on this set run the following commands:
```
cd models/RNN
th main_brnn.lua -data ../../data/opportunity.h5 -cpu \
                 -layerSize 179 -maxInNorm 2.283772707 \
                 -learningRate 0.02516758 -sequenceLength 81 \
                 -carryOverProb 0.915735543 -numLayers 1 \
                 -logdir EXP_brnn
```
This will train a model only using your CPUs, which will take a while (make sure you have some form of BLAS library installed). On my laptop this will take approx. 5 min per epoch, and it will likely not converge before epoch 60. If your environment is set up for gpu-based computation, try using `-gpu 1` instead of the `-cpu` flag for a significant speedup.

## Other models
The python-based `data_reader.py` is new and substitutes for the original but unmaintainable Matlab-scripts used previously. So far it only supports `opportunity` and sample-based evaluation, which will be addressed shortly.