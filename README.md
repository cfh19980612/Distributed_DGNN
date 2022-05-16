# The implementation for distributed DGNN algorithms

## Create the conda environment
```
conda create -n 'DGNN' python=3.8
```

## Install the requirements
```
pip install -r requirements.txt
```

## Run test for local training
### For training DGNN on a single node:
```
python test.py --json-path=./parameters.json --dataset=Epinion --test-type=local 
```

### For training DGNN on multiple nodes:
```
python test.py --json-path=./parameters.json --dataset=Epinion --world_size=4 --gate=True --test-type=dp
```
#### Configuration parameters
--- dataset: 'Enron', 'Epinion', 'Youtube', 'Flickr'
--- world_size: the minimum is 1
--- gate: True for customized communication and False, otherwise
--- test-type: dp for data parallel with CPU processes, ddp for distributed data parallel with GPUs
