<img alt="All-Pairs example" src="examples/all_pairs_survey.png?raw=true" width="400">

# all-pairs

A data generator for studying learning with weak supervision.  The two primary components included in this repo are as follows:

1. The data generator for the All-Pairs problem.  The purpose of this dataset is to explore the limits of learning with weak supervision.  Samples can be generated on the fly during training, an example is provided.
1. The Type-Net model which can be more easily trained to solve the All-Pairs problem than conventional vision models.

See the accompanying paper on arXiv for details: [A New Benchmark and Progress Toward Improved Weakly Supervised Learning](https://arxiv.org/abs/1807.00126)

Code is parameterized with argparse, so see individual files for details of configurations.  

Use `python SCRIPT_NAME.py --help` to see help on each parameter.

# Documentation
### Data generator

The main generator is provided in `allpairs/grid_generator.py`. 
We have included a pytorch dataset and dataloader in `code_pytorch/grid_loader.py`

### Typenet

A simple example of typenet is provided in `code_pytorch/typenet.py`

# Getting Started

### Requirements

See getting started below.  To get started you should have the following:

- git
- pip
- virtualenv (pip install virtualenv)
- Python

### macOS

After cloning or downloading this git repo, install the requirements:

```
cd ml-all-pairs
virtualenv env           (in python3: python3 -m venv env)
source env/bin/activate
pip install -r requirements.txt
```

Test the rendering.

```
python examples/make_survey_strip.py
```

Validate that the images produced are exactly as expected:

```
python test.py
```

Generate samples for analysis; writes the png image files to the "dest" directory and save the ground-truth to a csv file:

```
mkdir samples
python generate.py --pixels 72 --num-pairs 4 --num-classes 4 --num 1000 --dest samples --csv groundtruth.csv
```

### To Train Type-Net

1. Install pytorch.

1. Train the Type-Net model to solve the 4-4 All-Pairs problem.  
```
python train-pytorch-simple.py
```
1. To have more control over the parameters, you can use the following:
```
python train-pytorch.py --num-classes=4 --num-pairs=4
```

### Example Results

To see the results of training the 4-4 All-Pairs problem, run the commands below:

```
python train-pytorch-simple.py | tee examples/results.txt
cd examples
python plot_results.py
```
We plot the maximum validation accuracy because the batch norm moving statistics (used in validation) are often wrong as the weights change.

<img alt="All-Pairs example" src="examples/training-4-4.png?raw=true" width="400">

