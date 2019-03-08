
# Precise simulation of electromagnetic calorimeter showers using a Wasserstein Generative Adversarial Network
...the training code of the corresponding publication.

# Site and repository under development!

## References
* Computing and Software for Big Science: [https://link.springer.com/article/10.1007%2Fs41781-018-0019-7](https://link.springer.com/article/10.1007%2Fs41781-018-0019-7)
* preprint on arXiv: [https://arxiv.org/abs/1807.01954](https://arxiv.org/abs/1807.01954)


## Recommended hardware prerequisites
- NVIDIDA GPU GTX 1080 or better
- 4GB RAM (for loading the training data)

## Software prerequisites
The code has been tested with:
- Python 2.7
- numpy v 1.16 or newer
- ROOT 5 or 6
- Tensorflow v1.5 with keras and GPU support (recommended)

## Input data files
Input data files for the training are provided on the CERN computing infrastructure:
- ```/afs/cern.ch/work/t/tquast/public/Sept2017_HGCALTB_Sim```

## Running the code
- Training command: ```python training.py --EpochStart 0  --Nepochs 150 --checkpoint_dir <directory where network files are stored> --input_dir <Directory where the input files are>``` (training duration for 150 epochs with recommended hardware ~30h)
