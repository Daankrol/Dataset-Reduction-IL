<p align="center">
    <br>
        &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
        <img src="https://github.com/decile-team/cords/blob/2d78caf54d871976bd703f9fd14e7906264fffa1/docs/source/imgs/cords_logo.png" width="500"/>
    </br>
    <br>
        <strong> COResets and Data Subset selection </strong>
    </br>
</p>

<p align="center">
    <a href="https://github.com/decile-team/cords/blob/main/LICENSE.txt">
        <img alt="GitHub" src="https://img.shields.io/github/license/decile-team/cords?color=blue">
    </a>
    <a href="https://decile.org/">
        <img alt="Decile" src="https://img.shields.io/badge/website-online-green">
    </a>  
    <a href="https://cords.readthedocs.io/en/latest/">
        <img alt="Documentation" src="https://img.shields.io/badge/docs-passing-brightgreen">
    </a>
    <a href="#">
        <img alt="GitHub Stars" src="https://img.shields.io/github/stars/decile-team/cords">
    </a>
    <a href="#">
        <img alt="GitHub Forks" src="https://img.shields.io/github/forks/decile-team/cords">
    </a>
</p>

<h3 align="center">
<p>Reduce end to end training time from days to hours and hours to minutes using coresets and data selection.
</h3>


## In this README

- [What is CORDS?](#what-is-cords?)
- [Installation](#installation)
    - [Installing via pip](#installing-via-pip)
    - [Installing from source](#installing-from-source)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Results](#results)
- [Publications](#publications)


## What is CORDS?

[CORDS](https://cords.readthedocs.io/en/latest/) is COReset and Data Selection library for making machine learning time, energy, cost, and compute efficient. CORDS is built on top of pytorch. Deep Learning systems are extremely compute intensive today with large turn around times, energy inefficiencies, higher costs and resourse requirements (see https://arxiv.org/pdf/1907.10597.pdf and https://arxiv.org/abs/1906.02243 for more details on quantifications of these impacts). CORDS is an effort to make deep learning more energy, cost, resource and time efficient while not sacrificing accuracy. The following are the goals CORDS tries to achieve:

<p align="center"><i><b>Data Efficiency</b></i></p>
<p align="center"><i><b>Reducing End to End Training Time</b></i></p>
<p align="center"><i><b>Reducing Energy Requirement</b></i></p>
<p align="center"><i><b>Reducing Resource (GPU) Requirement and Costs</b></i></p>

The primary purpose of CORDS is to select the right representative data subsets from massive datasets, and it does so iteratively. CORDS uses some recent advances in data subset selection and particularly, ideas of coresets and submodularity select such subsets. CORDS implements a number of state of the art data subset selection algorithms 
and coreset algorithms. Some of the algorithms currently implemented with CORDS include:

- [GLISTER [1]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.glisterstrategy)
- [GradMatch [2]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.ompgradmatchstrategy)
- [CRAIG [2,3]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.craigstrategy)
- [SubmodularSelection [4,5,6]](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.submodularselectionstrategy) (Facility Location, Feature Based Functions, Coverage, Diversity)
- [RandomSelection](https://cords.readthedocs.io/en/latest/strategies/cords.selection_strategies.supervisedlearning.html#module-cords.selectionstrategies.supervisedlearning.randomstrategy)

We are continuously incorporating newer and better algorithms into CORDS. Some of the features of CORDS includes:

- Reproducability of SOTA in Data Selection and Coresets: Enable easy reproducability of SOTA described above. We are trying to also add more algorithms so if you have an algorithm you would like us to include, please let us know,.
- Benchmarking: We have benchmarked CORDS (and the algorithms present right now) on several datasets including CIFAR-10, CIFAR-100, MNIST, SVHN and ImageNet. 
- Ease of Use: One of the main goals of CORDS is that it is easy to use and add to CORDS. Feel free to contribute to CORDS!
- Modular design: The data selection algorithms are separate from the training loop, thereby enabling modular design and also varied scenarios of utility.
- Broad number of usecases: CORDS is currently implemented for simple image classification tasks, but we are working on integrating a number of additional use cases like object detection, speech recognition, semi-supervised learning, Auto-ML, etc.

## Installation

1. To install latest version of CORDS package using PyPI:

    ```python
    pip install -i https://test.pypi.org/simple/ cords
    ```

2. To install using source:

    ```bash
    git clone https://github.com/decile-team/cords.git
    cd cords
    pip install -r requirements/requirements.txt
    ```


## Documentation

Learn more about CORDS at our [documentation](https://cords.readthedocs.io/en/latest/).


## Tutorials

Here are some [tutorials](https://github.com/decile-team/cords/tree/main/notebooks) to get you started with CORDS. 

- [General Data Selection](https://github.com/decile-team/cords/blob/main/notebooks/general_dataselection.py)
- [GLISTER](https://github.com/decile-team/cords/blob/main/notebooks/glister_example.py)
- [Random Selection](https://github.com/decile-team/cords/blob/main/notebooks/tutorial_random.ipynb)


## Results

The below link contains the jupyter notebook link for cifar10 timing analysis experiments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) CIFAR10 Notebook](https://colab.research.google.com/drive/1xT6sGmDGMz8XBDmOKs5cl1cipX0Ss1sh?usp=sharing)

Results are obtained by running each dataset with different strategies for 300 epochs. The following experimental plots shows the relative test error vs speed up for different strategies.

### CIFAR10

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/5c705778a5444c07e4bd4b123d217300fc8bcf54/docs/source/imgs/cifar10_test_accuracy.png" width="700"/>
    </br>
</p>

### CIFAR100

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/5c705778a5444c07e4bd4b123d217300fc8bcf54/docs/source/imgs/cifar100_test_accuracy.png" width="700"/>
    </br>
</p>

### MNIST

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/5c705778a5444c07e4bd4b123d217300fc8bcf54/docs/source/imgs/mnist_test_accuracy.png" width="700"/>
    </br>
</p>

### SVHN

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/5c705778a5444c07e4bd4b123d217300fc8bcf54/docs/source/imgs/svhn_test_accuracy.png" width="700"/>
    </br>
</p>

### ImageNet

<p align="center">
    <br>
        <img src="https://github.com/decile-team/cords/blob/5c705778a5444c07e4bd4b123d217300fc8bcf54/docs/source/imgs/imagenet_test_accuracy.png" width="700"/>
    </br>
</p>


## Publications

[1] Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer, [GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning](https://arxiv.org/abs/2012.10630), 35th AAAI Conference on Artificial Intelligence, AAAI 2021

[2] S Durga, Krishnateja Killamsetty, Abir De, Ganesh Ramakrishnan, Baharan Mirzasoleiman, Rishabh Iyer, Grad-Match: A Gradient Matching based Data Selection Framework for Efficient Learning

[3] Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. [Coresets for Data-efficient Training of Machine Learning Models](https://arxiv.org/abs/1906.01827). In International Conference on Machine Learning (ICML), July 2020

[4] Kai Wei, Rishabh Iyer, Jeff Bilmes, [Submodularity in Data Subset Selection and Active Learning](http://proceedings.mlr.press/v37/wei15-supp.pdf), International Conference on Machine Learning (ICML) 2015

[5] Vishal Kaushal, Rishabh Iyer, Suraj Kothiwade, Rohan Mahadev, Khoshrav Doctor, and Ganesh Ramakrishnan, [Learning From Less Data: A Unified Data Subset Selection and Active Learning Framework for Computer Vision](https://arxiv.org/abs/1901.01151), 7th IEEE Winter Conference on Applications of Computer Vision (WACV), 2019 Hawaii, USA

[6] Wei, Kai, et al. [Submodular subset selection for large-scale speech training data](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.496.6287&rep=rep1&type=pdf), 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.
