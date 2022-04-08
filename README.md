# Regime Switching Time-series Generator

This project generates data streams switching from different models that are based in dataset with different characteristics.



## Setup and Usage

Download the source code from this repository and install the requirements from `requirements.txt` using PyPi.

The Python code uses the R `rugarch` library (version 1.4.1) through the `rpy2` wrapper. 

A path to a functioning R environment with the above-mentioned version of `rugarch` installed needs to be given in the config file `config.yaml`. 

To generate a synthetic set, run `python -m src.generator` . Datasets to represent market data with different states, and changes between these states need to be specified beforehand in `config.yaml`. Parameter ranges to explore to fit models to these states, and the type of shifts, are also specified in this config file.



## More Information

For more information about this generator, see the PhD thesis **Adaptive Algorithms For Classification On High-Frequency Data Streams: Application To Finance**.
