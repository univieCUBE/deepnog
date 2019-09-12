This repository is devided into two parts:
* Deep learning based project to develop and train neural networks for orthologous group predictions in eggNOG 5.0 (and possibly other databases). 
* Deep learning based tool, built upon the previous project, to predict for a given protein sequence its corresponding most likely orthologous group in eggNOG 5.0. 

# Folder structure
### data

Stores notebooks and python script used to preprocess data for training purposes.

### train
Stores notebooks used for training NN-architectures used in deepNOG-prediction-tool.

### tool

DeepNOG command line tool for orthologous group predictions. See README.md in tool/ for detailed usage description.