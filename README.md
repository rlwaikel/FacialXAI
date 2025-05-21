# Facial XAI

This repository contains the code and data for the paper "Evaluation of a Deep Learning and XAI based Facial Phenotyping Tool for Genetic Syndromes: A Clinical User Study"


## How to reproduce the figures

Create a conda environment with the required packages. The code was tested with Python 3.12:  
```bash conda env create -n xai python=3.12```



```bash
├── code  
│   ├── data/
│   │   ├── images/*.png: 18 face images used in the user study
│   │   ├── user-study/*.csv: raw Qualtrics output
│   │   ├── xai-figures/
│   │   ├── IRB002178.Version_1_Survey.pdf
│   │   ├── IRB002178.Version_2_Survey.pdf
│   ├── output/*
│   ├── create_merged_dataset.py
│   ├── make_plots.py
```

* See stimuli images in `data/images/` and the IRB documents in `data/user-study/`. The IRB documents contain the survey questions and the form. The survey was conducted using Qualtrics. The `data/xai-figures/` folder contains the XAI figures used in the survey. 

* Raw Qualtric outputs are difficult to read, thus first we extract the data from the Qualtrics output and save it in a more readable format:  
```bash python create_merged_file.py```

* The script `make_plots.py` generates the figures in the paper. It takes the output of the previous step (```merged_table.csv```) as input.  
```bash python make_plots.py```