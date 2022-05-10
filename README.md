# Gaze Target Estimation Inspired by Interactive Attention

This repository contains the code for our: Gaze Target Estimation inspired by interactive attention (VSG-IA)


## Overview


The current repository has released the test code and pre-processed data on the GazeFollow dataset. 

The complete code will be released as soon as possible.

- [x] Evaluation code on GazeFollow/VideoTargetAttention datasets
- [ ] Pre-process code
- [ ] Training code


## Instruction
1.Clone our repo and make directory "datasets".

```bash
git clone https://github.com/nkuhzx/VSG-IA
cd VSG-IA
mkdir datasets
```

2.Download the GazeFollow/VideoTargetAttention dataset refer to [ejcgt's work](https://github.com/ejcgt/attention-target-detection) and download them to datasets directory.


3.Download the preprocess file for test
```bash
sh download_preprocess.sh
```

4.Download the model weight for test
```bash
sh download_weight.sh
```

5.OPTIONAL SETTINGS

1)We provide a conda environment.yml file and you can re-create the environment we used.

```bash
conda env create -f environment.yml
```

2)We provide the [test part of the gazefollow dataset](https://drive.google.com/file/d/1X-J9WIE1m3GhRru7lR6ogLCB72cIvgq1/view?usp=sharing) for evaluation, you can download it to datasets directory
and then unzip the gazefollow.zip in this directory.

3)You can download [model weights for GazeFollow](https://drive.google.com/file/d/1vb1TMAz_y9Zy3ajud9mFAQWjmJVKfgo3/view?usp=sharing) to your computer manually and [model weights for VideoTargetAttention](https://drive.google.com/file/d/19o2z4Qp7mcG319KpBFg3F830_xlqx-n5/view?usp=sharing).
The folder 'modelparas' need to contain the model_gazefollow.pt and model_videotargetattention.pt file.

A required structure is:

        VSG-IA/
        ├── datasets/
        │   ├── gazefollow/
        │   │   ├── test/
        │   │   ├── train/
        │   │   └── ...     
        │   ├── gazefollow_graphinfo/
        │   │   └── test_graph_data.hdf5
        │   ├── gazefollow_masks/
        │   │   └── test_masks/
        │   ├── gazefollow_annotation/
        │   │   ├── test_annotation.txt
        │   │   └── train_annotation.txt       
        │   ├── videotargetattention/
        │   │   ├── annotations/
        │   │   ├── images/
        │   │   └── ...     
        │   ├── videotargetattention_graphinfo/
        │   │   └── test_graph_data_vat.hdf5
        │   ├── videotargetattention_masks/
        │   │   ├── CBS This Morning/
        │   │   └── ...    
        │   ├── videotargetattention_annotation/
        │   │   ├── test_annotation.txt
        │   │   └── train_annotation.txt 
        ├── modelparas/
        │   ├── model_videotargetattention.txt
        │   └── model_gazefollow.gt
        ├── vsgia_model/
        │   ├── config/
        │   ├── dataset/
        │   ├── models/
        │   ├── utils/        
        │   ├── main.py
        │   ├── tester.py
        │   ├── main_vat.py
        │   ├── tester_vat.py
        │   └── __init__.py
        ├── download_dataset.sh
        ├── download_weight.sh
        ├── environment.yaml
        └── README.md
        
## Evaluation

### Evaluation on GazeFollow dataset


1.use gpu
```bash
cd vsgia_model
python main.py --gpu --is_test (use gpu)
```

2.only use cpu
```bash
cd vsgia_model
python main.py --is_test (only use cpu)
```

After the evaluation process, the program will reports the Average Distance, Minimum Distance and AUC.

### Evaluation on VideoTargetAttention dataset
1.use gpu

```bash
cd vsgia_model
python main_vat.py --gpu --is_test (use gpu)
```

2.only use cpu
```bash
cd vsgia_model
python main_vat.py --is_test (only use cpu)
```

## Acknowledgement
We thank the associate editors and reviewers for their constructive suggestions.









