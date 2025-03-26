---

# FireNet  

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>  

**Spatio-temporal Multimodal Wildfire Spread Prediction**  

## Setup  
Install the required dependencies using:  
```bash
pip install -r requirements.txt
```  

## Downloading and Preparing the Dataset  
This repository uses the **WildfireSpreadTS** dataset, which is freely available at [Zenodo](https://zenodo.org/records/8006177).  

To speed up training, convert the dataset to HDF5 format by running:  
```bash
python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR
```  

## Running Experiments  
Run training with the following command:  
```bash
python3 train.py --config=cfgs/convlstm_cbam/full_run.yaml \
                 --trainer=cfgs/trainer_single_gpu.yaml \
                 --data=cfgs/data_monotemporal_full_features.yaml \
                 --seed_everything=0 \
                 --trainer.max_epochs=200 \
                 --do_test=True \
                 --data.data_dir YOUR_DATA_DIR \
                 --model_name YOUR_MODEL_NAME
```  

---

## Project Organization
The most up-to-date code compatible with the **WildfireSpreadTS** dataset is currently located in **WSTS**. Since we recently transitioned datasets, we are in the process of migrating documents from **WSTS** to our main module, **FireNet**.  

```
├── LICENSE            
├── Makefile           
├── README.md          
├── WSTS               <- Latest working code compatible with the WildfireSpreadTS dataset
│   ├── cfgs           <- YAML configuration files for trainer, data loader, and model settings
│   ├── src            <- Main working directory containing models and preprocessing code
    │   ├── __init__.py         
    │   └── train.py
    │   ├── dataloader         
    │   ├── models 
├── data
│   ├── external       
│   ├── interim        
│   ├── processed      
│   └── raw            
│
├── docs               
│
├── models             
│
├── notebooks          
│
├── pyproject.toml      
│                         
│
├── references         
│
├── reports            
│   └── figures        
│
├── requirements.txt  
│                         
│
├── setup.cfg          
│
└── firenet            <- Outdated source code for the old dataset, to be integrated with WSTS
```

## References 
**WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction**
S. Gerard, Y. Zhao, and J. Sullivan, “WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction,” Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. [Online]. Available: https://openreview.net/forum?id=RgdGkPRQ03

**CBAM: Convolutional Block Attention Module**
S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” arXiv preprint arXiv:1807.06521, 2018. [Online]. Available: https://arxiv.org/abs/1807.06521

--------

