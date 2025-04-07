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
python3 baselines/src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR
```  

## Running Experiments  
Run training with the following command:  
```
python3 train.py
```  

---

## Project Organization
```
├── LICENSE            
├── Makefile           
├── README.md          
├── baselines          <- Baseline models (persistence, logistic regression, convLSTM, convLSTM + CBAM, UTAE)
│   ├── cfgs           <- YAML configuration files for trainer, data loader, and baseline model settings
│   ├── src            <- Main working directory containing baseline models and preprocessing code
    │   ├── __init__.py         
    │   └── train.py
    │   ├── dataloader         
    │   ├── models 
├── data       
│   ├── processed      
│   └── raw                        
│
├── prithvi                      <- Latest working code modifications to Prithvi 100M ViT
    ├── __init__.py
    ├── config.json
    ├── firenet_prithvi.py       <- Linear mapping + Prithvi encoder + SegFormer decoder
    ├── pretrain.py              <- Code for pretraining Prithvi on our dataset
    ├── prithvi_dataloader.py
    ├── prithvi_mae.py
    ├── exploration
├── cnn2plus1d               <- (2+1)D CNN with Atrous Spatial Pyramid Pooling (ASPP)
    ├── __init__.py
    ├── cnn2plus1d.yaml
    ├── firenet3dcnn.py
    ├── CBAM.py

├── notebooks          
│
├── pyproject.toml      
│                         
│       
│
├── reports            
│   └── figures        
│
├── requirements.txt  
│                         
│
├── setup.cfg          
```

## References 
**WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction**
S. Gerard, Y. Zhao, and J. Sullivan, “WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction,” Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. [Online]. Available: https://openreview.net/forum?id=RgdGkPRQ03

**CBAM: Convolutional Block Attention Module**
S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “CBAM: Convolutional block attention module,” arXiv preprint arXiv:1807.06521, 2018. [Online]. Available: https://arxiv.org/abs/1807.06521

--------

