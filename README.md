Here's a cleaner and more structured version of your README with improved clarity and formatting:

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

## Citations  
- The **WSTS** module was forked from [SebastianGer/WildfireSpreadTS](https://github.com/SebastianGer/WildfireSpreadTS.git).  
- All code related to the **CBAM architecture** is our own.  

---

## Project Organization
The most up-to-date code compatible with the **WildfireSpreadTS** dataset is currently located in **WSTS**. Since we recently transitioned datasets, we are in the process of migrating documents from **WSTS** to our main module, **FireNet**.  

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── WSTS               <- **Latest working code compatible with the WildfireSpreadTS dataset**
│   ├── cfgs           <- YAML configuration files for trainer, data loader, and model settings
│   ├── src            <- Main working directory containing models and preprocessing code
    │   ├── __init__.py         
    │   └── train.py
    │   ├── dataloader         
    │   ├── models 
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         firenet and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── firenet            <- **Outdated source code for the old dataset, to be integrated with WSTS**  
```

--------

