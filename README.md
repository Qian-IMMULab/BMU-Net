# BMU-Net: A PyTorch Implementation

This is a PyTorch/GPU implementation of the paper "A multimodal machine-learning model for the stratification of breast cancer risk".

## Install

Clone repo and install requirements.txt

```python
git clone https://github.com/IMMULab/BMUNet
cd BMUNet
pip install -r requirements.txt
```

## Data

### Input images

Mammography module:  two mammography images (paired CC and MLO views)

Ultrasound module: six ultrasound images (including transverse and longitudinal views of B-mode, colour Doppler and elastography images)

BMU-Net model: unimodal, bimodal, multimodal inputs

## Inference

1. Download models from the latest release 
2. Make a csv file with own data to test-path (example as `dataset_csv/mamm_test.csv`)

```python
# Mamm module
python test_mamm.py --weight-path ./BMUNet_weight/mamm/mamm_model.bin --test-path ./dataset_csv/mamm_test.csv
# Us module
python test_us.py --weight-path ./BMUNet_weight/us/us_model.bin --test-path ./dataset_csv/us_test.csv
# Bmu model
python test_bmu.py --weight-path ./BMUNet_weight/bmu/bmu_images_metadata_model.bin --test-path ./dataset_csv/bmu_test.csv
```

## Training

1. Make csv files with train and val dataset
2. Convert DICOM format into PNG files `src/data/convert_dicom_to_png.py`
3. Calculate the mean and std of mammography images in the train set(mamm module and bmu model) `src/data/get_dataset_stats.py`
4. Change training parameters as required `config.yml`

```python
# Multi gpu parallelism
sh run_distributed.sh main_mamm.py

# Single gpu
python main_mamm.py # mamm module
python main_us.py # us module
python main_bmu.py # bmu model
```

## **License**

**AGPL-3.0 License:** See the [LICENSE](https://github.com/IMMULab/BMUNet/blob/main/LICENSE) file for more details.

## Note

The Mirai model(`mirai_model/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p`) is used as pretrained model for Mammography module, where [onconet](https://github.com/yala/Mirai) is the Mirai model code