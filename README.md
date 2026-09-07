# ALIVE: Amortizing Linguistic Information into Reusable Visual Evidence for Efficient Image-Text Retrieval

## Introduction

This is the source code of "ALIVE: Amortizing Linguistic Information into Reusable Visual Evidence for Efficient Image-Text Retrieval". 

This manuscript has been submitted to Neurocomputing.

## Requirements

We recommended the following dependencies:

- Python 3.11
- PyTorch 2.11.0
- NumPy 1.26.4
- scipy==1.14.1
- pandas==2.2.3
- scikit-learn==1.5.2
- pillow==10.4.0
- matplotlib==3.9.2
- pyyaml==6.0.2
- opencv-python-headless==4.10.0.84
- imageio==2.37.0
  ......

## Data

We have prepared the caption files for two datasets in  `data/` folder, hence you just need to download the images of the datasets. 
The Flickr30K (f30k) images can be downloaded in [flickr30k-images](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset). The MSCOCO (coco) images can be downloaded in [train2014](http://images.cocodataset.org/zips/train2014.zip), and [val2014](http://images.cocodataset.org/zips/val2014.zip).
We hope that the final data are organized as follows:

```
data
├── coco  # coco captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── testall_ids.txt
│   ├── testall_caps.txt
│   └── id_mapping.json
│
├── f30k  # f30k captions
│   ├── train_ids.txt
│   ├── train_caps.txt
│   ├── test_ids.txt
│   ├── test_caps.txt
│   └── id_mapping.json
│
├── flickr30k-images # f30k images
│
├── coco-images # coco images
│   ├── train2014
│   └── val2014
```

## Train Models

Modify the parameter in the `arguments.py` file. Then run `train.py`:

```
python train.py --dataset f30k --vit_type ./save/vit-base --batch_size 128
```

## Evaluate Models

Modify the parameter in the `eval.py` file. Then run `eval.py`:

```
python eval.py
```
