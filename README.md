# tristar-takehome

## Motivation
There are couple different models I would like to try on this dataset, as well as different pre-processing techniques to increase signal strength.

### Models
1. YOLO - good for benchmarking
2. ResNet - A commonly used medical imaging classifcation model
3. ViT - compare how vision transformers do on this task


## Image processing techniques

#### Contrast Limited Adaptive Histogram Equalization (CLAHE) 
Usually best for greyscale images but I thought I would give it a try here. Good for this data as it regularly has regions that look uniform, but might hide important information. 

CLAHE Image/Normal Image

![alt text](assets/sample_0_clahe.png "CLAHE") ![alt text](assets/sample_0_normal.png "Normal")


#### Reinhard Color normalization
This could be really useful for improving the models performance on this dataset as its goal is to standardize colors across the samples in the dataset. Since there is quite a large variance in the skin tone, lighting,

#### Image sharpening
Helpful for enhancing image features.

#### Guassian Filter

## Approach

To establish baseline performance I trained a YOLO classification. While the image augmentations and various hyperparameters used to train this model are not optimized, most of the augmentations in there beneficial. I left them all in to make things a bit easier.

Yolo Augmentations
![alt text](assets/yolo_augmentations.jpg "Yolo Augmentations")

### Configuration

Docker is a pre-req to run this repository.
Because ultralytics is dependent on pytorch, we can just grab the ultralytics docker image from their registry and use it for training the yolo model, as well as training the ResNet and ViT models.

### Pull down the latest ultralytics image and run it
```bash
./start_docker
```

NOTE- might need to make a dockerfile to support any pre-processing steps I take

#### YOLO
How to evaluate the yolo mode

#### ResNet
How to evaluate the resnet model

## Model Results

| Model | Model Accuracy on Test Set| Pre-Processing Applied|
| :---         | :---:    | ---:          |
| ResNET                | 0.916     |None    |
| ResNET                | 0.910     |CLAHE       |
| YOLO(baseline)        | 0.891     |Standard UltralyticsAugmentations       |
| ResNET                | 0.857     |Reinhard     |
| ViT                   | Cell 5    |Cell 6        |

