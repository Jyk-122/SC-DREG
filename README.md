# Single-View-Semantically-Consistent-Deformable-2D-3D-Registration
Implementation of paper - Towards Semantically-Consistent Deformable 2D-3D Registration for 3D Craniofacial Structure Estimation from A Single-View Lateral Cephalometric Radiograph

## Setup
The code can be run under any environment with >= Python 3.7 and >= PyTorch 1.8.

Install the following packages:
```
pip install -r requirements.txt
```

The parameters needed in training and testing can be downloaded from [model_downloads](https://drive.google.com/drive/folders/1dnHwxNSsBbt2NoXKIAehunFnwS5kQDop?usp=sharing), and should be placed in `./models`.
The link contains reference cbct file, pretrained unet model of mandible segmentation for clinical X-ray image, pca coeff & mean parameters of cbct, and trained results of our coarse-to-fine model for CBCT.

## Training
We provide the training code. You can run the following commands based on specified `DATASET_PATH`:
```
export DATASET_PATH=/path/to/dataset
export CUDA_VISIBLE_DEVICES=1
python train.py \
    --model_name cbct_c2f_model \
    --pca_dim 60 \
    --param_path ./models \
    --dataset_path $DATASET_PATH \
```

## Testing
We provide trained results of CBCT in our paper. The example input X-Ray images are placed in `./tests`.
You can run the following commands to reconstruct the 3D CBCT:
```
export CUDA_VISIBLE_DEVICES=1
python test.py \
    --pca_dim 60 \
    --ckpt_path ./models/cbct_c2f_model_ckpt.tar \
    --param_path ./models \
    --input_path ./tests \
    --output_dir ./tests/output
``` 
The results will be saved in path `./tests/output`.