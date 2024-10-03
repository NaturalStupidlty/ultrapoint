## Installation
```
conda create --name superpoint python=3.8
conda activate superpoint
pip install -r requirements.txt
```

### Datasets

- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
- KITTI Odometry
    - [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
    - [download link](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)


### 1) Training MagicPoint on Synthetic Shapes
```
python train.py ../assets/configs/train/magicpoint_synthetic.yaml magicpoint_synth --eval
tensorboard --logdir assets/logs/magicpoint_synth_2024-09-27_10:44:24
```
You don't need to download synthetic data. You will generate it when first running it.

### 2) Generating pseudo labels
This is the step of homography adaptation for joint training.

#### COCO
```
python generate_pseudo_labels.py ../assets/configs/generate_pseudo_labels/magicpoint_coco_export.yaml
```
#### KITTI
```
python generate_pseudo_labels.py ../assets/configs/generate_pseudo_labels/magicpoint_kitti_export.yaml
```

### 3) Training SuperPoint
You need pseudo ground truth labels from step 2).

#### COCO
```
python train.py ../assets/configs/train/superpoint_coco_heatmap.yaml superpoint_coco
```
#### KITTI
```
python train.py ../assets/configs/train/superpoint_kitti_heatmap.yaml superpoint_kitti
```

### 4) Export/ Evaluate the metrics on HPatches
- Use pretrained model or specify your model in config file
- ```./run_export.sh``` will run export then evaluation.

#### Export
- download HPatches dataset (link above). Put in the $DATA_DIR.
```python export.py <export task> <config file> <export folder>```
- Export keypoints, descriptors, matching
```
python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test
```
#### evaluate
```python evaluation.py <path to npz files> [-r, --repeatibility | -o, --outputImg | -homo, --homography ]```
- Evaluate homography estimation/ repeatability/ matching scores ...
```
python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching
```

### 5) Export/ Evaluate repeatability on SIFT
- Refer to another project: [Feature-preserving image denoising with multiresolution filters](https://github.com/eric-yyjau/image_denoising_matching)
```shell
# export detection, description, matching
python export_classical.py export_descriptor configs/classical_descriptors.yaml sift_test --correspondence

# evaluate (use 'sift' flag)
python evaluation.py logs/sift_test/predictions --sift --repeatibility --homography 
```

- specify the pretrained model

## Pretrained models
### Current best model
- *COCO dataset*
```logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar```
- *KITTI dataset*
```logs/superpoint_kitti_heat2_0/checkpoints/superPointNet_50000_checkpoint.pth.tar```
### model from magicleap
```pretrained/superpoint_v1.pth```

## Jupyter notebook 
```shell
# show images saved in the folders
jupyter notebook
notebooks/visualize_hpatches.ipynb 
```
