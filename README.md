# pytorch-superpoint

This is a PyTorch implementation of  "SuperPoint: Self-Supervised Interest Point Detection and Description." Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich. [ArXiv 2018](https://arxiv.org/abs/1712.07629).
This code is partially based on the tensorflow implementation
https://github.com/rpautrat/SuperPoint.

Please be generous to star this repo if it helps your research.
This repo is a bi-product of our paper [deepFEPE(IROS 2020)](https://github.com/eric-yyjau/pytorch-deepFEPE.git).

## Differences between our implementation and original paper
- *Descriptor loss*: We tested descriptor loss using different methods, including dense method 
(as paper but slightly different) and sparse method. We notice sparse loss can converge more efficiently with similar
performance. The default setting here is sparse method.


## Installation
```
conda create --name superpoint python=3.8
conda activate superpoint
pip install -r requirements.txt
```

### Path setting
- paths for datasets ($DATA_DIR), logs are set in `.env`

### Dataset
Datasets should be downloaded into $DATA_DIR. The Synthetic Shapes dataset will also be generated there. 
The folder structure should look like:

```
datasets/ ($DATA_DIR)
|-- COCO
|   |-- train2014
|   |   |-- file1.jpg
|   |   `-- ...
|   `-- val2014
|       |-- file1.jpg
|       `-- ...
`-- HPatches
|   |-- i_ajuntament
|   `-- ...
`-- synthetic_shapes  # will be automatically created
`-- KITTI (accumulated folders from raw data)
|   |-- 2011_09_26_drive_0020_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_28_drive_0001_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_29_drive_0004_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_09_30_drive_0016_sync
|   |   |-- image_00/
|   |   `-- ...
|   |-- ...
|   `-- 2011_10_03_drive_0027_sync
|   |   |-- image_00/
|   |   `-- ...
```
- MS-COCO 2014 
    - [MS-COCO 2014 link](http://cocodataset.org/#download)
- HPatches
    - [HPatches link](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
- KITTI Odometry
    - [KITTI website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
    - [download link](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)


### 1) Training MagicPoint on Synthetic Shapes
```
python train.py assets/configs/train/magicpoint_synthetic.yaml magicpoint_synth --eval
tensorboard --logdir assets/logs/magicpoint_synth_2024-09-27_10:44:24
```
you don't need to download synthetic data. You will generate it when first running it.
Synthetic data is exported in `./datasets`. You can change the setting in `.env`.

### 2) Exporting detections on MS-COCO / kitti
This is the step of homography adaptation to export pseudo ground truth for joint training.
- make sure the pretrained model in config file is correct
- make sure COCO dataset is in '$DATA_DIR' (defined in .env)
- you can export hpatches or coco dataset by editing the 'task' in config file

#### export coco
```
python export.py assets/configs/generate_pseudo_labels/magicpoint_coco_export.yaml magicpoint_coco_homography
```
#### export kitti
```
python export.py assets/configs/generate_pseudo_labels/magicpoint_kitti_export.yaml magicpoint_kitti_homography
```
<!-- #### export tum
- config
  - check the 'root' in config file
  - set 'datasets/tum_split/train.txt' as the sequences you have
```
python export.py export_detector_homoAdapt configs/magicpoint_tum_export.yaml magicpoint_base_homoAdapt_tum
``` -->


### 3) Training Superpoint on MS-COCO/ KITTI
You need pseudo ground truth labels from step 2). Then, as usual, you need to set config file before training.
- config file
  - root: specify your labels root
  - root_split_txt: where you put the train.txt/ val.txt split files (no need for COCO, needed for KITTI)
  - labels: the exported labels from homography adaptation
  - pretrained: specify the pretrained model (you can train from scratch)
- 'eval': turn on the evaluation during training 

#### General command
```
python train4.py <train task> <config file> <export folder> --eval
```

#### COCO
```
python train4.py train configs/superpoint_coco_train_heatmap.yaml superpoint_coco --eval --debug
```
#### kitti
```
python train4.py train configs/superpoint_kitti_train_heatmap.yaml superpoint_kitti --eval --debug
```

- set your batch size (originally 1)
- refer to: 'train_tutorial.md'

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
