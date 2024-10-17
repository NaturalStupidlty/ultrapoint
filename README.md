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


### 1. Training MagicPoint on Synthetic Shapes
```python train.py ../assets/configs/train/magicpoint_synthetic.yaml magicpoint_synth```

### Tensorboard logs
```bash
tensorboard --logdir /home/ihor/projects/ultrapoint/assets/logs/
```

You don't need to download synthetic data. You will generate it when first running it.

### 2. Generating pseudo labels
This is the step of homography adaptation for joint training.

#### SSIR
```python generate_pseudo_labels.py ../assets/configs/generate_pseudo_labels/magicpoint_ssir.yaml```

### 3. Training SuperPoint

#### SSIR
```python train.py ../assets/configs/train/superpoint_ssir.yaml superpoint_ssir```

### 4. Export/ Evaluate the metrics on HPatches
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
