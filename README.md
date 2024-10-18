## Installation
```bash
conda create --name superpoint python=3.8
conda activate superpoint
pip install -r requirements.txt
```

### Datasets

- [MS-COCO 2014 ](http://cocodataset.org/#download)
- [HPatches](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
- [KITTI Odometry](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)


### 1. Training MagicPoint [Synthetic Shapes]
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

### 4. Evaluate the metrics [HPatches]

#### Export

```python export.py export_descriptor  configs/magicpoint_repeatability_heatmap.yaml superpoint_hpatches_test```

#### Evaluate

```python evaluation.py logs/superpoint_hpatches_test/predictions --repeatibility --outputImg --homography --plotMatching```
