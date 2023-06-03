# LayoutAnalysis_YOLOv7

## Traning

## Installation

``` shell
# Clone YOLOv7.
git clone https://github.com/WongKinYiu/yolov7.git

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# go to code folder
cd /yolov7

# pip install required packages
pip install -r requirements.txt
```

</details>

## Training
Download [`pretrain model`](https://github.com/WongKinYiu/yolov7#performance)

Edit config

``` shell
# config/data_custom.yaml
train: dataset/train.txt
val: dataset/val.txt 
test: dataset/val.txt  

# number of classes
nc: 3

# class names
names: [ 'header', 'table', 'text']

# config/yolov7.yaml
# parameters
nc: 3  # number of classes
...
```

Single GPU training

``` shell
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data config/data_custom.yaml --img 640 640 --cfg config/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
ml
```

Multiple GPU training

``` shell
# train p5 models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data config/data_custom.yaml --img 640 640 --cfg config/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml

```

## Transfer learning

[`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt)

Single GPU finetuning for custom dataset

``` shell
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data config/data_custom.yamll --img 640 640 --cfg config/yolov7.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml

```

## Inference

On video:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
```

On image:
``` shell
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source image.jpg
```

## Export

**Pytorch to ONNX with NMS (and inference)** <a href="https://colab.research.google.com/github/WongKinYiu/yolov7/blob/main/tools/YOLOv7onnx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
```shell
python export.py --weights yolov7.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```
## Test:
On image:
``` shell
python layout_analysis.py --model yolov7.onnx --image image.jpg
```
![alt](https://github.com/duyquang392/LayoutAnalysis_YOLov7/blob/main/doc/demo.jpg)
