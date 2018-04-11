# ASTER: *A*ttentional *S*cene *TE*xt *R*ecognizer with Flexible Rectification 

ASTER is an accurate scene text recognizer with flexible rectification mechanism.

The implementation of ASTER reuses code from [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).

## Prerequisites

Install [Protocol Buffers](https://github.com/google/protobuf) (version>=2.6)

In Ubuntu 16.04
```
sudo apt install cmake libcupti-dev
pip3 install --user tensorflow-gpu protobuf tqdm numpy editdistance
```

NOTE: ASTER is developed and tested with TensorFlow r1.4.

## Installation
  1. Go to `c_ops/` and run `build.sh` to build custom operators
  2. Execute `protoc aster/protos/*.proto --python_out=.` to build protobuf files

## Training and on-the-fly evaluation
To run the example training, execute

```
python3 aster/train.py \
  --exp_dir experiments/aster \
  --num_clones 2
```

Change the configuration in `experiments/aster/trainval.prototxt` to configure your own training process.

While training, you can run a separate program that repeatedly evaluates the checkpoints produced by the training every few minutes.

```
python3 aster/eval.py \
   --exp_dir experiments/aster
```

Evaluation configuration is also in `trainval.prototxt`.
