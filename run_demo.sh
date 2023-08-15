#!/bin/bash

# python demo/demo.py --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml --input-folder "datasets/pickup_examples/cassie" --threshold 0.05 --display MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth OUTPUT_DIR output/demo/DLA34_nms_0.5_0.05/cassie

python demo/demo.py --config-file cubercnn://omni3d/cubercnn_Res34_FPN.yaml --input-folder "datasets/pickup_examples/cassie" --threshold 0.01 --display MODEL.WEIGHTS download_weights/cubercnn_Res34_FPN.pth OUTPUT_DIR output/demo/ResNet34_nms_0.1_0.01/cassie
