# TensoRT Scaled YOLOv4


<a href="https://996.icu"><img src="https://img.shields.io/badge/link-996.icu-red.svg" alt="996.icu" /></a>


TensorRT for Scaled YOLOv4(yolov4-csp.cfg)

很多人都写过TensorRT版本的yolo了，我也来写一个。

## 测试环境

    ubuntu 18.04 
    jetpack 4.4
    CUDA 10.0
    TensorRT7.1

## 快速开始

### 1、生成onnx模型
	git clone https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp
	git clone https://github.com/talebolano/TensorRT-Scaled-YOLOv4
	cp TensorRT-Scaled-YOLOv4/script/models.py yolov4-csp/models/
	cp TensorRT-Scaled-YOLOv4/script/export.py yolov4-csp/
	下载yolov4-csp.weights
	cd yolov4-csp
	python3 export.py



### 2、编译
 
	cd ../TensorRT-Scaled-YOLOv4
    mkdir build 
    cd build
    cmake ..
    make -j8

### 3、转换onnx模型到trt模型

    ./makeCudaEngine -i ../../yolov4-csp/yolov4-csp.onnx -o yolov4-csp.trt

### 4、测试

    ./inferYoloCuda  -e yolov4-csp.trt -i 你的图片 -show -save

## 速度效果

 Mode | GPU | inference time | Ap
---:|:---:|:---:|:---:
FP16 | V100 | 12ms | -
FP16 | xavier  | 35ms | -

