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
	git clone --branch yolov4-csp https://github.com/WongKinYiu/ScaledYOLOv4
	git clone https://github.com/talebolano/TensorRT-Scaled-YOLOv4
	cp TensorRT-Scaled-YOLOv4/script/* ScaledYOLOv4/
	下载yolov4-csp.weights到ScaledYOLOv4/
	cd ScaledYOLOv4
	python3 export.py



### 2、编译
 
	cd ../TensorRT-Scaled-YOLOv4
    mkdir build 
    cd build
    cmake ..
    make -j8

### 3、转换onnx模型到trt模型

    ./makeCudaEngine -i ../../ScaledYOLOv4/yolov4-csp.onnx -o yolov4-csp.trt

### 4、测试

    ./inferYoloCuda  -e yolov4-csp.trt -i 你的图片 -show -save

## 速度效果

 Mode | GPU | inference time | Ap
---:|:---:|:---:|:---:
FP16 | V100 | 12ms | -
FP16 | xavier  | 35ms | -

## 使用mish插件层

1、下载TensorRT的开源版，在builtin_op_importers.cpp中注册Mish插件。将以下代码粘贴到builtin_op_importers.cpp底部

	DEFINE_BUILTIN_OP_IMPORTER(Mish)
	{
    	ASSERT(inputs.at(0).is_tensor(),  nvonnxparser::ErrorCode::kUNSUPPORTED_NODE); // input
    	std::vector<nvinfer1::ITensor*> tensors;
    	nvinfer1::ITensor* input = &convertToTensor(inputs.at(0),ctx);
    	tensors.push_back(input);
    
    	const std::string pluginName = "Mish_TRT";
    	const std::string pluginVersion = "001";
    	std::vector<nvinfer1::PluginField> f;

    	const auto mPluginRegistry = getPluginRegistry();
    	const auto pluginCreator
    	    = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "");
    	nvinfer1::PluginFieldCollection fc;
    	fc.nbFields = f.size();
    	fc.fields = f.data();
    	nvinfer1::IPluginV2* plugin = pluginCreator->createPlugin(node.name().c_str(), &fc);

    	ASSERT(plugin != nullptr && "Mish plugin was not found in the plugin registry!",
        ErrorCode::kUNSUPPORTED_NODE);
    	nvinfer1::IPluginV2Layer* layer = ctx->network()->addPluginV2(tensors.data(), tensors.size(), *plugin);
    	RETURN_ALL_OUTPUTS(layer);
	}

2、在ScaledYOLOv4/models/models.py中使用MishImplementation()替代Mish()，将

	modules.add_module('activation', Mish())
替换为

	modules.add_module('activation', MishImplementation())

3、生成onnx模型并转换为trt模型

	python3 export.py
	./makeCudaEngine -i ../../ScaledYOLOv4/yolov4-csp.onnx -o yolov4-csp.trt

4、测试

	./inferYoloCuda  -e yolov4-csp.trt -i 你的图片 -show -save

### 注意：在xavier中使用mish插件要比不使用慢10ms以上。