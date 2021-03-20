import torch
import models
from models.models import *


cfg = "models/yolov4-csp.cfg"
checkpoint = "yolov4-csp.weights"


model = Darknet(cfg,img_size=(640, 640))
load_darknet_weights(model, checkpoint)

x = torch.randn(1,3,640,640)

torch.onnx.export(
    model,x,"yolov4-csp.onnx",verbose=True,input_names=['input'],output_names=['conf','cls','bbox']
)
