import torch
import torch.nn as nn
import argparse
import models
from models.yolo import *
from utils.torch_utils import intersect_dicts

def parse():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cfg",type=str,default="models/yolov4-p5.yaml")
    arg.add_argument("--checkpoint",type=str,default="yolov4-p5.pt")
    arg.add_argument("--h",type=int,default=896)
    arg.add_argument("--w",type=int,default=896)
    arg.add_argument("--output",type=str,default="yolov4-p5.onnx")
    arg.add_argument("--nc",type=int,default=80)
    return arg.parse_args()

if __name__ == "__main__":
    opt = parse()

    model = Model(opt.cfg, ch=3,nc=opt.nc)
    model.eval()
    model.model[-1].export = True
    exclude = ['anchor'] if opt.cfg else []  # exclude keys
    ckpt = torch.load(opt.checkpoint)  # load checkpoint
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    x = torch.randn(1,3,opt.h,opt.w)
    numup = 0
    for i,m in enumerate(model.model):
        if isinstance(m,nn.Upsample):
            numup += 1

    scaledh = opt.h/(2**(2+numup))
    scaledw = opt.w/(2**(2+numup))
    for i,m in enumerate(model.model):
        if isinstance(m,nn.Upsample):
            model.model[i] = nn.Upsample(size=(scaledh,scaledw))
            scaledw *= 2
            scaledh *= 2    

    torch.onnx.export(
        model,x,opt.output,verbose=True,input_names=['input'],output_names=['conf','cls','bbox'],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
