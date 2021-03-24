#include "mish.h"
#include <iostream>
#include <assert.h>
#include <cuda_fp16.h>

namespace {
    constexpr const char* MISH_PLUGIN_VERSION{"001"};
    constexpr const char* MISH_PLUGIN_NAME{"Mish_TRT"};
}


MishPlugin::MishPlugin (/* args */){

}
MishPlugin::MishPlugin (void const *seralData, size_t serialLength){

}  //反序列化
MishPlugin::~MishPlugin (){
    terminate();
}
int MishPlugin::getNbOutputs() const{
    return 1;
}

DataType MishPlugin::getOutputDataType(int index,const DataType *inputTypes, int nbInputs) const{
    return inputTypes[0];    
}

size_t MishPlugin::getWorkspaceSize(const PluginTensorDesc *inputs,int nbInputs, const PluginTensorDesc *ouputs,int nbOutputs) const{
    return 0;
}

DimsExprs MishPlugin::getOutputDimensions(int outputIndex,const DimsExprs* inputs,int nbInputs,IExprBuilder &exprBuilder) {
    DimsExprs ret;
    ret.nbDims = 4;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[0].d[1];
    ret.d[2] = inputs[0].d[2];
    ret.d[3] = inputs[0].d[3];

    return ret;
}

int MishPlugin::enqueue(const PluginTensorDesc* inputDesc,const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void *workspace,
                cudaStream_t stream){
    
    mtype = inputDesc[0].type;
    int n = inputDesc[0].dims.d[0]*inputDesc[0].dims.d[1]*inputDesc[0].dims.d[2]*inputDesc[0].dims.d[3];
    //std::cout<<inputDesc[0].dims.d[0]<<inputDesc[0].dims.d[1]<<inputDesc[0].dims.d[2]<<inputDesc[0].dims.d[3]<<std::endl;
    switch (mtype){
        case DataType::kHALF:{
            //std::cout<<"fp16"<<std::endl;
            const half* input_data = static_cast<const half*>(inputs[0]);
            half* out_data= static_cast<half*>(outputs[0]);
            computeMish(stream,n,input_data,out_data);
            break;
        }
        case DataType::kFLOAT:{
            const float* input_data = static_cast<const float*>(inputs[0]);
            float* out_data= static_cast<float*>(outputs[0]);
            computeMish(stream,n,input_data,out_data);
            break;
        }
        default: std::cout<<"error data type"<<std::endl;
    }

    return 0;

}

bool MishPlugin::supportsFormatCombination(int pos, const PluginTensorDesc *inOut,int nbInputs,int nbOutputs){
    return ((inOut[pos].type == nvinfer1::DataType::kFLOAT || inOut[pos].type == nvinfer1::DataType::kHALF)
        && inOut[pos].format == nvinfer1::PluginFormat::kNCHW && inOut[pos].type == inOut[0].type);
}
void MishPlugin::configurePlugin(const DynamicPluginTensorDesc *in,int nbInputs,
                        const DynamicPluginTensorDesc *out, int nbOutputs){
}

void MishPlugin::attachToContext(cudnnContext *cudnn, cublasContext *blas, IGpuAllocator *allocator){}
void MishPlugin::detachFromContext(){}
int MishPlugin::initialize(){return 0;}
void MishPlugin::terminate(){}
size_t MishPlugin::getSerializationSize() const{return 0;}
void MishPlugin::serialize(void *buffer) const { }
void MishPlugin::destroy()
{ 
    delete this;
}
void MishPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* MishPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}
IPluginV2DynamicExt *MishPlugin::clone() const{
    auto plugin = new MishPlugin();
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}
const char* MishPlugin::getPluginType() const
{
    return MISH_PLUGIN_NAME;
}

const char* MishPlugin::getPluginVersion() const
{
    return MISH_PLUGIN_VERSION;
}


PluginFieldCollection MishPluginCreator::mFc{};
std::vector<PluginField>
    MishPluginCreator::mPluginAttributes;


MishPluginCreator::MishPluginCreator(){


}
const char* MishPluginCreator::getPluginName() const
{
    return MISH_PLUGIN_NAME;
}

const char* MishPluginCreator::getPluginVersion() const
{
    return MISH_PLUGIN_VERSION;
}

const PluginFieldCollection* MishPluginCreator::getFieldNames()
{
    return &mFc;
}
IPluginV2DynamicExt* MishPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{

    MishPlugin* obj = new MishPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* MishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    MishPlugin* obj = new MishPlugin{serialData, serialLength}; 
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

void MishPluginCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *MishPluginCreator::getPluginNamespace()
    const {
  return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(MishPluginCreator);

bool initMishPlugin() { return true; }

