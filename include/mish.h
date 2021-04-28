#ifndef _MISH_H_
#define _MISH_H_

#include "NvInfer.h"
#include <string>
#include "NvInferPlugin.h"
#include "cuda_fp16.h"
#include <vector>

using namespace nvinfer1;

int computeMish(cudaStream_t stream, int n, const float* input, float* output);
int computeMish(cudaStream_t stream, int n, const half* input, half* output);

bool initMishPlugin();

class MishPlugin final : public IPluginV2DynamicExt
{
private:
    const char* mPluginNamespace;
    std::string mNamespace;
    bool mInitialized;
    DataType mtype;
public:
    MishPlugin (/* args */);  //初始化 clone
    MishPlugin (void const *seralData, size_t serialLength);  //反序列化
    ~MishPlugin ();
    int getNbOutputs() const override;
    DimsExprs getOutputDimensions(int outputIndex,const DimsExprs* inputs,int nbInputs,IExprBuilder &exprBuilder) override;
    int initialize() override;
    void terminate() override;

    size_t getWorkspaceSize(const PluginTensorDesc *inputs,int nbInputs, const PluginTensorDesc *ouputs,int nbOutputs) const override;
    int enqueue(const PluginTensorDesc* inputDesc,const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void *workspace,
                cudaStream_t stream) override;

    size_t getSerializationSize() const override;
    void serialize(void *buffer) const override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut,int nbInputs,int nbOutputs) override;
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    void destroy() override;
    IPluginV2DynamicExt *clone() const override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char *getPluginNamespace() const override;
    DataType getOutputDataType(int index,const DataType *inputTypes, int nbInputs) const override;
    void attachToContext(cudnnContext *cudnn, cublasContext *blas, IGpuAllocator *allocator) override;
    void detachFromContext() override;
    void configurePlugin(const DynamicPluginTensorDesc *in,int nbInputs,
                        const DynamicPluginTensorDesc *out, int nbOutputs) override; //在 initial前被调用
};

class MishPluginCreator : public IPluginCreator
{
public:
    MishPluginCreator();
    ~MishPluginCreator() override=default;
    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection*getFieldNames() override;
    IPluginV2DynamicExt *createPlugin(const char *name,const PluginFieldCollection *fc) override;
    IPluginV2DynamicExt *deserializePlugin(const char *name,const void *serialData,size_t serialLength) override;
    void setPluginNamespace(const char *pluginNamespace) override;
    const char *getPluginNamespace() const override;
private:
    static PluginFieldCollection mFc;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};


#endif