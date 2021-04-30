#ifndef __TN_UTIL_H_
#define __TN_UTIL_H_

#include "NvInfer.h"
#include <iostream>
#include "assert.h"

#ifndef CHECK
#define CHECK(status)                                                               \                                           
    do                                                                              \                                          
    {                                                                               \                                        
        auto ret = (status);                                                        \                                        
        if (ret != 0)                                                               \                                       
        {                                                                           \                                      
            std::cerr << "Cuda failure: " << ret << std::endl;                      \                                     
            abort();                                                                \                                    
        }                                                                           \                                   
                                                                                    \
    } while (0)                                                                     
#endif

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity=Severity::kWARNING):reportSeverity(severity)
    {}
    void log(Severity severity,const char*msg) override
    {
        if(severity>reportSeverity){
            return;
        }
        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr<<"INTERNAL_ERROR ";break;
        case Severity::kERROR: std::cerr<<"ERROR" ;break;
        case Severity::kWARNING: std::cerr<<"WARNING ";break;
        case Severity::kINFO: std::cerr<<"INFO ";break;
        
        default:std::cerr<<"UNKNOWN"<<std::endl;
            break;
        }
        std::cerr<<msg<<std::endl;
    }
    Severity reportSeverity;

    nvinfer1::ILogger& getTRTLogger()
    {
        return *this;
    }

    Severity getReportableSeverity() const
    {
        return reportSeverity;
    }

};

#endif