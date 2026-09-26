#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "nanodet_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

int MODEL_Init(void);
int MODEL_RunInference(void);
int8_t *MODEL_GetInputData(void);
int8_t *MODEL_GetOutputData(uint32_t index);
uint32_t MODEL_GetInputSize(void);
uint32_t MODEL_GetOutputSize(uint32_t index);
float MODEL_GetOutputScale(uint32_t index);
int32_t MODEL_GetOutputZeroPoint(uint32_t index);
uint32_t MODEL_GetTensorArenaUsed(void);
int MODEL_GetDetections(NanoDetDetection *detections, int max_detections);

#ifdef __cplusplus
}
#endif

#endif