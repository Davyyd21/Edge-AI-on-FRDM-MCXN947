#ifndef NANODET_DECODER_H
#define NANODET_DECODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NANODET_NUM_CLASSES   3
#define NANODET_NUM_POINTS    189
#define NANODET_OUTPUT_VALUES 35
#define NANODET_MAX_DETECTIONS 20

typedef struct
{
    float x1;
    float y1;
    float x2;
    float y2;

    float score;
    int class_id;
} NanoDetDetection;

int NanoDet_Decode(const int8_t *output,float scale,int32_t zero_point,NanoDetDetection *detections,int max_detections);

#ifdef __cplusplus
}
#endif

#endif