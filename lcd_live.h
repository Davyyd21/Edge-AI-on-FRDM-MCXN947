#ifndef LCD_LIVE_H
#define LCD_LIVE_H

#include <stdint.h>
#include "nanodet_decoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void LCD_LiveInit(void);


void LCD_LiveShowCameraStripe(
    uint32_t stripe_index,
    const uint16_t *pixels
);

void LCD_LiveDrawDetectionsOnStripe(
    uint16_t *pixels,
    uint32_t stripe_index,
    const NanoDetDetection *detections,
    uint32_t count
);

void LCD_LiveDrawBoundingBox(
    int x0,
    int y0,
    int x1,
    int y1,
    uint16_t color
);

void LCD_LiveShowCameraFrame(const uint16_t *pixels);

void LCD_LiveDrawText(
    int x,
    int y,
    const char *text,
    uint16_t color
);

void LCD_LiveDrawDetectionLabel(
    int x,
    int y,
    const char *class_name,
    int score_x1000,
    uint16_t color
);

void LCD_DebugPrintState(void);

#ifdef __cplusplus
}

#endif

#endif