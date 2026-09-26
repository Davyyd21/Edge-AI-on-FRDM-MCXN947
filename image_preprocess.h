#ifndef IMAGE_PREPROCESS_H
#define IMAGE_PREPROCESS_H

#include <stdint.h>

#define MODEL_INPUT_WIDTH    96U
#define MODEL_INPUT_HEIGHT   96U
#define MODEL_INPUT_CHANNELS 3U
#define MODEL_INPUT_IMAGE_SIZE (MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * MODEL_INPUT_CHANNELS)

/*
 * Converts a single RGB565 pixel into separate 8-bit red, green, and blue channels.
 */
void RGB565_To_RGB888(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b);

/*
 * Resizes a camera frame to the model input dimensions and converts
 * the RGB565 image into the int8 format expected by the neural network.
 */
void RGB565_ResizeToINT8(const uint16_t *src, int8_t *dst);

#endif