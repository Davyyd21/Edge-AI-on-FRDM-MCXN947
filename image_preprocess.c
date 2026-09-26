#include "image_preprocess.h"

#include <stddef.h>
#include <stdint.h>

#include "camera_stripe_capture.h"

#define INPUT_ZERO_POINT (-14)

#define R_INV_STD_SCALE_Q16 61325
#define G_INV_STD_SCALE_Q16 61397
#define B_INV_STD_SCALE_Q16 60184

#define R_MEAN_Q16 ((int32_t)(103.53f * 65536.0f))
#define G_MEAN_Q16 ((int32_t)(116.28f * 65536.0f))
#define B_MEAN_Q16 ((int32_t)(123.675f * 65536.0f))

/*
 * Converts one normalized RGB channel value into the int8 quantized
 * representation expected by the neural network input tensor
 */
static int8_t QuantizeChannel(uint8_t value, int32_t mean_q16, int32_t inv_std_scale_q16)
{
    int32_t value_q16 = (int32_t)value << 16;// Convert to Q16 format-Q16 is a fixed-point representation where the integer part is stored in the upper 16 bits and the fractional part is stored in the lower 16 bits. This allows for precise representation of decimal values using integers.
    int32_t diff_q16 = value_q16 - mean_q16;// doing this for the purpose of normalization, which is a common preprocessing step in machine learning. Normalization helps to center the data around zero and scale it to a standard range, which can improve the performance and convergence of the neural network during training and inference.
    int64_t product = (int64_t)diff_q16 * inv_std_scale_q16;// The product is calculated in 64-bit integer space to prevent overflow during the multiplication of two 32-bit integers. This is important because the result of multiplying two 32-bit integers can exceed the maximum value that can be represented by a 32-bit integer, leading to incorrect results. By using a 64-bit integer for the product, we ensure that the full range of possible values can be accurately represented without overflow.

    int32_t quantized;

    if (product >= 0)
    {
        quantized = (int32_t)((product + 32768LL) >> 16U);// The addition of 32768 before the right shift is a technique known as rounding. When you right-shift a number, you effectively divide it by a power of two (in this case, 65536). However, this operation truncates any fractional part, which can lead to a loss of precision. By adding 32768 (which is half of 65536) before the shift, we are effectively rounding the result to the nearest integer instead of simply truncating it. This helps to maintain better accuracy in the quantization process.
    }
    else
    {
        quantized = (int32_t)((product - 32768LL) >> 16U);
    }

    quantized += INPUT_ZERO_POINT;

    /*
     * Keep the final value inside the range supported by an int8 tensor
     */
    if (quantized > 127)
    {
        quantized = 127;
    }
    else if (quantized < -128)
    {
        quantized = -128;
    }

    return (int8_t)quantized;
}

/*
 * RGB565 stores red, green, and blue with fewer bits than standard RGB888.
 * This expands each channel back to 8 bits so the camera image can be
 * normalized and prepared for the neural network.
 */
void RGB565_To_RGB888(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b)
{
    if ((r == NULL) || (g == NULL) || (b == NULL))
    {
        return;
    }
    // Extract the 5-bit red, 6-bit green, and 5-bit blue components from the RGB565 pixel value. The RGB565 format uses 5 bits for red, 6 bits for green, and 5 bits for blue, packed into a single 16-bit integer. The bitwise operations are used to isolate each color component from the packed pixel value.
    uint8_t r5 = (uint8_t)((pixel >> 11U) & 0x1FU);
    uint8_t g6 = (uint8_t)((pixel >> 5U) & 0x3FU);
    uint8_t b5 = (uint8_t)(pixel & 0x1FU);
    // Convert the 5-bit and 6-bit color components to 8-bit values. The conversion is done by shifting the bits to the left and filling in the lower bits with a portion of the original value. This helps to approximate the original color intensity when expanding from a lower bit depth to a higher bit depth.
    *r = (uint8_t)((r5 << 3U) | (r5 >> 2U));
    *g = (uint8_t)((g6 << 2U) | (g6 >> 4U));
    *b = (uint8_t)((b5 << 3U) | (b5 >> 2U));
}

/*
 * Downsamples the 320x240 camera frame to the model's 96x96 input size,
 * converts RGB565 pixels to RGB888, and stores the result as an int8 tensor.
 */
void RGB565_ResizeToINT8(const uint16_t *src, int8_t *dst)
{
    if ((src == NULL) || (dst == NULL))
    {
        return;
    }

    for (uint32_t y = 0U; y < MODEL_INPUT_HEIGHT; y++)
    {
        // Calculate the corresponding y-coordinate in the source image based on the current y-coordinate in the destination image. The formula scales the y-coordinate from the destination image's height to the source image's height, effectively mapping each row of the destination image to a row in the source image.
        uint32_t src_y = (y * CAMERA_HEIGHT) / MODEL_INPUT_HEIGHT;

        if (src_y >= CAMERA_HEIGHT)
        {
            src_y = CAMERA_HEIGHT - 1U;
        }

        for (uint32_t x = 0U; x < MODEL_INPUT_WIDTH; x++)
        {
            uint32_t src_x = (x * CAMERA_WIDTH) / MODEL_INPUT_WIDTH;

            if (src_x >= CAMERA_WIDTH)
            {
                src_x = CAMERA_WIDTH - 1U;
            }
            // Retrieve the RGB565 pixel value from the source image at the calculated coordinates. The source image is stored as a 1D array, so the index is calculated by multiplying the y-coordinate by the width of the image and adding the x-coordinate. This gives the correct position in the array for the desired pixel.
            uint16_t pixel = src[src_y * CAMERA_WIDTH + src_x];

            uint8_t r;
            uint8_t g;
            uint8_t b;

            RGB565_To_RGB888(pixel, &r, &g, &b);

            uint32_t dst_index = (y * MODEL_INPUT_WIDTH + x) * MODEL_INPUT_CHANNELS;

            dst[dst_index] = QuantizeChannel(r, R_MEAN_Q16, R_INV_STD_SCALE_Q16);
            dst[dst_index + 1U] = QuantizeChannel(g, G_MEAN_Q16, G_INV_STD_SCALE_Q16);
            dst[dst_index + 2U] = QuantizeChannel(b, B_MEAN_Q16, B_INV_STD_SCALE_Q16);
        }
    }
}