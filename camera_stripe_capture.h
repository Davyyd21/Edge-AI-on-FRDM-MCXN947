#ifndef CAMERA_STRIPE_CAPTURE_H
#define CAMERA_STRIPE_CAPTURE_H

#include <stdbool.h>
#include <stdint.h>

#define CAMERA_WIDTH          320U
#define CAMERA_HEIGHT         240U
#define CAMERA_STRIPE_H       15U
#define CAMERA_STRIPE_HEIGHT  CAMERA_STRIPE_H
#define CAMERA_STRIPE_COUNT   (CAMERA_HEIGHT / CAMERA_STRIPE_H)
#define CAMERA_STRIPE_PIXELS  (CAMERA_WIDTH * CAMERA_STRIPE_H)
#define CAMERA_FRAME_PIXELS   (CAMERA_WIDTH * CAMERA_HEIGHT)

void CAMERA_StripeCaptureInit(void);
bool CAMERA_WaitForStripe(uint32_t *stripe_index,const uint16_t **pixels);
void CAMERA_ReleaseStripe(void);
void CAMERA_DebugPrintMemory(void);

#endif
