#include "camera_stripe_capture.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "fsl_common.h"
#include "fsl_debug_console.h"
#include "fsl_inputmux.h"
#include "fsl_smartdma.h"
#include "fsl_smartdma_prv.h"

/*
 * SmartDMA accesses this structure directly, so keeping it 32-byte aligned
 * ensures that the hardware can safely access it.
 */
static smartdma_camera_param_t s_smartdma_param __attribute__((aligned(32)));

/*
 * Full QVGA RGB565 camera frame:
 * 320 x 240 pixels x 2 bytes = 153600 bytes.
 *
 * This buffer stays in normal SRAM because it is too large for the 96 KB SRAMX.
 */
static volatile uint16_t s_camera_frame[CAMERA_FRAME_PIXELS] __attribute__((aligned(32)));

/*
 * SmartDMA executes its camera capture code using this dedicated stack.
 * It is also aligned because the SmartDMA hardware accesses it directly.
 */
static uint32_t s_smartdma_stack[64] __attribute__((aligned(32)));

/*
 * The SmartDMA interrupt only sets this flag.
 * The actual frame processing is deliberately kept outside the ISR.
 */
static volatile bool s_frame_ready = false;

static void CAMERA_SmartDmaCallback(void *param)
{
    (void)param;

    s_frame_ready = true;
}

void CAMERA_DebugPrintMemory(void)
{
    uintptr_t start = (uintptr_t)&s_camera_frame[0];
    uintptr_t end = start + sizeof(s_camera_frame);

    PRINTF("\r\nCAMERA MEMORY:\r\n");
    PRINTF("  frame start      = 0x%08X\r\n", (unsigned int)start);
    PRINTF("  frame end        = 0x%08X\r\n", (unsigned int)end);
    PRINTF("  frame size       = %u bytes\r\n", (unsigned int)sizeof(s_camera_frame));
    PRINTF("  SmartDMA stack   = 0x%08X\r\n", (unsigned int)(uintptr_t)s_smartdma_stack);
    PRINTF("  SmartDMA params  = 0x%08X\r\n", (unsigned int)(uintptr_t)&s_smartdma_param);
    PRINTF("\r\n");
}

void CAMERA_StripeCaptureInit(void)
{
    s_frame_ready = false;

    /*
     * Start from clean memory so that stale data cannot be mistaken for
     * a newly captured frame or valid SmartDMA state.
     */
    memset(&s_smartdma_param, 0, sizeof(s_smartdma_param));
    memset((void *)s_camera_frame, 0, sizeof(s_camera_frame));
    memset(s_smartdma_stack, 0, sizeof(s_smartdma_stack));

    /*
     * Route the camera timing signals into SmartDMA.
     * Without these INPUTMUX connections, SmartDMA cannot see the
     * PCLK, HREF, and VSYNC signals coming from the OV7670.
     */
    INPUTMUX_Init(INPUTMUX0);
    INPUTMUX_AttachSignal(INPUTMUX0, 0U, kINPUTMUX_GpioPort0Pin4ToSmartDma);
    INPUTMUX_AttachSignal(INPUTMUX0, 1U, kINPUTMUX_GpioPort0Pin11ToSmartDma);
    INPUTMUX_AttachSignal(INPUTMUX0, 2U, kINPUTMUX_GpioPort0Pin5ToSmartDma);

    /*
     * Load the camera capture firmware into SmartDMA and register the
     * callback that will be called when a complete frame has been captured.
     */
    SMARTDMA_InitWithoutFirmware();
    SMARTDMA_InstallFirmware(SMARTDMA_CAMERA_MEM_ADDR, s_smartdmaCameraFirmware, SMARTDMA_CAMERA_FIRMWARE_SIZE);
    SMARTDMA_InstallCallback(CAMERA_SmartDmaCallback, NULL);

    NVIC_SetPriority(SMARTDMA_IRQn, 3U);
    NVIC_EnableIRQ(SMARTDMA_IRQn);

    /*
     * We capture one complete frame into a single buffer.
     * No ping-pong buffer or stripe index is needed for this setup.
     */
    s_smartdma_param.smartdma_stack = s_smartdma_stack;
    s_smartdma_param.p_buffer = (uint32_t *)s_camera_frame;
    s_smartdma_param.p_stripe_index = NULL;
    s_smartdma_param.p_buffer_ping_pong = NULL;

    CAMERA_DebugPrintMemory();

    PRINTF("SmartDMA parameters:\r\n");
    PRINTF("  p_buffer          = 0x%08X\r\n", (unsigned int)(uintptr_t)s_smartdma_param.p_buffer);
    PRINTF("  smartdma_stack    = 0x%08X\r\n", (unsigned int)(uintptr_t)s_smartdma_param.smartdma_stack);

    /*
     * Start the SmartDMA camera firmware in the full-frame QVGA capture mode.
     */
    SMARTDMA_Boot(kSMARTDMA_CameraWholeFrameQVGA, &s_smartdma_param, 0x2U);
}

bool CAMERA_WaitForStripe(uint32_t *stripe_index, const uint16_t **pixels)
{
    static uint32_t frame_count = 0U;

    /*
     * Wait until the SmartDMA interrupt tells us that a complete frame
     * is ready. The timeout prevents the application from being stuck
     * forever if the camera stops producing frames.
     */
    uint32_t timeout = 50000000U;

    while (!s_frame_ready && timeout > 0U)
    {
        timeout--;
    }

    if (!s_frame_ready)
    {
        PRINTF("ERROR: Timeout waiting for SmartDMA frame\r\n");
        return false;
    }

    frame_count++;

    /*
     * Print a couple of pixels as a quick sanity check that SmartDMA
     * is actually writing camera data into the frame buffer.
     */
    PRINTF("FRAME %u: pixel[0] = 0x%04X, pixel[1000] = 0x%04X\r\n",
           (unsigned int)frame_count,
           (unsigned int)s_camera_frame[0],
           (unsigned int)s_camera_frame[1000]);

    if (stripe_index != NULL)
    {
        *stripe_index = 0U;
    }

    /*
     * The frame is already stored in the shared camera buffer, so we
     * simply return its address instead of copying 150 KB of data.
     */
    if (pixels != NULL)
    {
        *pixels = (const uint16_t *)s_camera_frame;
    }

    return true;
}

void CAMERA_ReleaseStripe(void)
{
    /*
     * The current frame has been consumed. Clearing the flag allows
     * SmartDMA to signal the next completed frame.
     */
    s_frame_ready = false;
}