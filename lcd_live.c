#include "lcd_live.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "app.h"

#include "fsl_common.h"
#include "fsl_debug_console.h"
#include "fsl_flexio_mculcd.h"
#include "fsl_gpio.h"
#include "fsl_st7796s.h"

#include "FreeRTOS.h"
#include "task.h"

#include "camera_stripe_capture.h"

#define LCD_LOGICAL_WIDTH  320U
#define LCD_LOGICAL_HEIGHT 480U

#define LCD_CAMERA_HEIGHT 240U

/*
 * FlexIO is used as the parallel interface between the MCU and the ST7796S.
 * These settings describe the exact pins, shifters, and timer used by the LCD bus.
 */
static FLEXIO_MCULCD_Type s_flexio_lcd =
{
    .flexioBase = DEMO_FLEXIO,
    .busType = kFLEXIO_MCULCD_8080,
    .dataPinStartIndex = DEMO_FLEXIO_DATA_PIN_START,
    .ENWRPinIndex = DEMO_FLEXIO_WR_PIN,
    .RDPinIndex = DEMO_FLEXIO_RD_PIN,
    .txShifterStartIndex = DEMO_FLEXIO_TX_START_SHIFTER,
    .txShifterEndIndex = DEMO_FLEXIO_TX_END_SHIFTER,
    .rxShifterStartIndex = DEMO_FLEXIO_RX_START_SHIFTER,
    .rxShifterEndIndex = DEMO_FLEXIO_RX_END_SHIFTER,
    .timerIndex = DEMO_FLEXIO_TIMER,
    .setCSPin = NULL,
    .setRSPin = NULL,
    .setRDWRPin = NULL
};

static flexio_mculcd_handle_t s_lcd_xfer_handle;
static TaskHandle_t s_lcd_wait_task = NULL;
static volatile status_t s_lcd_xfer_status = kStatus_Success;
static bool s_lcd_async_enabled = false;

static void LCD_SetCSPin(bool set)
{
    GPIO_PinWrite(DEMO_LCD_CS_GPIO, DEMO_LCD_CS_PIN, set ? 1U : 0U);
}

static void LCD_SetRSPin(bool set)
{
    GPIO_PinWrite(DEMO_LCD_RS_GPIO, DEMO_LCD_RS_PIN, set ? 1U : 0U);
}

static void LCD_SetResetPin(bool set)
{
    GPIO_PinWrite(DEMO_LCD_RST_GPIO, DEMO_LCD_RST_PIN, set ? 1U : 0U);
}

/*
 * This callback is executed when the FlexIO IRQ transfer has finished.
 * The CameraTask is waiting for this notification instead of continuously
 * polling the FlexIO peripheral, so the CPU can execute the inference task
 * while the LCD transfer is in progress.
 */
static void LCD_TransferCallback(FLEXIO_MCULCD_Type *base,
                                 flexio_mculcd_handle_t *handle,
                                 status_t status,
                                 void *userData)
{
    (void)base;
    (void)handle;
    (void)userData;

    s_lcd_xfer_status = status;

    if (s_lcd_wait_task != NULL)
    {
        BaseType_t higher_priority_task_woken = pdFALSE;

        vTaskNotifyGiveFromISR(s_lcd_wait_task, &higher_priority_task_woken);
        portYIELD_FROM_ISR(higher_priority_task_woken);
    }
}

/*
 * Each command or data transfer is wrapped in a FlexIO transaction so the
 * LCD sees a clean start-to-finish communication sequence.
 */
static status_t LCD_WriteCommand(void *dbi_xfer_handle, uint32_t command)
{
    FLEXIO_MCULCD_Type *flexio_lcd =
        (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    FLEXIO_MCULCD_StartTransfer(flexio_lcd);
    FLEXIO_MCULCD_WriteCommandBlocking(flexio_lcd, command);
    FLEXIO_MCULCD_StopTransfer(flexio_lcd);

    return kStatus_Success;
}

static status_t LCD_WriteData(void *dbi_xfer_handle,
                              void *data,
                              uint32_t length_bytes)
{
    FLEXIO_MCULCD_Type *flexio_lcd =
        (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    FLEXIO_MCULCD_StartTransfer(flexio_lcd);
    FLEXIO_MCULCD_WriteDataArrayBlocking(
        flexio_lcd,
        data,
        length_bytes);
    FLEXIO_MCULCD_StopTransfer(flexio_lcd);

    return kStatus_Success;
}

static status_t LCD_WriteMemory(void *dbi_xfer_handle,
                                uint32_t command,
                                const void *data,
                                uint32_t length_bytes)
{
    FLEXIO_MCULCD_Type *flexio_lcd =
        (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    /*
     * LCD initialization and clearing happen before the FreeRTOS scheduler
     * starts, so the original blocking transfer is kept for that phase.
     */
    if (!s_lcd_async_enabled)
    {
        FLEXIO_MCULCD_StartTransfer(flexio_lcd);
        FLEXIO_MCULCD_WriteCommandBlocking(flexio_lcd, command);
        FLEXIO_MCULCD_WriteDataArrayBlocking(
            flexio_lcd,
            data,
            length_bytes);
        FLEXIO_MCULCD_StopTransfer(flexio_lcd);

        return kStatus_Success;
    }

    /*
     * The ST7796S driver expects this callback to return only after its memory
     * write has completed. We therefore use the non-blocking FlexIO transfer
     * internally, then sleep the current task until the IRQ callback signals
     * that the transfer is finished.
     */
    if (s_lcd_wait_task == NULL)
    {
        s_lcd_wait_task = xTaskGetCurrentTaskHandle();
    }
    //notifications are used,to explain it simply for the LCD transfer, we use a notification to tell the task that the transfer is done, so we can wait for the notification instead of busy waiting, which would waste CPU cycles and power. The notification is sent from the LCD transfer callback when the transfer is done, so we can wait for it here and continue execution when we receive it.
    (void)ulTaskNotifyTake(pdTRUE, 0U);

    s_lcd_xfer_status = kStatus_Success;

    flexio_mculcd_transfer_t xfer =
    {
        .command = command,
        .dataAddrOrSameValue = (uint32_t)(uintptr_t)data,
        .dataSize = length_bytes,
        .mode = kFLEXIO_MCULCD_WriteArray,
        .dataOnly = false
    };

    status_t status = FLEXIO_MCULCD_TransferNonBlocking(
        flexio_lcd,
        &s_lcd_xfer_handle,
        &xfer);

    if (status != kStatus_Success)
    {
        PRINTF(
            "LCD ERROR: non-blocking transfer start failed: %d\r\n",
            (int)status);

        return status;
    }
    //here the notiification is used to wait for the transfer to complete, so we can return from this function only when the transfer is done, as expected by the ST7796S driver. The notification is sent from the LCD transfer callback when the transfer is done, so we can wait for it here and continue execution when we receive it.
    (void)ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    return s_lcd_xfer_status;
}

/*
 * The generic ST7796S driver uses these callbacks to communicate through
 * our FlexIO implementation instead of accessing the LCD bus directly.
 */
static dbi_xfer_ops_t s_flexio_dbi_ops =
{
    .writeCommand = LCD_WriteCommand,
    .writeData = LCD_WriteData,
    .writeMemory = LCD_WriteMemory,
    .readMemory = NULL,
    .setMemoryDoneCallback = NULL
};

static st7796s_handle_t s_lcd_handle;

/*
 * One 320x15 RGB565 stripe uses 9600 bytes.
 * Keeping this reusable buffer in SRAMX avoids putting it in the main data area.
 */
static uint16_t s_black_stripe[CAMERA_STRIPE_PIXELS]
    __attribute__((section(".sramx"), aligned(32)));

static void LCD_HardReset(void)
{
    LCD_SetResetPin(false);
    SDK_DelayAtLeastUs(20000U, SystemCoreClock);

    LCD_SetResetPin(true);
    SDK_DelayAtLeastUs(120000U, SystemCoreClock);
}

static void LCD_InitFlexIO(void)
{
    const gpio_pin_config_t gpio_output_high =
    {
        .pinDirection = kGPIO_DigitalOutput,
        .outputLogic = 1U
    };

    GPIO_PinInit(
        DEMO_LCD_RST_GPIO,
        DEMO_LCD_RST_PIN,
        &gpio_output_high);

    GPIO_PinInit(
        DEMO_LCD_CS_GPIO,
        DEMO_LCD_CS_PIN,
        &gpio_output_high);

    GPIO_PinInit(
        DEMO_LCD_RS_GPIO,
        DEMO_LCD_RS_PIN,
        &gpio_output_high);

    s_flexio_lcd.setCSPin = LCD_SetCSPin;
    s_flexio_lcd.setRSPin = LCD_SetRSPin;

    PRINTF("LCD: initializing FlexIO...\r\n");

    flexio_mculcd_config_t config;

    FLEXIO_MCULCD_GetDefaultConfig(&config);

    config.baudRate_Bps = DEMO_FLEXIO_BAUDRATE_BPS;

    status_t status = FLEXIO_MCULCD_Init(
        &s_flexio_lcd,
        &config,
        DEMO_FLEXIO_CLOCK_FREQ);

    if (status != kStatus_Success)
    {
        PRINTF(
            "LCD ERROR: FlexIO init failed: %d\r\n",
            (int)status);

        while (1)
        {
        }
    }

    /*
     * FreeRTOS FromISR APIs may only be called from interrupts whose
     * priority is at or below configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY.
     *
     * The FlexIO LCD transfer completion callback uses
     * vTaskNotifyGiveFromISR(), so the FlexIO IRQ must explicitly use a
     * FreeRTOS-compatible priority instead of relying on the reset/default
     * NVIC priority.
     */
    NVIC_SetPriority(
        FLEXIO_IRQn,
        configLIBRARY_MAX_SYSCALL_INTERRUPT_PRIORITY);

    /*
     * Create the transactional FlexIO handle after the peripheral and its
     * interrupt priority have been configured. The SDK enables the FlexIO
     * IRQ inside this function.
     */
    status = FLEXIO_MCULCD_TransferCreateHandle(
        &s_flexio_lcd,
        &s_lcd_xfer_handle,
        LCD_TransferCallback,
        NULL);

    if (status != kStatus_Success)
    {
        PRINTF(
            "LCD ERROR: FlexIO transfer handle init failed: %d\r\n",
            (int)status);

        while (1)
        {
        }
    }
}

static void LCD_InitPanel(void)
{
    /*
     * The physical panel orientation requires the driver to rotate, invert,
     * and flip the displayed image so it appears correctly on the board.
     */
    const st7796s_config_t config =
    {
        .driverPreset = kST7796S_DriverPresetLCDPARS035,
        .pixelFormat = kST7796S_PixelFormatRGB565,
        .orientationMode = kST7796S_Orientation180,
        .teConfig = kST7796S_TEDisabled,
        .invertDisplay = true,
        .flipDisplay = true,
        .bgrFilter = false
    };

    LCD_HardReset();

    PRINTF("LCD: initializing ST7796S...\r\n");

    status_t status = ST7796S_Init(
        &s_lcd_handle,
        &config,
        &s_flexio_dbi_ops,
        &s_flexio_lcd);

    if (status != kStatus_Success)
    {
        PRINTF(
            "LCD ERROR: ST7796S init failed: %d\r\n",
            (int)status);

        while (1)
        {
        }
    }

    SDK_DelayAtLeastUs(20000U, SystemCoreClock);

    ST7796S_EnableDisplay(
        &s_lcd_handle,
        true);

    SDK_DelayAtLeastUs(20000U, SystemCoreClock);
}

static void LCD_Clear(void)
{
    memset(
        s_black_stripe,
        0,
        sizeof(s_black_stripe));

    PRINTF("LCD: clearing screen...\r\n");

    /*
     * The LCD is cleared stripe by stripe so the same small SRAMX buffer
     * can be reused instead of allocating a full 320x480 frame buffer.
     */
    for (uint32_t stripe = 0U;
         stripe < (LCD_LOGICAL_HEIGHT / CAMERA_STRIPE_H);
         stripe++)
    {
        uint16_t start_y =
            (uint16_t)(stripe * CAMERA_STRIPE_H);

        uint16_t end_y =
            (uint16_t)(start_y + CAMERA_STRIPE_H - 1U);

        ST7796S_SelectArea(
            &s_lcd_handle,
            0U,
            start_y,
            LCD_LOGICAL_WIDTH - 1U,
            end_y);

        ST7796S_WritePixels(
            &s_lcd_handle,
            s_black_stripe,
            CAMERA_STRIPE_PIXELS);
    }
}

void LCD_LiveInit(void)
{
    PRINTF("LCD: initialization beginning...\r\n");

    LCD_InitFlexIO();
    LCD_InitPanel();
    LCD_Clear();

    /*
     * From this point onward the scheduler will be running when LCD frame
     * transfers are requested, so runtime memory writes use the IRQ-based
     * transfer path.
     */
    s_lcd_async_enabled = true;

    PRINTF("LCD: ready\r\n");
}

void LCD_LiveShowCameraStripe(uint32_t stripe_index,
                              const uint16_t *pixels)
{
    if (pixels == NULL)
    {
        return;
    }

    if (stripe_index >= CAMERA_STRIPE_COUNT)
    {
        return;
    }

    uint16_t start_y =
        (uint16_t)(stripe_index * CAMERA_STRIPE_H);

    uint16_t end_y =
        (uint16_t)(start_y + CAMERA_STRIPE_H - 1U);

    ST7796S_SelectArea(
        &s_lcd_handle,
        0U,
        start_y,
        LCD_LOGICAL_WIDTH - 1U,
        end_y);

    ST7796S_WritePixels(
        &s_lcd_handle,
        (uint16_t *)pixels,
        CAMERA_STRIPE_PIXELS);
}

void LCD_LiveShowCameraFrame(const uint16_t *pixels)
{
    if (pixels == NULL)
    {
        return;
    }

    /*
     * The camera produces a 320x240 frame while the LCD is 320x480.
     * The frame is sent as sixteen consecutive 320x15 stripes.
     */
    for (uint32_t stripe = 0U;
         stripe < CAMERA_STRIPE_COUNT;
         stripe++)
    {
        const uint16_t *stripe_pixels =
            pixels + (stripe * CAMERA_STRIPE_PIXELS);

        PRINTF(
            "LCD: sending stripe %u\r\n",
            (unsigned int)stripe);

        LCD_LiveShowCameraStripe(
            stripe,
            stripe_pixels);

        PRINTF(
            "LCD: stripe %u finished\r\n",
            (unsigned int)stripe);
    }

    PRINTF("LCD: full frame finished\r\n");
}

void LCD_DebugPrintState(void)
{
    PRINTF("LCD HANDLE:\r\n");

    PRINTF(
        "  handle          = 0x%08X\r\n",
        (unsigned int)(uintptr_t)&s_lcd_handle);

    PRINTF(
        "  xferOps         = 0x%08X\r\n",
        (unsigned int)(uintptr_t)s_lcd_handle.xferOps);

    PRINTF(
        "  xferOpsData     = 0x%08X\r\n",
        (unsigned int)(uintptr_t)s_lcd_handle.xferOpsData);

    PRINTF(
        "  expectedOps     = 0x%08X\r\n",
        (unsigned int)(uintptr_t)&s_flexio_dbi_ops);

    PRINTF(
        "  expectedData    = 0x%08X\r\n",
        (unsigned int)(uintptr_t)&s_flexio_lcd);
}