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
    .flexioBase = DEMO_FLEXIO,// predefined macro that specifies the base address of the FlexIO peripheral used for the LCD interface. This base address is essential for the driver to access and control the FlexIO hardware registers, enabling communication with the LCD.
    .busType = kFLEXIO_MCULCD_8080,// The bus type is set to kFLEXIO_MCULCD_8080, which indicates that the FlexIO interface is configured to use the Intel 8080 parallel bus protocol. This protocol is commonly used for interfacing with LCDs and other peripherals, allowing for efficient data transfer between the microcontroller and the display.
    .dataPinStartIndex = DEMO_FLEXIO_DATA_PIN_START,
    .ENWRPinIndex = DEMO_FLEXIO_WR_PIN,
    .RDPinIndex = DEMO_FLEXIO_RD_PIN,
    .txShifterStartIndex = DEMO_FLEXIO_TX_START_SHIFTER,//in FlexIO, shifters are hardware modules that handle serial-to-parallel or parallel-to-serial data conversion. By defining the starting index, the code indicates which shifter(s) will be used for sending data to the LCD, allowing for proper configuration and operation of the FlexIO interface.
    .txShifterEndIndex = DEMO_FLEXIO_TX_END_SHIFTER,
    .rxShifterStartIndex = DEMO_FLEXIO_RX_START_SHIFTER,// Similar to the transmit shifters, these settings define which shifter(s) will be used for receiving data from the LCD. The starting and ending indices specify the range of shifters that will handle incoming data, ensuring that the FlexIO interface is correctly set up for bidirectional communication with the display.
    .rxShifterEndIndex = DEMO_FLEXIO_RX_END_SHIFTER,
    .timerIndex = DEMO_FLEXIO_TIMER,// The timer index specifies which FlexIO timer will be used to generate the necessary timing signals for the LCD interface. Timers are essential for controlling the timing of data transfers, ensuring that the signals sent to and received from the LCD are synchronized correctly.
    .setCSPin = NULL,// The setCSPin, setRSPin, and setRDWRPin function pointers are initialized to NULL, indicating that no specific functions have been assigned for controlling the Chip Select (CS), Register Select (RS), and Read/Write (RD/WR) pins of the LCD. These pins are typically used to manage the communication protocol with the LCD, and their control functions can be assigned later in the code as needed.
    .setRSPin = NULL,
    .setRDWRPin = NULL
};


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
 * Each command or data transfer is wrapped in a FlexIO transaction so the
 * LCD sees a clean start-to-finish communication sequence.
 */
static status_t LCD_WriteCommand(void *dbi_xfer_handle, uint32_t command)
{
    FLEXIO_MCULCD_Type *flexio_lcd = (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    FLEXIO_MCULCD_StartTransfer(flexio_lcd);
    FLEXIO_MCULCD_WriteCommandBlocking(flexio_lcd, command);
    FLEXIO_MCULCD_StopTransfer(flexio_lcd);

    return kStatus_Success;
}

static status_t LCD_WriteData(void *dbi_xfer_handle, void *data, uint32_t length_bytes)
{
    FLEXIO_MCULCD_Type *flexio_lcd = (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    FLEXIO_MCULCD_StartTransfer(flexio_lcd);
    FLEXIO_MCULCD_WriteDataArrayBlocking(flexio_lcd, data, length_bytes);
    FLEXIO_MCULCD_StopTransfer(flexio_lcd);

    return kStatus_Success;
}

static status_t LCD_WriteMemory(void *dbi_xfer_handle, uint32_t command, const void *data, uint32_t length_bytes)
{
    FLEXIO_MCULCD_Type *flexio_lcd = (FLEXIO_MCULCD_Type *)dbi_xfer_handle;

    FLEXIO_MCULCD_StartTransfer(flexio_lcd);
    FLEXIO_MCULCD_WriteCommandBlocking(flexio_lcd, command);
    FLEXIO_MCULCD_WriteDataArrayBlocking(flexio_lcd, data, length_bytes);
    FLEXIO_MCULCD_StopTransfer(flexio_lcd);

    return kStatus_Success;
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
static uint16_t s_black_stripe[CAMERA_STRIPE_PIXELS] __attribute__((section(".sramx"), aligned(32)));

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

    GPIO_PinInit(DEMO_LCD_RST_GPIO, DEMO_LCD_RST_PIN, &gpio_output_high);
    GPIO_PinInit(DEMO_LCD_CS_GPIO, DEMO_LCD_CS_PIN, &gpio_output_high);
    GPIO_PinInit(DEMO_LCD_RS_GPIO, DEMO_LCD_RS_PIN, &gpio_output_high);

    s_flexio_lcd.setCSPin = LCD_SetCSPin;
    s_flexio_lcd.setRSPin = LCD_SetRSPin;

    PRINTF("LCD: initializing FlexIO...\r\n");

    flexio_mculcd_config_t config;
    FLEXIO_MCULCD_GetDefaultConfig(&config);
    config.baudRate_Bps = DEMO_FLEXIO_BAUDRATE_BPS;

    status_t status = FLEXIO_MCULCD_Init(&s_flexio_lcd, &config, DEMO_FLEXIO_CLOCK_FREQ);

    if (status != kStatus_Success)
    {
        PRINTF("LCD ERROR: FlexIO init failed: %d\r\n", (int)status);

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

    status_t status = ST7796S_Init(&s_lcd_handle, &config, &s_flexio_dbi_ops, &s_flexio_lcd);

    if (status != kStatus_Success)
    {
        PRINTF("LCD ERROR: ST7796S init failed: %d\r\n", (int)status);

        while (1)
        {
        }
    }

    SDK_DelayAtLeastUs(20000U, SystemCoreClock);
    ST7796S_EnableDisplay(&s_lcd_handle, true);
    SDK_DelayAtLeastUs(20000U, SystemCoreClock);
}

static void LCD_Clear(void)
{
    memset(s_black_stripe, 0, sizeof(s_black_stripe));

    PRINTF("LCD: clearing screen...\r\n");

    /*
     * The LCD is cleared stripe by stripe so the same small SRAMX buffer
     * can be reused instead of allocating a full 320x480 frame buffer.
     */
    for (uint32_t stripe = 0U; stripe < (LCD_LOGICAL_HEIGHT / CAMERA_STRIPE_H); stripe++)
    {
        uint16_t start_y = (uint16_t)(stripe * CAMERA_STRIPE_H);
        uint16_t end_y = (uint16_t)(start_y + CAMERA_STRIPE_H - 1U);

        ST7796S_SelectArea(&s_lcd_handle, 0U, start_y, LCD_LOGICAL_WIDTH - 1U, end_y);
        ST7796S_WritePixels(&s_lcd_handle, s_black_stripe, CAMERA_STRIPE_PIXELS);
    }
}

void LCD_LiveInit(void)
{
    PRINTF("LCD: initialization beginning...\r\n");

    LCD_InitFlexIO();
    LCD_InitPanel();
    LCD_Clear();

    PRINTF("LCD: ready\r\n");
}

void LCD_LiveShowCameraStripe(uint32_t stripe_index, const uint16_t *pixels)
{
    if (pixels == NULL)
    {
        return;
    }

    if (stripe_index >= CAMERA_STRIPE_COUNT)
    {
        return;
    }

    uint16_t start_y = (uint16_t)(stripe_index * CAMERA_STRIPE_H);
    uint16_t end_y = (uint16_t)(start_y + CAMERA_STRIPE_H - 1U);

    ST7796S_SelectArea(&s_lcd_handle, 0U, start_y, LCD_LOGICAL_WIDTH - 1U, end_y);
    ST7796S_WritePixels(&s_lcd_handle, (uint16_t *)pixels, CAMERA_STRIPE_PIXELS);
}

void LCD_LiveShowCameraFrame(const uint16_t *pixels)
{
    if (pixels == NULL)
    {
        return;
    }

    /*
     * The camera produces a 320x240 frame while the LCD is 320x480.
     * The frame is therefore sent to the display as consecutive 15-pixel
     * stripes using the existing camera stripe geometry.
     */
    for (uint32_t stripe = 0U; stripe < CAMERA_STRIPE_COUNT; stripe++)
    {
        const uint16_t *stripe_pixels = pixels + (stripe * CAMERA_STRIPE_PIXELS);

        LCD_LiveShowCameraStripe(stripe, stripe_pixels);
    }
}

void LCD_DebugPrintState(void)
{
    PRINTF("LCD HANDLE:\r\n");
    PRINTF("  handle          = 0x%08X\r\n", (unsigned int)(uintptr_t)&s_lcd_handle);
    PRINTF("  xferOps         = 0x%08X\r\n", (unsigned int)(uintptr_t)s_lcd_handle.xferOps);
    PRINTF("  xferOpsData     = 0x%08X\r\n", (unsigned int)(uintptr_t)s_lcd_handle.xferOpsData);
    PRINTF("  expectedOps     = 0x%08X\r\n", (unsigned int)(uintptr_t)&s_flexio_dbi_ops);
    PRINTF("  expectedData    = 0x%08X\r\n", (unsigned int)(uintptr_t)&s_flexio_lcd);
}