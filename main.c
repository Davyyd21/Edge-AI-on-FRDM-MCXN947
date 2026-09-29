#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "app.h"
#include "board.h"
#include "fsl_common.h"
#include "fsl_debug_console.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

#include "camera_stripe_capture.h"
#include "lcd_live.h"
#include "ov7670_demo.h"
#include "image_preprocess.h"

#include "NeutronDriver.h"
#include "NeutronErrors.h"
#include "model.h"

void HardFault_Dump(uint32_t *stack);

#define NANODET_INPUT_SIZE        (96U * 96U * 3U)
#define INFERENCE_BUFFER_COUNT    2U
#define CAMERA_TASK_STACK_SIZE    512U
#define INFERENCE_TASK_STACK_SIZE 512U
#define CAMERA_TASK_PRIORITY      3U
#define INFERENCE_TASK_PRIORITY   3U


/*
 * Two buffers are used so the camera can prepare a new image while
 * the inference task is still processing the previous one.
 */
static int8_t s_inference_buffers[INFERENCE_BUFFER_COUNT][NANODET_INPUT_SIZE]
    __attribute__((section(".sramx")))
    __attribute__((aligned(16)));

static QueueHandle_t s_free_queue = NULL;
static QueueHandle_t s_ready_queue = NULL;


/*
 * These detections are shared between the inference and camera tasks.
 * The inference task updates them, while the camera task uses them to
 * draw the latest bounding boxes on the displayed image.
 */
static NanoDetDetection s_overlay_detections[NANODET_MAX_DETECTIONS]
    __attribute__((section(".sramx")))
    __attribute__((aligned(16)));

static volatile uint32_t s_overlay_count
    __attribute__((section(".sramx")))
    __attribute__((aligned(4)));


/*
 * Save the CPU state when a HardFault occurs. The selected stack pointer
 * is passed to HardFault_Dump() so the registers can be printed.
 */
__attribute__((naked))
void HardFault_Handler(void)
{
    __asm volatile(
        "tst lr, #4        \n"
        "ite eq            \n"
        "mrseq r0, msp     \n"
        "mrsne r0, psp     \n"
        "b HardFault_Dump  \n"
    );
}


/*
 * Print the registers saved by the processor when the fault happened.
 * These values are useful for finding the instruction and memory access
 * that caused the crash.->using arm-none-eabi-addr2line to find the source line from the PC value
 */
void HardFault_Dump(uint32_t *stack)
{
    PRINTF("\r\n========== HARDFAULT ==========\r\n");
    PRINTF("LR    = 0x%08X\r\n", (unsigned int)stack[5]);
    PRINTF("PC    = 0x%08X\r\n", (unsigned int)stack[6]);
    PRINTF("R0    = 0x%08X\r\n", (unsigned int)stack[0]);
    PRINTF("R1    = 0x%08X\r\n", (unsigned int)stack[1]);
    PRINTF("R2    = 0x%08X\r\n", (unsigned int)stack[2]);
    PRINTF("R3    = 0x%08X\r\n", (unsigned int)stack[3]);
    PRINTF("R12   = 0x%08X\r\n", (unsigned int)stack[4]);
    PRINTF("ARG5  = 0x%08X\r\n", (unsigned int)stack[7]);
    PRINTF("ARG6  = 0x%08X\r\n", (unsigned int)stack[8]);
    PRINTF("ARG7  = 0x%08X\r\n", (unsigned int)stack[9]);
    PRINTF("ARG8  = 0x%08X\r\n", (unsigned int)stack[10]);
    PRINTF("CFSR  = 0x%08X\r\n", (unsigned int)SCB->CFSR);
    PRINTF("HFSR  = 0x%08X\r\n", (unsigned int)SCB->HFSR);
    PRINTF("BFAR  = 0x%08X\r\n", (unsigned int)SCB->BFAR);
    PRINTF("================================\r\n");

    while (1)
    {
    }
}


/*
 * Copy the latest detections into a local buffer. Since the inference task
 * can update the shared detection array at the same time, the copy is
 * protected by a short critical section.
 */
static void SnapshotDetections(NanoDetDetection *dst, uint32_t *count)
{
    //the other task(CameraTask)cannot enter this critical section, so we can safely read the shared buffer and count
    taskENTER_CRITICAL();

    uint32_t n = s_overlay_count;

    if (n > NANODET_MAX_DETECTIONS)
    {
        n = NANODET_MAX_DETECTIONS;
    }
    //we don't let Cameratask to work with the shared buffer, so we can just copy the detections to the local buffer and return the count
    //the reason why we don't let Cameratask to work with the shared buffer is because we don't want to have a race condition, where Cameratask is reading the shared buffer while InferenceTask is writing to it, which would cause Cameratask to read invalid data, so we just copy the detections to the local buffer and return the count, so Cameratask can work with the local buffer without worrying about InferenceTask updating the shared buffer at the same time
    memcpy(dst, s_overlay_detections, n * sizeof(NanoDetDetection));//dst is a local buffer
    *count = n;

    taskEXIT_CRITICAL();
    //CameraTask will only use the local buffer and count, so it will not be affected by the inference task updating the shared buffer
}


/*
 * Get the detections produced by the latest inference and make them
 * available to the camera task for drawing.
 */
static void UpdateDetections(void)
{
    NanoDetDetection detections[NANODET_MAX_DETECTIONS];
    //we obtained the detections from the model and store them in a local buffer, then we copy them to the shared buffer in a critical section
    int count = MODEL_GetDetections(detections, NANODET_MAX_DETECTIONS);

    if (count < 0)
    {
        count = 0;
    }

    if (count > NANODET_MAX_DETECTIONS)
    {
        count = NANODET_MAX_DETECTIONS;
    }
    //we want to make sure that the shared buffer is not being accessed by the camera task while we are updating it, so we enter a critical section
    //because the camera task will not be able to enter this critical section, we can safely copy the detections to the shared buffer and update the count
    //in order for the CameraTask not to use old detections, while InferenceTask is updating the shared buffer, we need to make sure that the CameraTask will not be able to access the shared buffer while we are updating it, so we enter a critical section
    taskENTER_CRITICAL();

    memcpy(s_overlay_detections, detections, (size_t)count * sizeof(NanoDetDetection));
    s_overlay_count = (uint32_t)count;

    taskEXIT_CRITICAL();
    //after this,the CameraTask can take safely the snapshot of the shared buffer and use it to draw the detections on the camera frame, without worrying about the InferenceTask updating the shared buffer at the same time
}


/*
 * Print the detections in a readable form. The model uses numeric class
 * IDs, so they are converted to the names used by the project here.
 */
static void PrintDetections(uint32_t inference_counter)
{
    NanoDetDetection detections[NANODET_MAX_DETECTIONS];
    uint32_t count = 0U;

    SnapshotDetections(detections, &count);

    PRINTF("NANODET: detections=%u\r\n", (unsigned int)count);

    for (uint32_t i = 0U; i < count; i++)
    {
        const NanoDetDetection *det = &detections[i];
        const char *class_name = "Unknown";

        if (det->class_id == 0)
        {
            class_name = "Apple";
        }
        else if (det->class_id == 1)
        {
            class_name = "Cherry";
        }
        else if (det->class_id == 2)
        {
            class_name = "Tomato";
        }

        PRINTF("DET %u: %s score_x1000=%d box=(%d,%d)-(%d,%d)\r\n",
               (unsigned int)i,
               class_name,
               (int)(det->score * 1000.0f),
               (int)det->x1,
               (int)det->y1,
               (int)det->x2,
               (int)det->y2);
    }

    (void)inference_counter;
}


/*
 * Draw a four-pixel-thick rectangle directly into the RGB565 camera frame.
 * Coordinates are clamped first so a bad detection cannot write outside
 * the image buffer.
 */
static void DrawRectOnCameraFrame(uint16_t *frame, int x0, int y0, int x1, int y1, uint16_t color)
{
    if (frame == NULL)
    {
        return;
    }

    if (x0 < 0)
    {
        x0 = 0;
    }

    if (y0 < 0)
    {
        y0 = 0;
    }

    if (x1 >= (int)CAMERA_WIDTH)
    {
        x1 = (int)CAMERA_WIDTH - 1;
    }

    if (y1 >= (int)CAMERA_HEIGHT)
    {
        y1 = (int)CAMERA_HEIGHT - 1;
    }

    if (x0 >= (int)CAMERA_WIDTH || y0 >= (int)CAMERA_HEIGHT || x1 <= x0 || y1 <= y0)
    {
        return;
    }
    //thickness of the rectangle is 4 pixels, we draw the rectangle by drawing 4 lines, top, bottom, left and right, we use a for loop to draw the lines, we use the thickness variable to determine how many pixels to draw for each line
    for (int thickness = 0; thickness < 4; ++thickness)//thickness is the number of pixels to draw for each line, we draw the top and bottom lines first, then the left and right lines, we use the thickness variable to determine how many pixels to draw for each line
    {
        int top = y0 + thickness;
        int bottom = y1 - thickness;
        int left = x0 + thickness;
        int right = x1 - thickness;

        if (top <= y1)//we draw the top line first, we check if the top line is within the bounds of the image, if it is, we draw the line by setting the pixels in the frame to the color, we use a for loop to iterate over the x coordinates of the line, we use the top variable to determine the y coordinate of the line
        {
            for (int x = x0; x <= x1; ++x)
            {
                frame[top * CAMERA_WIDTH + x] = color;
            }
        }

        if (bottom >= y0)//we draw the bottom line, we check if the bottom line is within the bounds of the image, if it is, we draw the line by setting the pixels in the frame to the color, we use a for loop to iterate over the x coordinates of the line, we use the bottom variable to determine the y coordinate of the line
        {
            for (int x = x0; x <= x1; ++x)
            {
                frame[bottom * CAMERA_WIDTH + x] = color;
            }
        }

        if (left <= x1)
        {
            for (int y = y0; y <= y1; ++y)
            {
                frame[y * CAMERA_WIDTH + left] = color;
            }
        }

        if (right >= x0)
        {
            for (int y = y0; y <= y1; ++y)
            {
                frame[y * CAMERA_WIDTH + right] = color;
            }
        }
    }
}


/*
 * Convert the 96x96 NanoDet coordinates to the original 320x240 camera
 * coordinates and draw every detection on the current frame.
 *
 * Each class has its own RGB565 color so the detections can be distinguished
 * directly on the LCD.
 */
static void DrawDetectionsOnCameraFrame(uint16_t *frame)
{
    NanoDetDetection detections[NANODET_MAX_DETECTIONS];
    uint32_t count = 0U;

    SnapshotDetections(detections, &count);

    for (uint32_t i = 0U; i < count; i++)
    {
        const NanoDetDetection *det = &detections[i];
        //facem mapare liniara
        //we convert the 96x96 coordinates to the original 320x240 camera coordinates by multiplying the x and y coordinates by the width and height of the camera and dividing by 96, we use integer division to avoid floating point operations, we also clamp the coordinates to be within the bounds of the camera frame, we then draw a rectangle on the camera frame using the converted coordinates and a color based on the class_id of the detection
        //to understand easier the conversion, we can think of the 96x96 coordinates as a percentage of the camera frame, so we multiply by the width and height of the camera to get the actual pixel coordinates, then we divide by 96 to get the corresponding pixel coordinates in the camera frame
        int x0 = ((int)det->x1 * (int)CAMERA_WIDTH) / 96;//to explain it easy, we take the x1 coordinate of the detection, which is in the range [0, 96], and multiply it by the width of the camera frame (320), then divide by 96 to get the corresponding x coordinate in the camera frame. We do the same for y0, x1, and y1 using the respective coordinates of the detection and the height of the camera frame (240).
        int y0 = ((int)det->y1 * (int)CAMERA_HEIGHT) / 96;
        int x1 = ((int)det->x2 * (int)CAMERA_WIDTH) / 96;
        int y1 = ((int)det->y2 * (int)CAMERA_HEIGHT) / 96;
        //practic transformarile astea vor sa zica ca det->x1 in formatul 96x96 este echivalentul lui x0 in formatul 320x240
        uint16_t color = 0xFFFFU;//default color is white, if the class_id is not 0,1,2, we will use white color to draw the rectangle

        if (det->class_id == 0)
        {
            color = 0xF800U;//red
        }
        else if (det->class_id == 1)
        {
            color = 0x07E0U;//green
        }
        else if (det->class_id == 2)
        {
            color = 0x001FU;//blue
        }

        DrawRectOnCameraFrame(frame, x0, y0, x1, y1, color);
    }
}


/*
 * The inference task waits for a processed camera buffer, copies it into
 * the model input tensor, runs NanoDet and then returns the buffer to the
 * free queue so it can be reused by the camera.
 */
static void InferenceTask(void *pvParameters)
{
    (void)pvParameters;
    PRINTF("INFERENCE: task started\r\n");
    uint32_t inference_counter = 0U;
    while (1)
    {
        uint32_t buffer_index = 0U;
        if (xQueueReceive(s_ready_queue, &buffer_index, portMAX_DELAY) != pdPASS)
        {
            continue;
        }

        if (buffer_index >= INFERENCE_BUFFER_COUNT)
        {
            continue;
        }

        /*
          The model has its own input tensor, so the preprocessed image
          has to be copied from our buffer before starting inference
         */
        int8_t *model_input = MODEL_GetInputData();//contine adresa memoriei la care modelul asteapta datele de intrare, adica tensorul de intrare al modelului, care este un buffer de dimensiune 96x96x3, unde 3 reprezinta canalele RGB, iar 96x96 reprezinta dimensiunea imaginii de intrare. Aceasta functie returneaza un pointer catre acest buffer, astfel incat sa putem copia datele preprocesate din bufferul nostru in acest tensor de intrare al modelului.
        //practic aici copiem datele de tip 96x96x3
        if (model_input == NULL)
        {
            PRINTF("INFERENCE TASK: model input NULL\r\n");
            //portMAX_DELAY doesn't do pooling, so we can safely return the buffer to the free queue without blocking, and continue to the next iteration of the loop
            (void)xQueueSend(s_free_queue, &buffer_index, portMAX_DELAY);
            //eliberez bufferul de intrare, astfel incat sa poata fi reutilizat de catre CameraTask pentru a procesa urmatorul frame. Daca nu facem asta, atunci bufferul va ramane ocupat si CameraTask nu va putea sa-l foloseasca pentru a procesa urmatorul frame, ceea ce va duce la pierderea unor frame-uri si la o performanta mai slaba
            continue;
        }

        memcpy(model_input, s_inference_buffers[buffer_index], NANODET_INPUT_SIZE);

        inference_counter++;

        PRINTF("INFERENCE TASK: inference %u, buffer=%u\r\n",
               (unsigned int)inference_counter,
               (unsigned int)buffer_index);

        /*
         * MODEL_RunInference() is blocking. When it returns successfully,
         * MODEL_GetDetections() contains the results of this inference.
         */
        int status = MODEL_RunInference();

        if (status == 0)
        {
            UpdateDetections();
            PrintDetections(inference_counter);
        }
        else
        {
            PRINTF("INFERENCE TASK: MODEL_RunInference failed: %d\r\n", status);
        }

        /* The buffer is no longer needed by inference, so make it reusable. */
        (void)xQueueSend(s_free_queue, &buffer_index, portMAX_DELAY);
    }
}


/*
 * This task keeps the camera/LCD pipeline running continuously.
 *
 * Inference is optional for each individual frame: if no AI buffer is free,
 * the frame is still displayed but is simply not sent to NanoDet.
 */
static void CameraTask(void *pvParameters)
{
    (void)pvParameters;

    PRINTF("CAMERA: task started\r\n");

    uint32_t frame_counter = 0U;
    uint32_t dropped_inference_frames = 0U;

    while (1)
    {
        uint32_t frame_index = 0U;
        uint32_t buffer_index = 0U;
        const uint16_t *frame = NULL;

        /*
         * Do not block here. The camera should continue working even if
         * NanoDet is currently using both inference buffers.
         */
        //acel 0 ne spune ca nu vrem sa asteptam daca coada e goala, pentru ca nu vrem sa blocam CameraTask-ul, deci daca coada e goala, inseamna ca InferenceTask-ul nu a terminat inca de procesat frame-urile anterioare, deci nu avem niciun buffer de intrare disponibil pentru a procesa urmatorul frame, deci nu putem sa facem inferenta pe acest frame, dar il putem afisa pe LCD
        bool ai_buffer_available = (xQueueReceive(s_free_queue, &buffer_index, 0U) == pdPASS);

        if (ai_buffer_available && buffer_index >= INFERENCE_BUFFER_COUNT)
        {
            ai_buffer_available = false;
        }

        if (!CAMERA_WaitForStripe(&frame_index, &frame))
        {
            if (ai_buffer_available)
            {
                (void)xQueueSend(s_free_queue, &buffer_index, 0U);
            }

            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
            //practic daca n-am luat niciun frame de la camera, atunci nu avem ce sa facem, deci eliberam bufferul de intrare daca l-am luat si asteptam 1ms inainte de a incerca din nou sa luam un frame de la camera
        }

        /*
         * Draw the results from the latest completed inference before
         * displaying the current frame.
         */
        DrawDetectionsOnCameraFrame((uint16_t *)frame);

        LCD_LiveShowCameraFrame(frame);

        if (ai_buffer_available)
        {
            /*
             * Convert the camera's 320x240 RGB565 frame into the 96x96
             * int8 representation expected by NanoDet.
             */
            RGB565_ResizeToINT8(frame, s_inference_buffers[buffer_index]);

            CAMERA_ReleaseStripe();

            /*
             * The buffer is now ready for the inference task. A zero
             * timeout keeps the camera from blocking on this queue.
             */
            if (xQueueSend(s_ready_queue, &buffer_index, 0U) != pdPASS)//acel 0 ne spune ca nu vrem sa asteptam daca coada e plina, pentru ca nu vrem sa blocam CameraTask-ul, deci daca coada e plina, inseamna ca InferenceTask-ul nu a terminat inca de procesat frame-urile anterioare, deci eliberam bufferul de intrare si incrementam contorul de frame-uri pierdute
            {
                (void)xQueueSend(s_free_queue, &buffer_index, 0U);//daca nu am reusit sa punem bufferul in coada de ready, inseamna ca InferenceTask-ul nu a terminat inca de procesat frame-urile anterioare, deci eliberam bufferul de intrare si incrementam contorul de frame-uri pierdute
                dropped_inference_frames++;
                //xQueueSend(s_free_queue,...) ne spune ca vrem sa punem bufferul inapoi in coada de free, pentru ca InferenceTask-ul nu a terminat inca de procesat frame-urile anterioare, deci nu putem sa punem bufferul in coada de ready, deci il punem inapoi in coada de free, astfel incat CameraTask-ul sa poata sa-l foloseasca pentru a procesa urmatorul frame
            }
        }
        else
        {
            /*
             * Both AI buffers are busy, so this frame is displayed but
             * skipped by the inference pipeline.
             */
            CAMERA_ReleaseStripe();

            dropped_inference_frames++;

            if ((dropped_inference_frames % 30U) == 0U)
            {
                PRINTF("CAMERA: dropped AI frames=%u\r\n",
                       (unsigned int)dropped_inference_frames);
            }
        }

        frame_counter++;

        if ((frame_counter % 30U) == 0U)
        {
            PRINTF("CAMERA: frames=%u, AI drops=%u\r\n",
                   (unsigned int)frame_counter,
                   (unsigned int)dropped_inference_frames);
        }

        (void)frame_index;

        /* Give the inference task a chance to run after processing the frame. */
        taskYIELD();//we used this because even if the priorities of the tasks are the same, we want to give the inference task a chance to run after processing the frame, so we yield the CPU to the inference task, so it can run and process the frame, and then we can continue processing the next frame
    }
}


int main(void)
{
    /*
     * Initialize the board, clocks, pins and debug console before touching
     * the camera, LCD or the AI accelerator.
     */
    BOARD_InitHardware();

    /*
     * Neutron is initialized before the model because NanoDet uses it for
     * hardware-accelerated inference.
     */
    NeutronError neutron_error = neutronInit();

    if (neutron_error != ENONE)
    {
        PRINTF("Neutron init failed: %d\r\n", (int)neutron_error);

        while (1)
        {
        }
    }

    SDK_DelayAtLeastUs(500000U, SystemCoreClock);

    /* Initialize the OV7670 camera sensor. */
    DEMO_OV7670_Init();

    SDK_DelayAtLeastUs(1000000U, SystemCoreClock);

    /* Initialize the LCD before starting the camera capture pipeline. */
    LCD_LiveInit();

    SDK_DelayAtLeastUs(200000U, SystemCoreClock);

    /* Start the SmartDMA-based camera frame capture. */
    CAMERA_StripeCaptureInit();

    SDK_DelayAtLeastUs(100000U, SystemCoreClock);

    /*
     * Initialize NanoDet and allocate all tensors required by TensorFlow
     * Lite Micro. The tensor arena usage is printed for memory debugging.
     */
    int model_status = MODEL_Init();

    PRINTF("NanoDet tensor arena: %u bytes\r\n",
           (unsigned int)MODEL_GetTensorArenaUsed());

    if (model_status != 0)
    {
        PRINTF("NanoDet ERROR: MODEL_Init failed: %d\r\n", model_status);

        while (1)
        {
        }
    }

    /*
     * The free queue starts with both buffers available. The ready queue
     * remains empty until the camera produces the first processed image.
     */
    s_free_queue = xQueueCreate(INFERENCE_BUFFER_COUNT, sizeof(uint32_t));
    s_ready_queue = xQueueCreate(INFERENCE_BUFFER_COUNT, sizeof(uint32_t));

    if ((s_free_queue == NULL) || (s_ready_queue == NULL))
    {
        PRINTF("ERROR: Queue creation failed!\r\n");

        while (1)
        {
        }
    }

    /*
     * Put both inference buffers into the free queue so CameraTask can
     * start using them immediately.
     */
    for (uint32_t i = 0U; i < INFERENCE_BUFFER_COUNT; i++)
    {
        uint32_t buffer_index = i;

        if (xQueueSend(s_free_queue, &buffer_index, 0U) != pdPASS)//we did this verification because we want to make sure that the buffer is added to the free queue, if it fails, we print an error message and enter an infinite loop, because we cannot continue without the free queue being initialized properly
        {
            PRINTF("ERROR: failed to initialize buffer %u\r\n",
                   (unsigned int)i);

            while (1)
            {
            }
        }
    }

    /* No detections exist until the first successful NanoDet inference. */
    memset(s_overlay_detections, 0, sizeof(s_overlay_detections));
    s_overlay_count = 0U;

    PRINTF("FreeRTOS queues initialized.\r\n");

    /*
     * CameraTask handles frame capture, LCD output and preprocessing.
     */
    if (xTaskCreate(CameraTask,
                    "CameraTask",
                    CAMERA_TASK_STACK_SIZE,
                    NULL,
                    CAMERA_TASK_PRIORITY,
                    NULL) != pdPASS)
    {
        PRINTF("ERROR: CameraTask creation failed!\r\n");

        while (1)
        {
        }
    }

    PRINTF("CameraTask created successfully.\r\n");

    /*
     * Print the heap before creating the inference task. FreeRTOS allocates
     * the task stack from its configured heap.
     */
    PRINTF("Free heap before InferenceTask: %u\r\n",
           (unsigned int)xPortGetFreeHeapSize());

    /*
     * InferenceTask waits for ready buffers and runs the blocking NanoDet
     * inference. Both tasks use the same priority.
     */
    if (xTaskCreate(InferenceTask,
                    "InferenceTask",
                    INFERENCE_TASK_STACK_SIZE,
                    NULL,
                    INFERENCE_TASK_PRIORITY,
                    NULL) != pdPASS)
    {
        PRINTF("ERROR: InferenceTask creation failed!\r\n");

        while (1)
        {
        }
    }

    PRINTF("InferenceTask created successfully.\r\n");

    PRINTF("Free heap after InferenceTask: %u\r\n",
           (unsigned int)xPortGetFreeHeapSize());

    /*
     * From here FreeRTOS takes over. CameraTask and InferenceTask now
     * run independently and exchange image buffers through the queues.
     */
    PRINTF("Starting scheduler...\r\n");

    vTaskStartScheduler();

    /*
     * The scheduler normally never returns. If it does, something went
     * wrong during scheduler startup.
     */
    PRINTF("ERROR: vTaskStartScheduler returned!\r\n");

    while (1)
    {
    }
}