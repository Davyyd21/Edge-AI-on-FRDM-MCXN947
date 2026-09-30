#include "model.h"
#include "fsl_debug_console.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "nanodet_m_0.5x_fruits_neutron.h"
#include "nanodet_decoder.h"


/*
 * Keep the model, interpreter and decoded detections here so the rest of
 * the application can access the model through the functions in model.h.
 */
static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;

static NanoDetDetection s_detections[NANODET_MAX_DETECTIONS];
static int s_detection_count = 0;

static uint32_t s_tensor_arena_used = 0U;


/*
 * The operator resolver is created in a separate file because the NanoDet
 * model uses both standard TensorFlow Lite operators and the custom
 * NEUTRON_GRAPH operator.
 */
extern tflite::MicroOpResolver &MODEL_GetOpsResolver();


/*
 * TensorFlow Lite Micro uses this arena for tensors and intermediate data
 * required during inference. Its size was chosen to fit the current
 * NanoDet model on the MCXN947.
 *
 * The 16-byte alignment is required for the tensor memory used by the
 * embedded inference runtime.
 */
static uint8_t s_tensor_arena[146 * 1024] __attribute__((aligned(16)));


/*
 * Load the model, create the interpreter and allocate all tensors.
 *
 * This function also checks the model input/output configuration so that
 * problems are detected during startup instead of appearing later when
 * inference is already running.
 */
int MODEL_Init(void)
{
    PRINTF("NanoDet: GetModel...\r\n");

    s_model = tflite::GetModel(model_data);

    if (s_model == nullptr)
    {
        PRINTF("NanoDet ERROR: model is NULL\r\n");
        return -1;
    }

    PRINTF("NanoDet: model found, version=%d\r\n", (int)s_model->version());

    /*
     * TensorFlow Lite Micro expects the model schema version supported by
     * the runtime. A mismatch means the model cannot be safely interpreted.
     */
    if (s_model->version() != TFLITE_SCHEMA_VERSION)
    {
        PRINTF("NanoDet ERROR: schema mismatch\r\n");
        return -2;
    }

    PRINTF("NanoDet: schema OK\r\n");
    //Resolver = spune lui TFLM ce operații știe să execute
    //TFLM este practic runtime-ul care încarcă modelul NanoDet, gestionează tensorii și execută inference-ul,
    //folosind memoria noastră (tensor arena) și
    //backend-ul Neutron/NPU unde este cazul
    tflite::MicroOpResolver &resolver = MODEL_GetOpsResolver();

    PRINTF("NanoDet: resolver ready\r\n");

    /*
     * Keep the interpreter static because dynamic allocation is undesirable
     * on the embedded target. All tensor memory comes from s_tensor_arena.
     */
    //Interpreter = este obiectul care încarcă modelul și îl execută, folosind operațiile din resolver și memoria din tensor arena
    static tflite::MicroInterpreter static_interpreter(
        s_model,
        resolver,
        s_tensor_arena,
        sizeof(s_tensor_arena));

    s_interpreter = &static_interpreter;

    PRINTF("NanoDet: AllocateTensors...\r\n");

    TfLiteStatus status = s_interpreter->AllocateTensors();

    if (status != kTfLiteOk)
    {
        PRINTF("NanoDet ERROR: AllocateTensors failed, status=%d\r\n", (int)status);
        return -3;
    }

    PRINTF("NanoDet: AllocateTensors OK\r\n");

    /*
     * Keep track of the actual arena usage. This is useful when checking
     * how much RAM the model consumes on the MCXN947.
     */
    s_tensor_arena_used = (uint32_t)s_interpreter->arena_used_bytes();

    PRINTF("NanoDet tensor arena: %u bytes\r\n",
           (unsigned int)s_tensor_arena_used);

    TfLiteTensor *input = s_interpreter->input(0);

    if (input == nullptr)
    {
        PRINTF("NanoDet ERROR: input tensor is NULL\r\n");
        return -4;
    }

    PRINTF("NanoDet input: bytes=%u\r\n", (unsigned int)input->bytes);
    PRINTF("NanoDet input dims: ");

    /*
     * Print the complete input shape, for example:
     * 1x96x96x3
     */
    for (int i = 0; i < input->dims->size; ++i)
    {
        PRINTF("%d", input->dims->data[i]);

        if (i + 1 < input->dims->size)
        {
            PRINTF("x");
        }
    }

    PRINTF("\r\n");

    /*
     * The current NanoDet model has exactly one input tensor.
     */
    if (s_interpreter->inputs_size() != 1)
    {
        PRINTF("NanoDet ERROR: expected 1 input, got %d\r\n",
               (int)s_interpreter->inputs_size());
        return -5;
    }

    PRINTF("NanoDet: inputs=%d\r\n", (int)s_interpreter->inputs_size());

    /*
     * The exported NanoDet model also has one output tensor containing
     * the detection data used by the decoder.
     */
    if (s_interpreter->outputs_size() != 1)
    {
        PRINTF("NanoDet ERROR: expected 1 output, got %d\r\n",
               (int)s_interpreter->outputs_size());
        return -6;
    }

    PRINTF("NanoDet: outputs=%d\r\n", (int)s_interpreter->outputs_size());

    TfLiteTensor *output = s_interpreter->output(0);

    if (output == nullptr)
    {
        PRINTF("NanoDet ERROR: output tensor is NULL\r\n");
        return -7;
    }

    PRINTF("NanoDet output: bytes=%u\r\n", (unsigned int)output->bytes);
    PRINTF("NanoDet output dims: ");

    /*
     * Print the output shape so it is easy to verify that the tensor still
     * matches what nanodet_decoder expects.
     */
    for (int i = 0; i < output->dims->size; ++i)
    {
        PRINTF("%d", output->dims->data[i]);

        if (i + 1 < output->dims->size)
        {
            PRINTF("x");
        }
    }

    PRINTF("\r\n");

    /*
     * The output is quantized, so the decoder needs both the scale and
     * zero-point to convert the int8 values back to the model's float range.
     *
     * The value printed here is multiplied by 1,000,000 only to make the
     * small scale easier to read in the serial terminal.
     */
    uint32_t scale_u = (uint32_t)(output->params.scale * 1000000.0f);

    PRINTF("NanoDet output quantization: scale_x1e6=%u zero_point=%d\r\n",
           (unsigned int)scale_u,
           output->params.zero_point);

    PRINTF("NanoDet output type=%d\r\n", (int)output->type);

    /*
     * No detections are available until the first inference is completed.
     */
    s_detection_count = 0;
    memset(s_detections, 0, sizeof(s_detections));

    return 0;
}


/*
 * Run one complete NanoDet inference and decode its output.
 *
 * The input tensor has already been filled by the camera task. Invoke()
 * performs the actual model execution, after which the raw output tensor
 * is passed to NanoDet_Decode().
 */
int MODEL_RunInference(void)
{
    if (s_interpreter == nullptr)
    {
        PRINTF("INFERENCE ERROR: interpreter is NULL\r\n");
        return -1;
    }

    TfLiteTensor *input = s_interpreter->input(0);

    if (input == nullptr)
    {
        PRINTF("INFERENCE ERROR: input tensor is NULL\r\n");
        return -2;
    }

    PRINTF("INFERENCE: input=%p bytes=%u\r\n",
           (void *)input->data.int8,
           (unsigned int)input->bytes);

    PRINTF("INFERENCE: invoking...\r\n");

    /*
     * Invoke() is blocking, so this function returns only after the complete
     * NanoDet graph has finished executing.
     */
    TfLiteStatus status = s_interpreter->Invoke();

    PRINTF("INFERENCE: Invoke returned=%d\r\n", (int)status);

    if (status != kTfLiteOk)
    {
        PRINTF("INFERENCE ERROR: Invoke failed\r\n");
        return -3;
    }

    TfLiteTensor *output = s_interpreter->output(0);

    if (output == nullptr || output->data.int8 == nullptr)
    {
        PRINTF("DECODE ERROR: output unavailable\r\n");
        return -4;
    }

    /*
     * Keep the decoded detections separate from the shared result array
     * until decoding is complete. This avoids partially updating the
     * results if the decoder is still working.
     */
    static NanoDetDetection decoded[NANODET_MAX_DETECTIONS];

    int detection_count = NanoDet_Decode(
        output->data.int8,
        output->params.scale,
        output->params.zero_point,
        decoded,
        NANODET_MAX_DETECTIONS);

    if (detection_count < 0)
    {
        detection_count = 0;
    }

    if (detection_count > NANODET_MAX_DETECTIONS)
    {
        detection_count = NANODET_MAX_DETECTIONS;
    }

    /*
     * Store the results of this inference so the camera task can use them
     * when drawing the bounding boxes on the next frame.
     */
    memcpy(s_detections,decoded,(size_t)detection_count * sizeof(NanoDetDetection));
    s_detection_count = detection_count;

    PRINTF("NANODET: detections=%d\r\n", detection_count);

    for (int i = 0; i < detection_count; ++i)
    {
        const NanoDetDetection &det = s_detections[i];

        const char *class_name = "Unknown";

        if (det.class_id == 0)
        {
            class_name = "Apple";
        }
        else if (det.class_id == 1)
        {
            class_name = "Cherry";
        }
        else if (det.class_id == 2)
        {
            class_name = "Tomato";
        }

        PRINTF("DET %d: %s score_x1000=%d box=(%d,%d)-(%d,%d)\r\n",
               i,
               class_name,
               (int)(det.score * 1000.0f),
               (int)det.x1,
               (int)det.y1,
               (int)det.x2,
               (int)det.y2);
    }

    return 0;
}


/*
 * Copy the latest decoded detections into a caller-provided buffer.
 *
 * max_detections prevents this function from writing more detections than
 * the destination buffer can hold.
 */
int MODEL_GetDetections(NanoDetDetection *detections, int max_detections)
{
    if (detections == nullptr || max_detections <= 0)
    {
        return 0;
    }

    int count = s_detection_count;

    if (count > max_detections)
    {
        count = max_detections;
    }
    memcpy(detections,s_detections,(size_t)count * sizeof(NanoDetDetection));
    return count;
}

/*
 * Return a pointer to the model input tensor.
 *
 * CameraTask uses this interface to copy the preprocessed 96x96 image
 * into the tensor before MODEL_RunInference() is called.
 */
int8_t *MODEL_GetInputData(void)
{
    if (s_interpreter == nullptr)
    {
        return nullptr;
    }

    TfLiteTensor *input = s_interpreter->input(0);

    if (input == nullptr)
    {
        return nullptr;
    }

    return input->data.int8;
}


/*
 * Return a pointer to one of the model output tensors.
 *
 * The current model has one output, but keeping the index parameter makes
 * this helper usable with models containing multiple outputs as well.
 */
int8_t *MODEL_GetOutputData(uint32_t index)
{
    if (s_interpreter == nullptr)
    {
        return nullptr;
    }

    if (index >= (uint32_t)s_interpreter->outputs_size())
    {
        return nullptr;
    }

    TfLiteTensor *output = s_interpreter->output(index);

    if (output == nullptr)
    {
        return nullptr;
    }

    return output->data.int8;
}


/*
 * Return the size of an input tensor in bytes.
 */
uint32_t MODEL_GetInputSize(void)
{
    if (s_interpreter == nullptr)
    {
        return 0U;
    }

    TfLiteTensor *input = s_interpreter->input(0);

    if (input == nullptr)
    {
        return 0U;
    }

    return (uint32_t)input->bytes;
}


/*
 * Return the size of an output tensor in bytes.
 */
uint32_t MODEL_GetOutputSize(uint32_t index)
{
    if (s_interpreter == nullptr)
    {
        return 0U;
    }

    if (index >= (uint32_t)s_interpreter->outputs_size())
    {
        return 0U;
    }

    TfLiteTensor *output = s_interpreter->output(index);

    if (output == nullptr)
    {
        return 0U;
    }

    return (uint32_t)output->bytes;
}


/*
 * Return the quantization scale used by an output tensor.
 *
 * The decoder uses this together with the zero-point to convert the
 * quantized int8 output back into its real numerical values.
 */
float MODEL_GetOutputScale(uint32_t index)
{
    if (s_interpreter == nullptr)
    {
        return 0.0f;
    }

    if (index >= (uint32_t)s_interpreter->outputs_size())
    {
        return 0.0f;
    }

    TfLiteTensor *output = s_interpreter->output(index);

    if (output == nullptr)
    {
        return 0.0f;
    }

    return output->params.scale;
}


/*
 * Return the zero-point used to quantize an output tensor.
 */
int32_t MODEL_GetOutputZeroPoint(uint32_t index)
{
    if (s_interpreter == nullptr)
    {
        return 0;
    }

    if (index >= (uint32_t)s_interpreter->outputs_size())
    {
        return 0;
    }

    TfLiteTensor *output = s_interpreter->output(index);

    if (output == nullptr)
    {
        return 0;
    }

    return output->params.zero_point;
}


/*
 * Return the amount of tensor arena memory actually used by the model.
 * This is mainly useful for monitoring the RAM footprint of NanoDet.
 */
uint32_t MODEL_GetTensorArenaUsed(void)
{
    return s_tensor_arena_used;
}