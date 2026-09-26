#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/neutron/neutron.h"


/*
 * Register the operators required by the NanoDet model.
 *
 * The resolver tells TensorFlow Lite Micro which built-in and custom
 * operations are available when the model is loaded. Keeping only the
 * operators used by the model avoids allocating unnecessary resolver data.
 */
tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<13> s_microOpResolver;

    /* Operators used by the NanoDet network itself. */
    s_microOpResolver.AddPad();
    s_microOpResolver.AddConv2D();
    s_microOpResolver.AddLeakyRelu();
    s_microOpResolver.AddDepthwiseConv2D();
    s_microOpResolver.AddConcatenation();
    s_microOpResolver.AddReshape();
    s_microOpResolver.AddTranspose();
    s_microOpResolver.AddSlice();
    s_microOpResolver.AddResizeBilinear();
    s_microOpResolver.AddAdd();
    s_microOpResolver.AddLogistic();
    s_microOpResolver.AddQuantize();

    /*
     * NEUTRON_GRAPH is the custom operator used by the NXP Neutron backend.
     * This allows TensorFlow Lite Micro to recognize the Neutron graph stored
     * inside the .tflite model and execute it using the NPU.
     */
    s_microOpResolver.AddCustom(
        tflite::GetString_NEUTRON_GRAPH(),
        tflite::Register_NEUTRON_GRAPH());

    return s_microOpResolver;
}