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
//lista de operatori disponibili pentru TensorFlow Lite Micro
tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<13> s_microOpResolver;

    /* Operators used by the NanoDet network itself. */
    s_microOpResolver.AddPad();// Pad operator is used to add padding to the input tensor, which is necessary for certain convolution operations in the NanoDet model. Padding helps maintain the spatial dimensions of the input tensor after convolution, allowing the model to learn features effectively without losing information at the borders of the input image.
    s_microOpResolver.AddConv2D();// Conv2D operator is a fundamental building block of convolutional neural networks (CNNs) like NanoDet. It applies a set of learnable filters to the input tensor, producing feature maps that capture spatial hierarchies and patterns in the data. This operation is essential for extracting meaningful features from images, enabling the model to detect objects accurately.
    s_microOpResolver.AddLeakyRelu();// LeakyReLU operator introduces non-linearity into the model while allowing a small gradient for negative input values. This helps prevent the "dying ReLU" problem, where neurons become inactive and stop learning. In the context of NanoDet, LeakyReLU enhances the model's ability to learn complex patterns and improves overall performance in object detection tasks.
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
//graful Neutron care e trimis catre NPU poate folosi aceste operatii din lista/registru
//CPU-ul tot este folosit pentru partea de runtime TFLM, preprocessing, postprocessing/decoder NanoDet, coordonare