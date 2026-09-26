cmake_minimum_required(VERSION 3.22.0)

cmake_path(APPEND conf_file_path ${CMAKE_CURRENT_LIST_DIR} ${board}_${core_id} prj.conf)

set(CONF_FILE ${conf_file_path} ${CONF_FILE})

include(${SdkRootDirPath}/cmake/extension/mcux.cmake)

project(
    Edge_AI
    LANGUAGES C CXX ASM
    PROJECT_BOARD_PORT_PATH
        ${board_root}/${board}/display_examples/smartdma_camera_flexio_mculcd
)

# Enable TensorFlow Lite Micro memory usage reporting for debugging and profiling.
add_compile_definitions(TF_LITE_SHOW_MEMORY_USE)

include(${SdkRootDirPath}/CMakeLists.txt)

# Include the board-specific configuration and generated board files.
include("${board}_${core_id}/board_files.cmake")

# Application source files.
mcux_add_source(
    SOURCES
        main.c
        camera_stripe_capture.c
        lcd_live.c
        ov7670_demo.c
        image_preprocess.c
        model.cpp
        model_nanodet_ops_npu.cpp
        nanodet_decoder.cpp
)

# FreeRTOS kernel source files required by the application.
mcux_add_source(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel-upstream
    SOURCES
        croutine.c
        event_groups.c
        list.c
        queue.c
        stream_buffer.c
        tasks.c
        timers.c
)

# FreeRTOS ARM Cortex-M33 non-secure port implementation.
mcux_add_source(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel-upstream/portable/GCC/ARM_CM33_NTZ/non_secure
    SOURCES
        port.c
        portasm.c
        mpu_wrappers_v2_asm.c
)

# FreeRTOS kernel headers.
mcux_add_include(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel-upstream
    INCLUDES
        include
)

mcux_add_include(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel
    INCLUDES
        include
)

# FreeRTOS Cortex-M33 port headers.
mcux_add_include(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel-upstream/portable/GCC/ARM_CM33_NTZ/non_secure
    INCLUDES
        .
)

mcux_add_include(
    BASE_PATH ${SdkRootDirPath}/rtos/freertos/freertos-kernel-upstream/portable/ARMv8M/non_secure
    INCLUDES
        .
)

# Project and Neutron driver include directories.
mcux_add_include(
    INCLUDES
        .
        neutron/include
)

# Link the Neutron runtime libraries used by the Edge AI application.
mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/neutron/lib
    LIBS libNeutronDriver.a
)

mcux_add_library(
    BASE_PATH ${CMAKE_CURRENT_LIST_DIR}/neutron/lib
    LIBS libNeutronFirmware.a
)

# Convert the final application binary into a .bin image for flashing or deployment.
mcux_convert_binary(
    BINARY ${APPLICATION_BINARY_DIR}/${MCUX_SDK_PROJECT_NAME}.bin
)

# Add the board configuration directory used by MCUXpresso Config Tools.
mcux_add_config_mex_path(PATH ${board}_${core_id})

mcux_add_include(
    INCLUDES neutron/include
)

# Register the Neutron libraries with their project-local library names.
mcux_add_library(
    neutron_driver
    LIBS neutron/lib/libNeutronDriver.a
)

mcux_add_library(
    neutron_firmware
    LIBS neutron/lib/libNeutronFirmware.a
)