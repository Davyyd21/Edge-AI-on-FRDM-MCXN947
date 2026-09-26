/*
 * Based on the NXP Multiple Person Detection OV7670 demo.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef OV7670_DEMO_H
#define OV7670_DEMO_H

#ifdef __cplusplus 
extern "C" { // This line checks if the code is being compiled in a C++ environment. If so, it uses extern "C" to indicate that the following code should use C linkage, preventing name mangling and allowing C++ code to link with C functions.
#endif

void DEMO_OV7670_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* OV7670_DEMO_H */
