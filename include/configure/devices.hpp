/**
 * @file devices.hpp
 * @brief Define the available device and assign device index to different type of device
 *        The executor will attempt to execute the formula from the device with the
 *        minimum device number.
 *        So if some device is preferred, you should put it in the beginning of the list
 **/
#ifndef __CONFIGURE_H__
#define __CONFIGURE_H__

/**
 * @brief Define the device list
 **/
enum DeviceType{
	#ifdef __CUDACC__
	DEVICE_TYPE_CUDA,   /*!< Type for GPUs supports CUDA */
	#endif /*__CUDACC__*/
	
	DEVICE_TYPE_CPU,    /*!< General CPUs back-end */
	
	NUM_DEVICE_TYPES
	/* NOTICE: ANY DEVICE DECLEARATION BEYOND THIS POINT WILL HAVE NO EFFECT */
};

#endif /*__CONFIGURE_H__*/
