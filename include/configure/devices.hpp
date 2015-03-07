#ifndef __CONFIGURE_H__
#define __CONFIGURE_H__

/** @brief Device Type List */
enum DeviceType{
	#ifdef __CUDACC__
	DEVICE_TYPE_CUDA,   /*!< Type for GPUs supports CUDA */
	#endif /*__CUDACC__*/
	
	DEVICE_TYPE_CPU,    /*!< General CPUs backend */
	
	NUM_DEVICE_TYPES
	/* NOTICE: ANY DEVICE DECLEARATION BEYOND THIS POINT WILL HAVE NO EFFECT */
};

#endif /*__CONFIGURE_H__*/
