#ifndef __DEVICE_MANAGER_HPP__
#define __DEVICE_MANAGER_HPP__
namespace SpatialOps{

	/* template <typename Expr, typename Exec> Library */
	/** @brief Get the runtime environment for this device */
	template <int DeviceType> struct GetDeviceRuntimeEnv;  
	/** @brief Get the linker library for this device */
	template <int DeviceType, typename Expr, typename Executable> struct InvokeDeviceLibrary;
	/** @brief Get the execute method for this device */
	template <int DeviceType> struct GetExecutor;

	template <int DeviceType = 0> 
	struct InvokeDeviceDestructor{
		static inline void notify(int id){
			GetDeviceRuntimeEnv<DeviceType>::R::deallocate(id);
			InvokeDeviceDestructor<DeviceType + 1>::notify(id);
		}
	};
	template <>
	struct InvokeDeviceDestructor<NUM_DEVICE_TYPES>{
		static inline void notify(int id){}
	};

}
#endif
