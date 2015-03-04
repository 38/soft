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
	/** @brief Get the preprocessor for this device */
	template <int DeviceType, typename SExpr> struct InvokeDevicePP;

	/** Define default PP */
	template <int DeviceType, typename SExpr> 
	struct InvokeDevicePP{
		static inline const SExpr& preprocess(const SExpr& e)
		{
			return e;
		}
	};
}
#endif
