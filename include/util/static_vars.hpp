#ifndef __STATIC_VARS_HPP__
#define __STATIC_VARS_HPP__
namespace SpatialOps{
	/**
	 * @brief This class is used to build a static field that can hold without a source code file
	 **/
	template <typename Parent, typename T>
	struct StaticVar{
		static inline T& get()
		{
			static T t;
			return t;
		}
	};
}
#endif /*__STATIC_VARS_HPP__*/
