#include <map>
#ifndef __RUNTIME_CPU_HPP__
#define __RUNTIME_CPU_HPP__
namespace CPURuntime{
	struct CPURunTimeEnv{
		typedef void* DeviceMemory; 
		static inline void* allocate(unsigned size)
		{
			return malloc(size);
		}
		static inline void deallocate(DeviceMemory mem)
		{
			free(mem);
		}
		static inline DeviceMemory get_null_pointer()
		{
			return NULL;
		}
		static inline bool is_valid_pointer(DeviceMemory ptr)
		{
			return (ptr != NULL);
		}
		static inline void copy_from_host(DeviceMemory dest, void* sour, unsigned size) {}
		static inline void copy_to_host(void* dest, DeviceMemory sour, unsigned size){}

		template <typename Executable> struct AnnotationHandler{
			template<typename T> static inline bool run(const T&a, const typename T::Symbol&b){}
		};
		template <typename Executable>
		static bool execute(const Executable& e, const typename Executable::Symbol& s)
		{
			/* you can add annotation handler here 
			 * AnnotationHandler<typename Executable::Symbol>::run(e, s);
			 */
			int lx, ly, lz, hx, hy, hz;
			GetRange<typename Executable::Symbol>::get_range(s, lx, ly, lz, hx, hy, hz);
			/* Check if this formular is trying to do an infinity loop, just do the action on 0 */
			if(lx == INT_MIN || hx == INT_MAX) lx = 0, hx = 1;
			if(ly == INT_MIN || hy == INT_MAX) ly = 0, hy = 1;
			if(lz == INT_MIN || hz == INT_MAX) lz = 0, hz = 1;
			/* do the actual ops */ 
			for(int x = lx; x < hx; x ++)
				for(int y = ly; y < hy; y ++)
					for(int z = lz; z < hz; z ++)
						GetExecutor<DEVICE_TYPE_CPU>::execute(x, y, z, e);
			return true;
		}
	};
}
namespace SpatialOps{
	template <>
	struct GetDeviceRuntimeEnv<DEVICE_TYPE_CPU>{
		typedef CPURuntime::CPURunTimeEnv R;
	};
}
#endif
