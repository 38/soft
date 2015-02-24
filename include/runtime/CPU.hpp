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
		template <typename Executable>
		static bool execute(const Executable& e)
		{
			int lx, ly, lz, hx, hy, hz;
			GetRange<typename Executable::Symbol>::get_range(e._s, lx, ly, lz, hx, hy, hz);
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
