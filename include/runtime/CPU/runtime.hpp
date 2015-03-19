#ifndef __RUNTIME_CPU_RUNTIME_HPP__
#define __RUNTIME_CPU_RUNTIME_HPP__
namespace CPURuntime{
	struct CPURunTimeEnv{
		typedef void* DeviceMemory;
		static inline DeviceMemory allocate(unsigned size)
		{
			return malloc(size);
		}
		static inline bool deallocate(DeviceMemory mem)
		{
			free(mem);
			return true;
		}
		static inline DeviceMemory get_null_pointer()
		{
			return NULL;
		}
		static inline bool is_valid_pointer(DeviceMemory ptr)
		{
			return (ptr != NULL);
		}
		
		static inline bool copy_from_host(DeviceMemory dest, void* sour, unsigned size)
		{
			return true;
		}
		
		static inline bool copy_to_host(void* dest, DeviceMemory sour, unsigned size)
		{
			return true;
		}
		
		template <typename Executable, typename Param>
		static bool execute(const Param& e, const typename Executable::Symbol& s)
		{
			int lx, ly, lz, hx, hy, hz;
			GetRange<typename Executable::Symbol, EmptyEnv>::get_range(s, lx, ly, lz, hx, hy, hz);
			/* Check if this formula is trying to do an infinity loop, just do the action on 0 */
			if(lx == INT_MIN || hx == INT_MAX) lx = 0, hx = 1;
			if(ly == INT_MIN || hy == INT_MAX) ly = 0, hy = 1;
			if(lz == INT_MIN || hz == INT_MAX) lz = 0, hz = 1;
			
			/* Load the arguments before the execution, otherwise compiler will load the arguments for each loop */
			Executable::CodeType::load_arg(e);
			
			/* do the actual ops */
			for(int z = lz; z < hz; z ++)
			    for(int y = ly; y < hy; y ++)
			        for(int x = lx; x < hx; x ++)
			            GetExecutor<DEVICE_TYPE_CPU>::template execute<Executable>(x, y, z);
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
