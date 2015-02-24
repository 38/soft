#include <map>
#ifndef __RUNTIME_CPU_HPP__
#define __RUNTIME_CPU_HPP__
namespace CPURuntime{
	struct MemoryBlock{
		void* mem;
		MemoryBlock(size_t size){
			mem = malloc(size);
		}
		~MemoryBlock(){
			free(mem);
		}
	};
	static std::map<int, MemoryBlock*> _mem_map;
	struct CPURunTimeEnv{
		static inline void* allocate(int token, size_t size)
		{
			if(_mem_map.find(token) == _mem_map.end())
				_mem_map[token] = new MemoryBlock(size);
			if(_mem_map[token]) return _mem_map[token]->mem;
			return NULL;
		}
		static inline int deallocate(int token)
		{
			if(_mem_map.find(token) == _mem_map.end()) return 0;
			delete(_mem_map[token]);
			_mem_map.erase(_mem_map.find(token));
			return 0;
		}
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
