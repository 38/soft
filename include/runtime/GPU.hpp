#include <map>
#ifndef __RUNTIME_GPU_HPP__
#define __RUNTIME_GPU_HPP__
namespace GPURuntime{
	struct MemoryBlock{
		void* mem;
		MemoryBlock(size_t size){
			mem = cudaMalloc(size);
		}
		~MemoryBlock(){
			cudaFree(mem);
		}
	};
	template <typename Executable>
	__kernel__ static void execute_kernel(Executable e, int lx, int ly, int lz, int hx, int hy, int hz)
	{
		int x = lx + threadIdx.x + blockIdx.x * blockDim.x;
		int y = ly + threadIdx.y + blockIdx.y * blockDim.y;
		int z = lz + threadIdx.z + blockIdx.z * blockDim.z;
		if(x < hx && y < hy && z < hz) GetExecutor<DEVICE_TYPE_GPU>::execute(x, y, z, e);
	}
	static std::map<int, MemoryBlock*> _mem_map;
	struct GPURunTimeEnv{
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
		static inline ceil(int a, int b)
		{
			return (a + b - 1) / b;
		}
		template <typename Executable>
		static bool execute(const Executable& e)
		{
			int lx, ly, lz, hx, hy, hz;
			dim3 block_dim(8,8,8);
			dim3 grid_dim( ceil(hx - lx, 8),
					       ceil(hy - ly, 8),
						   ceil(hz - lz, 8));

			execute_kernel<<<block_dim, grid_dim>>>(e, lx, ly, lz, hx, hy, hz);
			return true;
		}
	};
}
namespace SpatialOps{
	template <>
	struct GetDeviceRuntimeEnv<DEVICE_TYPE_GPU>{
		typedef GPURuntime::GPURunTimeEnv R;
	};
}
#endif
