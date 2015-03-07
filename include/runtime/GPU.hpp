#include <map>
#ifndef __RUNTIME_GPU_HPP__
#define __RUNTIME_GPU_HPP__
namespace GPURuntime{
	/* This Wrap type can make all runtime data copy to the GPU when launching kernel function */
	template <typename Param>
	struct GPUParamWrap{
		char data[sizeof(Param)];
	};
	template <typename Executable, typename ParamType>
	__global__ void execute_kernel(const GPUParamWrap<ParamType> e, int lx, int ly, int lz, int hx, int hy, int hz)
	{
		int x = lx + threadIdx.x + blockIdx.x * blockDim.x;
		int y = ly + threadIdx.y + blockIdx.y * blockDim.y;
		int z = lz + threadIdx.z + blockIdx.z * blockDim.z;
		
		if(x < hx && y < hy && z < hz) GetExecutor<DEVICE_TYPE_CUDA>::execute<Executable>(x, y, z, &e);
	}
	struct GPURunTimeEnv{
		typedef void* DeviceMemory;
		static inline void* allocate(unsigned size)
		{
			void* ptr;
			if(cudaSuccess != cudaMalloc(&ptr, size)) return NULL;
			return ptr;
		}
		static inline void deallocate(DeviceMemory mem)
		{
			cudaFree(mem);
		}
		static inline void copy_from_host(DeviceMemory dest, void* sour, unsigned size)
		{
			cudaMemcpy(dest, sour, size, cudaMemcpyHostToDevice);
		}
		static inline void copy_to_host(void* dest, DeviceMemory sour, unsigned size)
		{
			cudaMemcpy(dest, sour, size, cudaMemcpyDeviceToHost);
		}
		static inline DeviceMemory get_null_pointer()
		{
			return NULL;
		}
		static inline bool is_valid_pointer(DeviceMemory ptr)
		{
			return (ptr != NULL);
		}
		static inline int ceil(int a, int b)
		{
			return (a + b - 1) / b;
		}
		template <typename Symbolic, typename Executable, typename ParamType>
		struct Executor{
			static inline bool execute(const ParamType& e, const typename Executable::Symbol s)
			{
				int lx, ly, lz, hx, hy, hz;
				GetRange<typename Executable::Symbol, EmptyEnv>::get_range(s, lx, ly, lz, hx, hy, hz);
				/* avoid inf-loop */
				if(lx == INT_MIN || hx == INT_MAX) lx = 0, hx = 1;
				if(ly == INT_MIN || hy == INT_MAX) ly = 0, hy = 1;
				if(lz == INT_MIN || hz == INT_MAX) lz = 0, hz = 1;
				int blockX = 8;
				int blockY = 8;
				int blockZ = 8;
				if(hx - lx < 8) blockX = hx - lx;
				if(hy - ly < 8) blockY = hy - lx;
				if(hz - lz < 8) blockZ = hz - lz;
				if(blockX == 0 || blockY == 0 || blockZ == 0) return true;
				dim3 block_dim(blockX, blockY, blockZ);
				dim3 grid_dim( ceil(hx - lx, blockX),
				               ceil(hy - ly, blockY),
				               ceil(hz - lz, blockZ));
				
				execute_kernel<Executable><<<block_dim, grid_dim, 0, 0>>>(*(GPUParamWrap<ParamType>*)&e, lx, ly, lz, hx, hy, hz);
				return true;
			}
		};
		/*
		 * TODO for GPU REDUCTION
		template <typename Expr, typename Executable, typename ParamType>
		struct Executor<symbol_annotation<Expr, gpu_reduction_annotation>, Executable, ParamType>{
		    static inline bool execute(const ParamType& e, const typename Executable::Symbol s)
		    {
		        if(gpu_reduction_annotation::Valid == 0 ||
		           
	        }
		
	    };*/
		template<typename Executable, typename ParamType>
		static inline bool execute(const ParamType& e, const typename Executable::Symbol& s)
		{
			return Executor<typename Executable::Symbol, Executable, ParamType>::execute(e, s);
		}
	};
}
namespace SpatialOps{
	template <>
	struct GetDeviceRuntimeEnv<DEVICE_TYPE_CUDA>{
		typedef GPURuntime::GPURunTimeEnv R;
	};
}
#endif
