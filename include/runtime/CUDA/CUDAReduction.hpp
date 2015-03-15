#ifndef __RUNTIME_CUDA_REDUCTION_HPP__
#define __RUNTIME_CUDA_REDUCTION_HPP__
namespace CUDAReduction{
	struct FieldIndex{
		int x, y, z;
		__device__ inline FieldIndex(int ix, int iy, int iz):x(ix), y(iy), z(iz){}
	};
	__device__ static inline FieldIndex convert_index(int idx, int lx, int ly, int lz, int dx, int dy, int dz)
	{
		return FieldIndex(lx + (idx % dx), ly + (idx / dx) % dy, lz + (idx / dx / dy));
	}
	/*
	 *  In this kernel, we assume that the gridDim = (1,1,1)
	 *  And blockSize = gridDim.x in the previous kernel function / 2, so that the gridSize should be
	 *  Smaller than 1024
	 */
	template<int blockSize, typename LVType, typename BufferType, typename OpType>
	__global__ static void gpu_combination_kernel(BufferType* in, LVType* out)
	{
		extern __shared__ BufferType sdata[];
		unsigned tid = threadIdx.x;
		sdata[tid] = OpType::eval(in[tid], in[tid + blockSize]);
		__syncthreads();
		if(blockSize >= 512 && tid < 256){sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 256]); __syncthreads();}
		if(blockSize >= 256 && tid < 128){sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 128]); __syncthreads();}
		if(blockSize >= 128 && tid < 64) {sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 64]); __syncthreads();}
		if(tid < 32){
			if(blockSize >= 64 && tid < 32) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 32]);
			if(blockSize >= 32 && tid < 16) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 16]);
			if(blockSize >= 16 && tid < 8) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 8]);
			if(blockSize >= 8 && tid < 4) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 4]);
			if(blockSize >= 4 && tid < 2) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 2]);
			if(blockSize >= 2 && tid < 1) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 1]);
		}
		if(tid == 0) *out = sdata[0];
	}
	template<int blockSize, typename RVExecutable, typename RVParamType, typename LVType, typename BufferType, typename OpType>
	__global__ static void gpu_reduction_kernel(CUDARuntime::CUDAParamWrap<RVParamType> rvdata,
	                                            BufferType* buffer, LVType* dest,
	                                            int lx, int ly, int lz, int hx, int hy, int hz)
	{
		extern __shared__ BufferType sdata[];
		typedef GetExecutor<DEVICE_TYPE_CUDA> CUDAExecutor;
		typedef typename RVExecutable::CodeType::RetType ResultType;
		unsigned tid = threadIdx.x;
		unsigned idx = blockIdx.x * blockSize * 2 + threadIdx.x;
		unsigned gridSize = gridDim.x * blockSize * 2;
		const unsigned dx = hx - lx, dy = hy - ly, dz = hz - lz;
		unsigned bound = dx * dy * dz;
		for(sdata[tid] = *dest; idx + blockSize < bound; idx += gridSize)
		{
			FieldIndex p0 = convert_index(idx, lx, ly, lz, dx, dy, dz);
			FieldIndex p1 = convert_index(idx + blockSize, lx, ly, lz, dx, dy, dz);
			ResultType v0 = CUDAExecutor::execute<RVExecutable>(p0.x, p0.y, p0.z, &rvdata);
			ResultType v1 = CUDAExecutor::execute<RVExecutable>(p1.x, p1.y, p1.z, &rvdata);
			sdata[tid] = OpType::eval(OpType::eval(sdata[tid], v0), v1);
		}
		if(idx < bound)
		{
			FieldIndex p = convert_index(idx, lx, ly, lz, dx, dy, dz);
			sdata[tid] = OpType::eval(sdata[tid], CUDAExecutor::execute<Executable>(p.x, p.y, p.z, &rvdata));
		}
		__syncthreads();
		if(blockSize >= 512 && tid < 256){sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 256]); __syncthreads();}
		if(blockSize >= 256 && tid < 128){sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 128]); __syncthreads();}
		if(blockSize >= 128 && tid < 64) {sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 64]); __syncthreads();}
		if(tid < 32)
		{
			if(blockSize >= 64 && tid < 32) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 32]);
			if(blockSize >= 32 && tid < 16) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 16]);
			if(blockSize >= 16 && tid < 8) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 8]);
			if(blockSize >= 8 && tid < 4) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 4]);
			if(blockSize >= 4 && tid < 2) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 2]);
			if(blockSize >= 2 && tid < 1) sdata[tid] = OpType::eval(sdata[tid], sdata[tid + 1]);
		}
		if(tid == 0) buffer[blockIdx.x] = sdata[0];
	}
	template<int blockSize, int gridSize, typename RVExecutable, typename RVParamType, typename LVType, typename BufferType, typename OpType>
	struct ReductionKernel {
		typedef ReductionKernel<blockSize / 2, gridSize, RVExecutable, RVParamType, LVType, BufferType, OpType> NextKernel;
		static inline void launch(const RVParamType& rvdata, BufferType* buf, LVType* dest,
		                          int lx, int ly, int lz, int hx, int hy, int hz,
		                          cudaStream_t stream)
		{
			int targetBS = (hx - hx) * (hy - hy) * (hz - hz) / (gridSize * 2);
			if(targetBS > blockSize)
			{
				size_t r_sm_size = sizeof(BufferType) * blockSize;
				size_t c_sm_size = sizeof(BufferType) * gridSize / 2;
				gpu_reduction_kernel<blockSize, RVExecutable, RVParamType, LVType, BufferType, OpType>
				                    <<<gridSize, blockSize, r_sm_size, stream>>>(rvdata, buf, dest, lx, ly, lz, hx, hy, hz);
				gpu_combination_kernel<gridSize/2, LVType, BufferType, OpType>
				                      <<<1, gridSize/2, c_sm_size, stream>>>(buf, dest);
			}
			else
			    NextKernel::launch(rvdata, buf, dest, lx, ly, lz, hx, hy, hz, stream);
		}
	};
	template<int gridSize, typename RVExecutable, typename RVParamType, typename LVType, typename BufferType, typename OpType>
	struct ReductionKernel<1, gridSize, RVExecutable, RVParamType, LVType, BufferType, OpType>{
		static inline void launch(const RVParamType& rvdata, BufferType* buf, LVType* dest,
		                          int lx, int ly, int lz, int hx, int hy, int hz,
		                          cudaStream_t stream)
		{
			
			gpu_reduction_kernel<1, RVExecutable, RVParamType, LVType, LVType, OpType>
			                    <<<gridSize, 1, sizeof(LVType) * gridSize, stream>>>(rvdata, dest, dest, lx, ly, lz, hx, hy, hz);
		}
	};
};
#endif
