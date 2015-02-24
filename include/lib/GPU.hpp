#ifndef __GPULIB_HPP__
#define __GPULIB_HPP__
#include <algorithm>
#include <cmath>
using namespace SpatialOps;
namespace GPULib{

	/* This Lib actuall generates the code */
	template <typename Symbol>
	struct ScalarLib;

	/* The glue */
	template <typename Expr, typename Executable> 
	struct Lib
	{
		__device__ static inline Expr eval(int x, int y, int z, const Executable& e)
		{
			return e._s;
		}
	};
	
	template <template <typename, typename> class Symbol, typename left, typename right, typename Executable>
	struct Lib<Symbol<left, right>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Symbol<left, right> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return ScalarLib<Symbol<left, right> >::template eval<RetType>(
					T1::eval(x, y, z, e._1), 
					T2::eval(x, y, z, e._2));
		}
	};
	
	template <template <typename> class Symbol, typename Operand, typename Executable>
	struct Lib<Symbol<Operand>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<Symbol<Operand> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return ScalarLib<Symbol<Operand> >::template eval<RetType>(T1::eval(x, y, z, e._1)); 
		}
	};

	template <typename T, typename Executable>
	struct Lib<Field<T>, Executable>{
		typedef T& RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return e._m[x + y * (e.hx - e.lx) + z * (e.hx - e.lx) * (e.hy - e.ly)];
		}
	};

	template<typename Dir, typename Executable>
	struct Lib<REFSYM(coordinate)<Dir> , Executable>{
		typedef typename Executable::Symbol S;
		typedef int RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return (int)S::X * x + (int)S::Y * y + (int)S::Z * z;
		}
	};

	template<int dx, int dy, int dz, typename Operand, typename Executable>
	struct Lib<REFSYM(shift)<Operand, dx, dy, dz>, Executable>{
		typedef typename Executable::Symbol S;
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(shift)<Operand, dx, dy, dz> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return T1::eval(x + (int)S::Dx,
					        y + (int)S::Dy,
							z + (int)S::Dz,
							e._1);
		}
	};

	template<typename Operand, typename Executable>
	struct Lib<REFSYM(window)<Operand>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(window)<Operand> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return 
			((e._s.low[0] <= x && x < e._s.high[0]) &&
			 (e._s.low[1] <= y && y < e._s.high[1]) &&
			 (e._s.low[2] <= z && z < e._s.high[2]))?T1::eval(x, y, z, e._1):e._s.defval;
		}
	};



#define GPU_SCALAR_RULE_2ARGS(sym, expr)\
	template<typename left, typename right>\
	struct ScalarLib<REFSYM(sym)<left, right> >{\
		typedef typename ExprTypeInfer<left>::R LType;\
		typedef typename ExprTypeInfer<right>::R RType;\
		template <typename RetType>\
		__device__ static inline RetType eval(LType _1, RType _2)\
		{\
			return (expr);\
		}\
	}

#define GPU_SCALAR_RULE_1ARG(sym, expr)\
	template<typename Operand>\
	struct ScalarLib<REFSYM(sym)<Operand> >{\
		typedef typename ExprTypeInfer<Operand>::R OPType;\
		template <typename RetType, typename OPType>\
		__device__ static inline RetType eval(OPType _1)\
		{\
			return (expr);\
		}\
	}

	/* Basic Operators */
	GPU_SCALAR_RULE_2ARGS(add, _1 + _2);
	GPU_SCALAR_RULE_2ARGS(sub, _1 - _2);
	GPU_SCALAR_RULE_2ARGS(mul, _1 * _2);
	GPU_SCALAR_RULE_2ARGS(div, _1 / _2);
	GPU_SCALAR_RULE_2ARGS(and, _1 & _2);
	GPU_SCALAR_RULE_2ARGS(or, _1 | _2);
	GPU_SCALAR_RULE_2ARGS(xor, _1 ^ _2);
	GPU_SCALAR_RULE_2ARGS(assign, _1 = _2);
	GPU_SCALAR_RULE_1ARG (neg, -_1);
	GPU_SCALAR_RULE_1ARG (not, ~_1);
	
	GPU_SCALAR_RULE_2ARGS(lt, _1 < _2);
	GPU_SCALAR_RULE_2ARGS(gt, _1 > _2);
	GPU_SCALAR_RULE_2ARGS(eq, _1 == _2);
	GPU_SCALAR_RULE_2ARGS(le, _1 <= _2);
	GPU_SCALAR_RULE_2ARGS(ge, _1 >= _2);
	GPU_SCALAR_RULE_2ARGS(ne, _1 != _2);

	/* Math Functions */
	GPU_SCALAR_RULE_1ARG(sin, std::sin((double)_1));
	GPU_SCALAR_RULE_1ARG(cos, std::cos((double)_1));
	GPU_SCALAR_RULE_1ARG(tan, std::tan((double)_1));
	GPU_SCALAR_RULE_1ARG(asin, std::asin((double)_1));
	GPU_SCALAR_RULE_1ARG(acos, std::acos((double)_1));
	GPU_SCALAR_RULE_1ARG(atan, std::atan((double)_1));
	GPU_SCALAR_RULE_1ARG(abs, std::abs((double)_1));
	GPU_SCALAR_RULE_1ARG(exp, std::exp((double)_1));
	GPU_SCALAR_RULE_1ARG(log, std::log((double)_1));
	GPU_SCALAR_RULE_1ARG(sqrt, std::sqrt((double)_1));

}
namespace SpatialOps{
	/* Export the Library */
	template <typename Expr, typename Executable>
	struct InvokeDeviceLibrary<DEVICE_TYPE_CUDA, Expr, Executable>
	{
		typedef GPULib::Lib<Expr, Executable> R;
	};
	template <>
	struct GetExecutor<DEVICE_TYPE_CUDA>
	{
		template <typename Executable>
		__device__ static inline void execute(int x, int y, int z, const Executable& e)
		{
			typedef typename Executable::CodeType Code;
			Code::eval(x, y, z, e);
		}
	};
}
#endif
