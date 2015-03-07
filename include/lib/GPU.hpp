#ifndef __GPULIB_HPP__
#define __GPULIB_HPP__
#include <algorithm>
#include <cmath>
using namespace SpatialOps;
namespace GPULib{

	/* This Lib actuall generates the code */
	template <typename Symbol>
	struct ScalarLib;
	
	/* Get parameters */
	template <typename Executable>
	__device__ static inline const void* get_operand_1(const void* mem){return (char*)mem + (int)Executable::_1;}
	template <typename Executable>
	__device__ static inline const void* get_operand_2(const void* mem){return (char*)mem + (int)Executable::_2;}
	template <typename Executable>
	__device__ static inline const typename Executable::Self& get_self(const void* mem){return *(typename Executable::Self*)((char*)mem + (int)Executable::_self);}

	/* The glue */
	template <typename Expr, typename Executable> 
	struct Lib
	{
		__device__ static inline Expr eval(int x, int y, int z, const void* e)
		{
			return get_self<Executable>(e);
		}
	};
	
	template <template <typename, typename> class Symbol, typename left, typename right, typename Executable>
	struct Lib<Symbol<left, right>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Symbol<left, right> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return ScalarLib<Symbol<left, right> >::template eval<RetType>(
					T1::eval(x, y, z, get_operand_1<Executable>(e)), 
					T2::eval(x, y, z, get_operand_2<Executable>(e)));
		}
	};
	
	template <template <typename> class Symbol, typename Operand, typename Executable>
	struct Lib<Symbol<Operand>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<Symbol<Operand> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return ScalarLib<Symbol<Operand> >::template eval<RetType>(T1::eval(x, y, z, get_operand_1<Executable>(e))); 
		}
	};

	template <typename T, typename Executable>
	struct Lib<Field<T>, Executable>{
		typedef T& RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			const typename Executable::Self s = get_self<Executable>(e);
			return s._m[(x - s.lx) + (y - s.ly) * (s.hx - s.lx) + (z - s.lz) * (s.hx - s.lx) * (s.hy - s.ly)];
		}
	};

	template<typename Dir, typename Executable>
	struct Lib<REFSYM(coordinate)<Dir> , Executable>{
		typedef typename Executable::Symbol S;
		typedef int RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return (int)S::X * x + (int)S::Y * y + (int)S::Z * z;
		}
	};

	template<int dx, int dy, int dz, typename Operand, typename Executable>
	struct Lib<REFSYM(shift)<Operand, dx, dy, dz>, Executable>{
		typedef typename Executable::Symbol S;
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(shift)<Operand, dx, dy, dz> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return T1::eval(x + (int)S::Dx,
					        y + (int)S::Dy,
							z + (int)S::Dz,
							get_operand_1<Executable>(e));
		}
	};

	template<typename Operand, typename Executable>
	struct Lib<REFSYM(window)<Operand>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(window)<Operand> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{

			const typename Executable::Self s = get_self<Executable>(e);
			const void* _1 = get_operand_1<Executable>(e);
			return 
			((s.lx <= x && x < s.hx) &&
			 (s.ly <= y && y < s.hy) &&
			 (s.lz <= z && z < s.hz))?T1::eval(x, y, z, _1):s.defval;
		}
	};
	template <typename T, typename Executable>
	struct Lib<LValueScalar<T>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<LValueScalar<T> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return T1::eval(0,0,0,get_operand_1<Executable>(e));
		}
	};
	
	template <typename Var, typename Op1, typename Op2, typename Executable>
	struct Lib<REFSYM(binding)<Var, Op1, Op2>, Executable>{
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Op2>::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return T2::eval(x,y,z,get_operand_2<Executable>(e));
		}
	};

	template <typename Var, typename Executable>
	struct Lib<REFSYM(ref)<Var>, Executable>{
		typedef typename ExprTypeInfer<REFSYM(ref)<Var> >::R RetType;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			const void* target = (const void*)(((char*)e) + (int)Executable::Offset);
			return Executable::Target::CodeType::eval(x,y,z, target);
		}
	};
	
	template <typename Operand, typename Annotation, typename Executable>
	struct Lib<REFSYM(annotation)<Operand, Annotation>, Executable>{
		typedef typename ExprTypeInfer<REFSYM(annotation)<Operand, Annotation> >::R RetType;
		typedef typename Executable::OP1Type::CodeType T1;
		__device__ static inline RetType eval(int x, int y, int z, const void* e)
		{
			return T1::eval(x, y, z, e);
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

	GPU_SCALAR_RULE_2ARGS(max, (_1 > _2)?_1:_2);
	GPU_SCALAR_RULE_2ARGS(min, (_1 < _2)?_1:_2);

}
namespace SpatialOps{
	/* Export the Library */
	template <typename Expr, typename Executable>
	struct InvokeDeviceLibrary<DEVICE_TYPE_CUDA, Expr, Executable>
	{
		typedef GPULib::Lib<Expr, Executable> R;
	};
	/* Define the Executor */
	template <>
	struct GetExecutor<DEVICE_TYPE_CUDA>
	{
		template <typename Executable>
		__device__ static inline void execute(int x, int y, int z, const void* e)
		{
			typedef typename Executable::CodeType Code;
			Code::eval(x, y, z, e);
		}
	};
	/* Define the Preprocessor */
	struct annotation_gpu_reduction{};
	/* Ok, we want to change the LVScalar_A <<= LVScalar_A + SExpr to the GPU Reduction symbol */
	template <template <typename, typename> class BinOp, typename ResultT, typename SExpr>
	struct InvokeDevicePP<DEVICE_TYPE_CUDA, REFSYM(assign)<LValueScalar<ResultT>, BinOp<LValueScalar<ResultT> , SExpr> > >
	{
		typedef REFSYM(assign)<LValueScalar<ResultT>, BinOp<LValueScalar<ResultT>, SExpr> > AssignmentExpr;
		typedef symbol_annotation<AssignmentExpr, annotation_gpu_reduction> Annotated;
		typedef Annotated RetType;
		static inline Annotated preprocess(const AssignmentExpr& e)
		{
			return Annotated(e);
		}
	};
}
#endif
