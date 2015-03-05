#ifndef __CPULIB_HPP__
#define __CPULIB_HPP__
#include <algorithm>
#include <cmath>
using namespace SpatialOps;
namespace CPULib{

	/* This Lib actuall generates the code */
	template <typename Symbol>
	struct ScalarLib;

	/* The glue */
	template <typename Expr, typename Executable> 
	struct Lib
	{
		static inline Expr eval(int x, int y, int z, const Executable& e)
		{
			return e._s;
		}
	};
	
	template <template <typename, typename> class Symbol, typename left, typename right, typename Executable>
	struct Lib<Symbol<left, right>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Symbol<left, right> >::R RetType;
		static inline RetType eval(int x, int y, int z, const Executable& e)
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
		static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return ScalarLib<Symbol<Operand> >::template eval<RetType>(T1::eval(x, y, z, e._1)); 
		}
	};

	template <typename T, typename Executable>
	struct Lib<Field<T>, Executable>{
		typedef T& RetType;
		static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return e._m[(x - e.lx) + (y - e.ly) * (e.hx - e.lx) + (z - e.lz) * (e.hx - e.lx) * (e.hy - e.ly)];
		}
	};

	template<typename Dir, typename Executable>
	struct Lib<REFSYM(coordinate)<Dir> , Executable>{
		typedef typename Executable::Symbol S;
		typedef int RetType;
		static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return (int)S::X * x + (int)S::Y * y + (int)S::Z * z;
		}
	};

	template<int dx, int dy, int dz, typename Operand, typename Executable>
	struct Lib<REFSYM(shift)<Operand, dx, dy, dz>, Executable>{
		typedef typename Executable::Symbol S;
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(shift)<Operand, dx, dy, dz> >::R RetType;
		static inline RetType eval(int x, int y, int z, const Executable& e)
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
		static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return 
			((e.lx <= x && x < e.hx) &&
			 (e.ly <= y && y < e.hy) &&
			 (e.lz <= z && z < e.hz))?T1::eval(x, y, z, e._1):e.defval;
		}
	};
	template <typename T, typename Executable>
	struct Lib<LValueScalar<T>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<LValueScalar<T> >::R RetType;
		static inline RetType eval(int x, int y, int z, const Executable& e)
		{
			return T1::eval(0,0,0,e._1);
		}
	};

#define CPU_SCALAR_RULE_2ARGS(sym, expr)\
	template<typename left, typename right>\
	struct ScalarLib<REFSYM(sym)<left, right> >{\
		typedef typename ExprTypeInfer<left>::R LType;\
		typedef typename ExprTypeInfer<right>::R RType;\
		template <typename RetType>\
		static inline RetType eval(LType _1, RType _2)\
		{\
			return (expr);\
		}\
	}

#define CPU_SCALAR_RULE_1ARG(sym, expr)\
	template<typename Operand>\
	struct ScalarLib<REFSYM(sym)<Operand> >{\
		typedef typename ExprTypeInfer<Operand>::R OPType;\
		template <typename RetType, typename OPType>\
		static inline RetType eval(OPType _1)\
		{\
			return (expr);\
		}\
	}

	/* Basic Operators */
	CPU_SCALAR_RULE_2ARGS(add, _1 + _2);
	CPU_SCALAR_RULE_2ARGS(sub, _1 - _2);
	CPU_SCALAR_RULE_2ARGS(mul, _1 * _2);
	CPU_SCALAR_RULE_2ARGS(div, _1 / _2);
	CPU_SCALAR_RULE_2ARGS(and, _1 & _2);
	CPU_SCALAR_RULE_2ARGS(or, _1 | _2);
	CPU_SCALAR_RULE_2ARGS(xor, _1 ^ _2);
	CPU_SCALAR_RULE_2ARGS(assign, _1 = _2);
	CPU_SCALAR_RULE_1ARG (neg, -_1);
	CPU_SCALAR_RULE_1ARG (not, ~_1);
	
	CPU_SCALAR_RULE_2ARGS(lt, _1 < _2);
	CPU_SCALAR_RULE_2ARGS(gt, _1 > _2);
	CPU_SCALAR_RULE_2ARGS(eq, _1 == _2);
	CPU_SCALAR_RULE_2ARGS(le, _1 <= _2);
	CPU_SCALAR_RULE_2ARGS(ge, _1 >= _2);
	CPU_SCALAR_RULE_2ARGS(ne, _1 != _2);

	/* Math Functions */
	CPU_SCALAR_RULE_1ARG(sin, std::sin(_1));
	CPU_SCALAR_RULE_1ARG(cos, std::cos(_1));
	CPU_SCALAR_RULE_1ARG(tan, std::tan(_1));
	CPU_SCALAR_RULE_1ARG(asin, std::asin(_1));
	CPU_SCALAR_RULE_1ARG(acos, std::acos(_1));
	CPU_SCALAR_RULE_1ARG(atan, std::atan(_1));
	CPU_SCALAR_RULE_1ARG(abs, std::abs(_1));
	CPU_SCALAR_RULE_1ARG(exp, std::exp(_1));
	CPU_SCALAR_RULE_1ARG(log, std::log(_1));
	CPU_SCALAR_RULE_1ARG(sqrt, std::sqrt(_1));


	CPU_SCALAR_RULE_2ARGS(max, std::max(_1, _2));
	CPU_SCALAR_RULE_2ARGS(min, std::min(_1, _2));

}
namespace SpatialOps{
	/* Export the Library */
	template <typename Expr, typename Executable>
	struct InvokeDeviceLibrary<DEVICE_TYPE_CPU, Expr, Executable>
	{
		typedef CPULib::Lib<Expr, Executable> R;
	};
	template <>
	struct GetExecutor<DEVICE_TYPE_CPU>
	{
		template <typename Executable>
		static inline void execute(int x, int y, int z, const Executable& e)
		{
			typedef typename Executable::CodeType Code;
			Code::eval(x, y, z, e);
		}
	};
}
#endif
