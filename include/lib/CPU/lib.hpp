#ifndef __LIB_CPU_LIB_HPP__
#define __LIB_CPU_LIB_HPP__
#include <algorithm>
#include <cmath>
using namespace SpatialOps;
namespace CPULib{
	
	/* This Lib actual generates the code */
	template <typename Symbol>
	struct ScalarLib;
	
	/* Get parameters */
	template <typename Executable>
	static inline const void* get_operand_1(const void* mem){return (char*)mem + (int)Executable::_1;}
	template <typename Executable>
	static inline const void* get_operand_2(const void* mem){return (char*)mem + (int)Executable::_2;}
	template <typename Executable>
	static inline const void* get_operand_3(const void* mem){return (char*)mem + (int)Executable::_3;}
	template <typename Executable>
	static inline const typename Executable::Self& get_self(const void* mem){return *(typename Executable::Self*)((char*)mem + (int)Executable::_self);}
	/* The glue */
	template <typename Expr, typename Executable>
	struct Lib
	{
		typedef StaticVar<Lib, Expr> arg;
		static inline void load_arg(const void* e)
		{
			arg::get() = get_self<Executable>(e);
		}
		static inline Expr eval(int x, int y, int z)
		{
			return arg::get();
		}
	};
	template <template <typename, typename, typename> class Symbol, typename Op1, typename Op2, typename Op3, typename Executable>
	struct Lib<Symbol<Op1, Op2, Op3>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename Executable::OP3Type::CodeType T3;
		typedef typename ExprTypeInfer<Symbol<Op1, Op2, Op3> >::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
			T2::load_arg(get_operand_2<Executable>(e));
			T3::load_arg(get_operand_3<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return ScalarLib<Symbol<Op1, Op2, Op3> >::template eval<RetType>(
			        T1::eval(x, y, z),
			        T2::eval(x, y, z),
			        T3::eval(x, y, z));
		}
	};
	template <template <typename, typename> class Symbol, typename left, typename right, typename Executable>
	struct Lib<Symbol<left, right>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Symbol<left, right> >::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
			T2::load_arg(get_operand_2<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return ScalarLib<Symbol<left, right> >::template eval<RetType>(
			        T1::eval(x, y, z),
			        T2::eval(x, y, z));
		}
	};
	
	template <template <typename> class Symbol, typename Operand, typename Executable>
	struct Lib<Symbol<Operand>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<Symbol<Operand> >::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return ScalarLib<Symbol<Operand> >::template eval<RetType>(T1::eval(x, y, z));
		}
	};
	
	template <typename T, typename Executable>
	struct Lib<Field<T>, Executable>{
		typedef T& RetType;
		typedef StaticVar<Lib, typename Executable::Self> arg;
		static inline void load_arg(const void* e)
		{
			arg::get() = get_self<Executable>(e);
		}
		static inline RetType eval(int x, int y, int z)
		{
			const typename Executable::Self& s = arg::get();
			return s._m[(x - s.lx) + (y - s.ly) * (s.hx - s.lx) + (z - s.lz) * (s.hx - s.lx) * (s.hy - s.ly)];
		}
	};
	
	template<typename Dir, typename Executable>
	struct Lib<REFSYM(coordinate)<Dir> , Executable>{
		typedef typename Executable::Symbol S;
		typedef int RetType;
		static inline void load_arg(const void* e)
		{
			
		}
		static inline RetType eval(int x, int y, int z)
		{
			return (int)S::X * x + (int)S::Y * y + (int)S::Z * z;
		}
	};
	
	template<int dx, int dy, int dz, typename Operand, typename Executable>
	struct Lib<REFSYM(shift)<Operand, dx, dy, dz>, Executable>{
		typedef typename Executable::Symbol S;
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(shift)<Operand, dx, dy, dz> >::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return T1::eval(x + (int)S::Dx,
			                y + (int)S::Dy,
			                z + (int)S::Dz);
		}
	};
	
	template<typename Operand, typename Executable>
	struct Lib<REFSYM(window)<Operand>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<REFSYM(window)<Operand> >::R RetType;
		typedef StaticVar<Lib, typename Executable::Self> arg;
		static inline void load_arg(const void* e)
		{
			arg::get() = get_self<Executable>(e);
			T1::load_arg(get_operand_1<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			const typename Executable::Self& s = arg::get() ;
			return
			((s.lx <= x && x < s.hx) &&
			 (s.ly <= y && y < s.hy) &&
			 (s.lz <= z && z < s.hz))?T1::eval(x, y, z):s.defval;
		}
	};
	template <typename T, typename Executable>
	struct Lib<LValueScalar<T>, Executable> {
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename ExprTypeInfer<LValueScalar<T> >::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return T1::eval(0,0,0);
		}
	};
	
	template <typename Var, typename Op1, typename Op2, typename Executable>
	struct Lib<REFSYM(binding)<Var, Op1, Op2>, Executable>{
		typedef typename Executable::OP1Type::CodeType T1;
		typedef typename Executable::OP2Type::CodeType T2;
		typedef typename ExprTypeInfer<Op2>::R RetType;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(get_operand_1<Executable>(e));
			T2::load_arg(get_operand_2<Executable>(e));
		}
		static inline RetType eval(int x, int y, int z)
		{
			return T2::eval(x,y,z);
		}
	};
	
	template <typename Var, typename Executable>
	struct Lib<REFSYM(ref)<Var>, Executable>{
		typedef typename ExprTypeInfer<REFSYM(ref)<Var> >::R RetType;
		static inline void load_arg(const void* e)
		{
		}
		static inline RetType eval(int x, int y, int z)
		{
			return Executable::Target::CodeType::eval(x,y,z);
		}
	};
	
	template <typename Operand, typename Annotation, typename Executable>
	struct Lib<REFSYM(annotation)<Operand, Annotation>, Executable>{
		typedef typename ExprTypeInfer<REFSYM(annotation)<Operand, Annotation> >::R RetType;
		typedef typename Executable::OP2Type::CodeType T1;
		static inline void load_arg(const void* e)
		{
			T1::load_arg(e);
		}
		static inline RetType eval(int x, int y, int z)
		{
			return T1::eval(x, y, z);
		}
	};
	
	#define CPU_SCALAR_RULE_3ARGS(sym, expr)\
	template<typename OP1, typename OP2, typename OP3>\
	struct ScalarLib<REFSYM(sym)<OP1, OP2, OP3> >{\
		typedef typename ExprTypeInfer<OP1>::R Operand1;\
		typedef typename ExprTypeInfer<OP2>::R Operand2;\
		typedef typename ExprTypeInfer<OP3>::R Operand3;\
		template <typename RetType>\
		static inline RetType eval(Operand1 _1, Operand2 _2, Operand3 _3)\
		{\
			return (expr);\
		}\
	}
	
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
	
	
	CPU_SCALAR_RULE_2ARGS(max, std::max((RetType)_1, (RetType)_2));
	CPU_SCALAR_RULE_2ARGS(min, std::min((RetType)_1, (RetType)_2));
	
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
		static inline typename Executable::CodeType::RetType execute(int x, int y, int z)
		{
			return Executable::CodeType::eval(x, y, z);
		}
	};
}
#endif
