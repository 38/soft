#ifndef __SYMBOL_LVALUE_HPP__
#define __SYMBOL_LVALUE_HPP__
#include <symbol/field.hpp>
namespace SpatialOps{
	template <typename T>
	struct LValueScalar{
		typedef Field<T> Operand;
		Operand operand;
		inline LValueScalar():operand(0,0,0,1,1,1){}
		inline LValueScalar(const LValueScalar& l):operand(l.operand){}
		inline const char* name() const
		{
			return "LValueScalar";
		}
	};
	template <typename T>
	struct GetNumOperands<LValueScalar<T> >{
		enum{
			R = 1
		};
	};
	template <typename T, typename Env>
	struct GetRange<LValueScalar<T>, Env >{
		static inline void get_range(const LValueScalar<T>& e, int& lx, int& ly, int &lz, int& hx, int& hy, int& hz)
		{
			lx = ly = lz = INT_MIN;
			hx = hy = hz = INT_MAX;
		}
	};
	template <typename T>
	struct ExprTypeInfer<LValueScalar<T> >{
		typedef T& R;
	};
}
#endif /*  __SYMBOL_LVALUE_HPP__ */
