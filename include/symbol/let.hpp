#ifndef __SYMBOL_LET_HPP__
#define __SYMBOL_LET_HPP__
#include <typeinfo>
namespace SpatialOps{
	/**
	 * @brief reference to a let-binding variable
	 **/
	template <typename var> 
	struct symbol_ref:public SymbolicExpression {
		typedef var Var;
		const inline char* name() const
		{
			return typeid(Var).name();
		}

	};
	template <typename var, typename Env>
	struct GetRange<symbol_ref<var>, Env>
	{
		typedef GetEnv<var, Env> EnvEntry;
		typedef typename EnvEntry::Expression Expr;
		typedef typename EnvEntry::Environ Environ;
		
		static void get_range(const symbol_ref<var>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int & hz)
		{
			typedef GetRange<Expr, Environ> RangeFinder;
			RangeFinder::get_range(EnvEntry::get(NULL), lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename var>
	struct TopLevelFlag<symbol_ref<var> >{
		static inline bool get(const symbol_ref<var>& e)
		{
			return e.top_level;
		}
		static inline void clear(const symbol_ref<var>& e)
		{
			((symbol_ref<var>*)&e)->top_level = 0;
		}
	};
	template <typename var>
	struct ExprTypeInfer<symbol_ref<var> >{
		typedef typename var::T R;
	};
	template<typename var>
	static inline symbol_ref<var> ref()
	{
		return symbol_ref<var>();
	}

	/* Symbol for binding */
	template <typename var, typename operand1, typename operand2>
	struct symbol_binding:public SymbolicExpression{
		typedef operand1 Operand_l;
		typedef operand2 Operand_r;
		typedef var Var;
		const inline char* name() const
		{
			return "Binding";
		}
		symbol_binding(const Operand_l& l, const Operand_r& r) :operand_l(l), operand_r(r){}
		Operand_l operand_l;
		Operand_r operand_r;
	};
	template <typename var, typename operand1, typename operand2>
	struct GetNumOperands<symbol_binding<var, operand1, operand2> >{
		enum{
			R = 2
		};
	};
	template <typename var, typename operand1, typename operand2, typename env>
	struct GetRange<symbol_binding<var, operand1, operand2>, env>
	{
		typedef AppendEnv<env, var, operand1, 0> NextEnv;
		static inline void get_range(const symbol_binding<var, operand1, operand2>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)
		{
			typedef GetRange<operand2, NextEnv> RangeFinder;
			NextEnv::get(&e.operand_l);
			RangeFinder::get_range(e.operand_r, lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename var, typename operand1, typename operand2>
	struct TopLevelFlag<symbol_binding<var, operand1, operand2> >{
		static inline bool get(const symbol_binding<var, operand1, operand2>& e)
		{
			return e.top_level;
		}
		static inline void clear(const symbol_binding<var, operand1, operand2>& e)
		{
			((symbol_binding<var, operand1, operand2>*)&e)->top_level = 0;
		}
	};
	template <typename var, typename operand1, typename operand2>
	struct ExprTypeInfer<symbol_binding<var, operand1, operand2> >{
		typedef typename ExprTypeInfer<operand2>::R R;
	};
	template <typename var, typename Op1, typename Op2>
	symbol_binding<var, Op1, Op2> let(const Op1& op1, const Op2& op2)
	{
		return symbol_binding<var, Op1, Op2>(op1, op2);
	}
	/*{
		struct VarX{typedef double T};
		let<VarX>(x, ref<VarX>() * ref<VarX>())

		because we know the symbolic expression type of 
		each binding-variable. So that we can retrieve 
		the reference by it's type

		So the list needs to memorize 
			1. the BindingId
			2. the variable type


	}*/

}
#endif
