#ifndef __BASICOPS_HPP__
#define __BASICOPS_HPP__
namespace SpatialOps{

	/* define the symbols */
	DEF_SYMBOL_2ARGS(add, false);
	DEF_SYMBOL_2ARGS(sub, false);
	DEF_SYMBOL_2ARGS(mul, false);
	DEF_SYMBOL_2ARGS(div, false);
	DEF_SYMBOL_2ARGS(and, false);
	DEF_SYMBOL_2ARGS(or, false);
	DEF_SYMBOL_2ARGS(xor, false);
	DEF_SYMBOL_2ARGS(assign, true);
	DEF_SYMBOL_1ARG(neg, false);
	DEF_SYMBOL_1ARG(not, false);

	DEF_SYMBOL_2ARGS(lt, false);
	DEF_SYMBOL_2ARGS(gt, false);
	DEF_SYMBOL_2ARGS(eq, false);
	DEF_SYMBOL_2ARGS(le, false);
	DEF_SYMBOL_2ARGS(ge, false);
	DEF_SYMBOL_2ARGS(ne, false);
	
	/* Define the operators */
	DEF_DEFAULT_OPERAND_2ARGS(add, +);
	DEF_DEFAULT_OPERAND_2ARGS(sub, -);
	DEF_DEFAULT_OPERAND_2ARGS(mul, *);
	DEF_DEFAULT_OPERAND_2ARGS(div, /);
	DEF_DEFAULT_OPERAND_2ARGS(and, &&);
	DEF_DEFAULT_OPERAND_2ARGS(and, ||);
	DEF_DEFAULT_OPERAND_2ARGS(and, ^);
	DEF_DEFAULT_OPERAND_2ARGS(assign, <<=);

	DEF_DEFAULT_OPERAND_2ARGS(lt, <);
	DEF_DEFAULT_OPERAND_2ARGS(gt, >);
	DEF_DEFAULT_OPERAND_2ARGS(eq, ==);
	DEF_DEFAULT_OPERAND_2ARGS(le, <=);
	DEF_DEFAULT_OPERAND_2ARGS(ge, >=);
	DEF_DEFAULT_OPERAND_2ARGS(ne, !=);

	/* Define the real uniary operators */
	template <typename TOperand>
	REFSYM(neg)<TOperand> operator -(const TOperand& operand)
	{
		TopLevelFlag<TOperand>::clear(operand);
		typedef REFSYM(neg)<TOperand> RetType;
		return RetType(operand);
	}

	template <typename TOperand>
	REFSYM(not)<TOperand> operator ~(const TOperand& operand)
	{
		TopLevelFlag<TOperand>::clear(operand);
		typedef REFSYM(not)<TOperand> RetType;
		return RetType(operand);
	}

	/* Then the type inference section */
	DEF_TYPE_INFERENCE_2ARGS(add, _1 + _2);
	DEF_TYPE_INFERENCE_2ARGS(sub, _1 - _2);
	DEF_TYPE_INFERENCE_2ARGS(mul, _1 * _2);
	DEF_TYPE_INFERENCE_2ARGS(div, _1 / _2);
	DEF_TYPE_INFERENCE_2ARGS(and, _1 & _2);
	DEF_TYPE_INFERENCE_2ARGS(or, _1 | _2);
	DEF_TYPE_INFERENCE_2ARGS(xor, _1 ^ _2);
	DEF_TYPE_INFERENCE_2ARGS(assign, _1);
	DEF_TYPE_INFERENCE_1ARG(neg, -_1);
	DEF_TYPE_INFERENCE_1ARG(not, ~_1);
	
	DEF_TYPE_INFERENCE_2ARGS(lt, _1 < _2);
	DEF_TYPE_INFERENCE_2ARGS(gt, _1 > _2);
	DEF_TYPE_INFERENCE_2ARGS(eq, _1 == _2);
	DEF_TYPE_INFERENCE_2ARGS(le, _1 <= _2);
	DEF_TYPE_INFERENCE_2ARGS(ge, _1 >= _2);
	DEF_TYPE_INFERENCE_2ARGS(ne, _1 != _2);
}
#endif
