#ifndef __MATHFUNC_HPP__
#define __MATHFUNC_HPP__
#include <algorithm>
#include <cmath>
namespace SpatialOps{
	/* Definition for the math function operators */
	DEF_SYMBOL_1ARG(sin, false);
	DEF_SYMBOL_1ARG(cos, false);
	DEF_SYMBOL_1ARG(tan, false);
	DEF_SYMBOL_1ARG(asin, false);
	DEF_SYMBOL_1ARG(acos, false);
	DEF_SYMBOL_1ARG(atan, false);
	DEF_SYMBOL_1ARG(abs, false);
	DEF_SYMBOL_1ARG(exp, false);
	DEF_SYMBOL_1ARG(log, false);
	DEF_SYMBOL_1ARG(sqrt, false);
	DEF_SYMBOL_2ARGS(max, false);
	DEF_SYMBOL_2ARGS(min, false);
	
	
	/* Define the operators */
	DEF_DEFAULT_OPERAND_1ARG(sin);
	DEF_DEFAULT_OPERAND_1ARG(cos);
	DEF_DEFAULT_OPERAND_1ARG(tan);
	DEF_DEFAULT_OPERAND_1ARG(asin);
	DEF_DEFAULT_OPERAND_1ARG(acos);
	DEF_DEFAULT_OPERAND_1ARG(atan);
	DEF_DEFAULT_OPERAND_1ARG(abs);
	DEF_DEFAULT_OPERAND_1ARG(exp);
	DEF_DEFAULT_OPERAND_1ARG(log);
	DEF_DEFAULT_OPERAND_1ARG(sqrt);
	DEF_DEFAULT_FUNCTION_2ARGS(min);
	DEF_DEFAULT_FUNCTION_2ARGS(max);
	
	
	/* Type inference section */
	DEF_TYPE_INFERENCE_1ARG(sin, std::sin(_1));
	DEF_TYPE_INFERENCE_1ARG(cos, std::cos(_1));
	DEF_TYPE_INFERENCE_1ARG(tan, std::tan(_1));
	DEF_TYPE_INFERENCE_1ARG(asin, std::asin(_1));
	DEF_TYPE_INFERENCE_1ARG(acos, std::acos(_1));
	DEF_TYPE_INFERENCE_1ARG(atan, std::atan(_1));
	DEF_TYPE_INFERENCE_1ARG(abs, std::abs(_1));
	DEF_TYPE_INFERENCE_1ARG(exp, std::exp(_1));
	DEF_TYPE_INFERENCE_1ARG(log, std::log(_1));
	DEF_TYPE_INFERENCE_1ARG(sqrt, std::sqrt(_1));
	DEF_TYPE_INFERENCE_2ARGS(min, std::min(_1, _2));
	DEF_TYPE_INFERENCE_2ARGS(max, std::max(_1, _2));

}
#endif

