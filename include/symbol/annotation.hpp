#ifndef __SYMBOL_ANNOTATION_HPP__
#define __SYMBOL_ANNOTATION_HPP__
namespace SpatialOps{
	/* Define the symbol to make linker annotations */
	template <typename what, typename annotation>
	struct symbol_annotation:public SymbolicExpression{
		typedef what Operand;
		inline symbol_annotation(const what& op): operand(op){}
		const inline char* name() const
		{
			return "Annotation";
		}
		const Operand operand;
	};
	template <typename operand, typename annotation>
	struct GetNumOperands<symbol_annotation<operand, annotation> >{
		enum{
			R = 1
		};
	};
	template <typename operand, typename annotation>
	struct GetRange<symbol_annotation<operand, annotation> >{
		static void get_range(const symbol_annotation<operand, annotation>& e, int& lx, int& ly, int&lz, int& hx, int& hy, int& hz)
		{
			typedef typename GetRange<operand>::R RangeFinder;
			RangeFinder::get_range(e.operand,lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename operand, typename annotation>
	struct TopLevelFlag<symbol_annotation<operand, annotation> >{
		static inline bool get(const symbol_annotation<operand, annotation>& e)
		{
			return e.top_level;
		}
		static inline void clear(const symbol_annotation<operand, annotation>& e)
		{
			((symbol_annotation<operand,annotation>*)&e)->top_level = 0;
		}
	};
	template <typename operand, typename annotation>
	struct ExprTypeInfer<symbol_annotation<operand, annotation> >{
		typedef typename ExprTypeInfer<operand>::R R;
	};
	/* internal use only, We do not provide a interface to user to access this symbol */
}
#endif
