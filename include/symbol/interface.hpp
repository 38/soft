/**
 * @brief the top level file for syntax library, define
 *
 *        the template interface for the syntax description
 * @detail In the syntax description part, your C++ expression will be converted in to a
 *         *Symbolic Expression*, which is a tree strcture of symbolic types.
 *         This result can carries all the information that is needed to run, but
 *         it does not contains any code to execute.
 *         After that the linker will look for the library and figure out what code to use
 *         for each symbol and produce a actually runnable code.
 *         In the symbolic expression phrase, we do not check the type compatibility of the
 *         expression. But the error can finally be figured out during the linker generates
 *         the runtime code.
 * @file interface.hpp
 **/
#include <stdio.h>
#include <limits.h>
#ifndef __BASE_HPP__
#define __BASE_HPP__
namespace SpatialOps{
	/**
	 * @brief Get the Number of operands for an operator
	 * @param Operator the operator that we what to query
	 * @note  New operator that with non-zero operands should be specialized
	 **/
	template <typename Operator>
	struct GetNumOperands{
		enum{
			R = 0   /* 0 by Default */
		};
	};
	/**
	 * @brief Get the range of this operator
	 * @details The range of the expression is determinded by the
	 *         minmum range of each term.
	 *         So that if this not apply, we should insert some
	 *         window operator to extend or shrink the range
	 * @param Expr the expression to query
	 * @note  Operator that with finte range should rewrite this
	 **/
	template <typename Expr, typename Env>
	struct GetRange{
		static inline void get_range(const Expr& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)
		{
			lx = ly = lz = INT_MIN;
			hx = hy = hz = INT_MAX;
		}
	};
	/**
	 * @brief Check/Set the flag for top-level expressions
	 **/
	template <class Expr>
	struct TopLevelFlag{
		static inline bool get(const Expr& e){ return false;}
		static inline void clear(const Expr& e){}
	};
	/**
	 * @brief Infer the type of an symbolic expression
	 **/
	template <class Expr>
	struct ExprTypeInfer{
		typedef Expr R;
	};
	/**
	 * @brief Type for X-Direction
	 **/
	struct XDir{};
	/**
	 * @brief Type for Y-Direction
	 **/
	struct YDir{};
	/**
	 * @brief Type for Z-Direction
	 **/
	struct ZDir{};
	/**
	 * @brief Convert the direction type to a spatial vector
	 * @param D the direction
	 **/
	template<typename D>
	struct GetDirectVec;
	
	/******************
	* Implementations*
	******************/
	/**
	 * @brief run this symbolic expression
	 **/
	template <int DevId, typename SymExpr> struct SymExprExecutor;
	template <class Expr>
	static inline void run(const Expr& expr)
	{
		TopLevelFlag<Expr>::clear(expr);
		if(!SymExprExecutor<0, Expr>::execute_symexpr(expr))
		{
			fprintf(stderr, "failed to execute the expression!");
		}
	}
	template<>
	struct GetDirectVec<XDir>{
		enum{
			X = 1,
			Y = 0,
			Z = 0
		};
	};
	template<>
	struct GetDirectVec<YDir>{
		enum{
			X = 0,
			Y = 1,
			Z = 0
		};
	};
	template<>
	struct GetDirectVec<ZDir>{
		enum{
			X = 0,
			Y = 0,
			Z = 1
		};
	};
}

struct SymbolicExpression{
	bool top_level;
	SymbolicExpression():top_level(true){}
	SymbolicExpression(const SymbolicExpression&):top_level(false){}
};

template <bool runnable>
struct SymbolDestruction{
	template<typename symexp>
	static inline void invoke(const symexp& expr)
	{
		run(expr);
	}
};
template <>
struct SymbolDestruction<false>{
	template<typename symexp>
	static inline void invoke(const symexp& expr)
	{}
};
/* Define a symbol with 1 arg */
#define DEF_SYMBOL_1ARG(id, runnable) \
        template <typename op>\
        struct symbol_##id: public SymbolicExpression{\
	        typedef op Operand;\
	        inline const char* name() const {return #id;}\
	        symbol_##id(const op& oper) :operand(oper){}\
	        ~symbol_##id(){\
		        if(top_level)\
		            SymbolDestruction<runnable>::template invoke<symbol_##id>(*this);\
	        }\
	        const op operand; \
        };\
        template <typename op>\
        struct GetNumOperands<symbol_##id<op> >{\
	        enum{\
		        R = 1 \
	        };\
        };\
        template <typename op, typename Env>\
        struct GetRange<symbol_##id<op>, Env>{\
	        static void get_range(const symbol_##id<op>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)\
	        {\
		        typedef GetRange<op, Env> RangeFinder;\
		        RangeFinder::get_range(e.operand, lx, ly, lz, hx, hy, hz);\
	        }\
        };\
        template <typename op>\
        struct TopLevelFlag<symbol_##id<op> >{\
	        static inline bool get(const symbol_##id<op>& e)\
	        {\
		        return e.top_level;\
	        }\
	        static inline void clear(const symbol_##id<op>& e)\
	        {\
		        ((symbol_##id<op>*)&e)->top_level = 0;  /* really dirty, but really helps */\
	        }\
        }

/* define a symbol with 2 args */
#define DEF_SYMBOL_2ARGS(id, runnable) \
        template <typename left, typename right>\
        struct symbol_##id: public SymbolicExpression{\
	        typedef left Operand_l;\
	        typedef right Operand_r;\
	        inline const char* name() const {return #id;}\
	        symbol_##id(const left& l, const right& r) : operand_l(l), operand_r(r){}\
	        ~symbol_##id(){\
		        if(top_level)\
		            SymbolDestruction<runnable>::template invoke<symbol_##id>(*this);\
	        }\
	        const left operand_l;\
	        const right operand_r;\
        };\
        template <typename left, typename right>\
        struct GetNumOperands<symbol_##id<left, right> >{\
	        enum{\
		        R = 2 \
	        };\
        };\
        template <typename left, typename right, typename Env>\
        struct GetRange<symbol_##id<left, right>, Env>{\
	        static void get_range(const symbol_##id<left, right>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)\
	        {\
		        typedef GetRange<left, Env> RangeFinderLeft;\
		        typedef GetRange<right, Env> RangeFinderRight;\
		        int tlx, tly, tlz, thx, thy, thz;\
		        RangeFinderLeft::get_range(e.operand_l, lx, ly, lz, hx, hy, hz);\
		        RangeFinderRight::get_range(e.operand_r, tlx, tly, tlz, thx, thy, thz);\
		        if(lx < tlx) lx = tlx;\
		        if(ly < tly) ly = tly;\
		        if(lz < tlz) lz = tlz;\
		        if(hx > thx) hx = thx;\
		        if(hy > thy) hy = thy;\
		        if(hz > thz) hz = thz;\
	        }\
        };\
        template <typename left, typename right>\
        struct TopLevelFlag<symbol_##id<left, right> >{\
	        static inline bool get(const symbol_##id<left, right>& e)\
	        {\
		        return e.top_level;\
	        }\
	        static inline void clear(const symbol_##id<left, right>& e)\
	        {\
		        ((symbol_##id<left, right>*)&e)->top_level = 0;  /* really dirty, but really helps */\
	        }\
        }


#define REFSYM(id) symbol_##id

/* define a default binary operator */
#define DEF_DEFAULT_OPERAND_2ARGS(id, opname) \
        template <typename left, typename right>\
        static inline REFSYM(id)<left, right> operator opname(const left& l, const right& r)\
        {\
	        TopLevelFlag<left>::clear(l);\
	        TopLevelFlag<right>::clear(r);\
	        return REFSYM(id)<left, right>(l, r);\
        }

/* define a default binary operator */
#define DEF_DEFAULT_FUNCTION_2ARGS(id) \
        template <typename left, typename right>\
        static inline REFSYM(id)<left, right> id(const left& l, const right& r)\
        {\
	        TopLevelFlag<left>::clear(l);\
	        TopLevelFlag<right>::clear(r);\
	        return REFSYM(id)<left, right>(l, r);\
        }

/* define a default uniary operator */
#define DEF_DEFAULT_OPERAND_1ARG(id) \
        template <typename TOperand>\
        REFSYM(id)<TOperand> id(const TOperand& operand)\
        {\
	        TopLevelFlag<TOperand>::clear(operand);\
	        typedef symbol_##id<TOperand> RetType;\
	        return RetType(operand);\
        }
/* Define a type-inference rule for a binary operator */
#define DEF_TYPE_INFERENCE_2ARGS(id, infer_expr)\
    template<typename left, typename right>\
    struct ExprTypeInfer<REFSYM(id)<left, right> >{\
	    typedef typename ExprTypeInfer<left>::R LeftType;\
	    typedef typename ExprTypeInfer<right>::R RightType;\
	    static LeftType &_1;\
	    static RightType &_2;\
	    typedef typeof(infer_expr) R;\
    }
/* Define a type-inference rule for a uniary operator */
#define DEF_TYPE_INFERENCE_1ARG(id, infer_expr)\
    template<typename TOperand>\
    struct ExprTypeInfer<REFSYM(id)<TOperand> >{\
	    typedef typename ExprTypeInfer<TOperand>::R OperType;\
	    static OperType &_1;\
	    typedef typeof(infer_expr) R;\
    }
#endif
