#include <stdio.h>
#include <limits.h>
#ifndef __BASE_HPP__
#define __BASE_HPP__
namespace SpatialOps{
	
	/* Get the number of operand of this operator */
	template <typename Operator>
	struct GetNumOperands{
		enum{R = 0};
	};
	
	/* Get the range of this expression */
	template <typename Expr, typename Env>
	struct GetRange{
		static inline void get_range(const Expr& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)
		{
			lx = ly = lz = INT_MIN;
			hx = hy = hz = INT_MAX;
		}
	};
	
	/* Get/Set the top level flag */
	template <class Expr>
	struct TopLevelFlag{
		static inline bool get(const Expr& e){ return false;}
		static inline void clear(const Expr& e){}
	};
	
	/* Infer the return type of the expression  */
	template <class Expr>
	struct ExprTypeInfer{
		typedef Expr R;
	};
	
	/* Directions */
	struct XDir{};
	struct YDir{};
	struct ZDir{};
	
	/* Get The Direction */
	template<typename D> struct GetDirectVec;
	
	/* Previous defs of runtime templates */
	template <int DevId, typename SymExpr> struct PreferredExecutor;
	template <int DevId, typename SymExpr> struct SymExprExecutor;
	
	/* Run the symbolic expression */
	template <class Expr>
	static inline void run(const Expr& expr)
	{
		TopLevelFlag<Expr>::clear(expr);
		
		if(DevicePreference::get() != -1 && !PreferredExecutor<0, Expr>::execute_symexpr(expr))
		    fprintf(stderr, "Warning: Expression execution failed on the preferred device\n");
		
		if(!SymExprExecutor<0, Expr>::execute_symexpr(expr))
		    fprintf(stderr, "failed to execute the expression!\n");
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
}

/* Define a symbol with 1 argument */
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

/* define a symbol with 2 arguments */
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

/* define a default unary operator */
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
/* Define a type-inference rule for a unary operator */
#define DEF_TYPE_INFERENCE_1ARG(id, infer_expr)\
    template<typename TOperand>\
    struct ExprTypeInfer<REFSYM(id)<TOperand> >{\
	    typedef typename ExprTypeInfer<TOperand>::R OperType;\
	    static OperType &_1;\
	    typedef typeof(infer_expr) R;\
    }
#endif
