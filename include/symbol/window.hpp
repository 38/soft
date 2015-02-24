#ifndef __WINDOW_HPP__
#define __WINDOW_HPP__
namespace SpatialOps{
	/* Symbol for Window */
	template <typename what>
	struct symbol_window:public SymbolicExpression{
		typedef typename ExprTypeInfer<what>::R defval_type; 
		typedef what Operand;
		const Operand operand;
		int low[3], high[3];
		defval_type defval;
		inline symbol_window(const what& op, 
							 int lx, int ly, int lz,
							 int hx, int hy, int hz,
							 defval_type dval): operand(op)
		{
			low[0] = lx, low[1] = ly, low[2] = lz;
			high[0] = hx, high[1] = hy, high[2] = hz;
			defval = dval;
		}
		const inline char* name() const
		{
			static char _buf[100];
			snprintf(_buf, 100, "Window<from [%d, %d, %d] to [%d, %d, %d]>", 
					 low[0], low[1], low[2],
					 high[0], high[1], high[2]
					);
			return _buf;
		}
		inline void get_range(int& lx, int& ly, int&lz,
							  int& hx, int& hy, int&hz) const
		{
			lx = low[0], ly = low[1], lz = low[2];
			hx = high[0], hy = high[1], hz = high[2];
		}
	};
	template <typename operand>
	struct GetNumOperands<symbol_window<operand> >{
		enum{
			R = 1 
		};
	};
	template <typename operand>
	struct GetRange<symbol_window<operand> >{
		static void get_range(const symbol_window<operand>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)
		{
			e.get_range(lx, ly, lz, hx, hy, hz);
		}
	};
	
	template <typename operand>
	struct TopLevelFlag<symbol_window<operand> >{
		static inline bool get(const symbol_window<operand>& e)
		{
			return e.top_level;
		}
		static inline void clear(const symbol_window<operand>& e)
		{
			((symbol_window<operand>*)&e)->top_level = 0;
		}
	};
	/* Operator for the Window */
	template<typename operand>
	symbol_window<operand> window(const operand& what, typename symbol_window<operand>::defval_type val,
								  int lx, int ly, int lz,
								  int hx, int hy, int hz)
	{
		TopLevelFlag<operand>::clear(what);
		return symbol_window<operand>(what, lx, ly, lz, hx, hy, hz, val);
	}
	
	template <typename operand>
	struct ExprTypeInfer<symbol_window<operand> >{
		typedef typename ExprTypeInfer<operand>::R R; 
	};
}
#endif /*__WINDOW_HPP__*/
