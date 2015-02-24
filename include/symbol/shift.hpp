#ifndef __SHIFT_HPP__
#define __SHIFT_HPP__
namespace SpatialOps{
	/* Define the symbol for shift */
	template <typename what, int dx, int dy, int dz>
	struct symbol_shift:public SymbolicExpression{
		typedef what Operand;
		enum{
			Dx = dx,
			Dy = dy,
			Dz = dz,
		};
		inline symbol_shift(const what& op): operand(op){}
		const inline char* name() const
		{
			static char _buf[100];
			snprintf(_buf, 100, "Shift<%d, %d, %d>", Dx, Dy,Dz);
			return _buf;
		}
		inline void get_range(int& lx, int& ly, int&lz,
							  int& hx, int& hy, int&hz) const
		{
			typedef GetRange<Operand> RangeFinder;
			RangeFinder::get_range(operand, lx, ly, lz, hx, hy, hz);
			if(lx != INT_MIN) lx += Dx;
			if(ly != INT_MIN) ly += Dy;
			if(lz != INT_MIN) lz += Dz;
			if(hx != INT_MAX) hx += Dx;
			if(hy != INT_MAX) hy += Dy;
			if(hz != INT_MAX) hz += Dz;
		}
		const Operand operand;
	};
	template <typename operand, int x, int y, int z>
	struct GetNumOperands<symbol_shift<operand, x, y, z> >{
		enum{
			R = 1 
		};
	};
	template <typename operand, int x, int y, int z>
	struct GetRange<symbol_shift<operand, x, y, z> >{
		static const void get_range(const symbol_shift<operand, x,y,z>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)
		{
			e.get_range(lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename operand, int x, int y, int z>
	struct TopLevelFlag<symbol_shift<operand, x, y, z> >{
		static inline bool get(const symbol_shift<operand, x, y, z>& e)
		{
			return e.top_level;
		}
		static inline void clear(const symbol_shift<operand, x, y, z>& e)
		{
			((symbol_shift<operand, x, y, z>*)&e)->top_level = 0;
		}
	};
	/* define the shift operator */
	template<int dx ,int dy, int dz, typename operand>
	symbol_shift<operand, dx, dy, dz> shift(const operand& what)
	{
		TopLevelFlag<operand>::clear(what);
		return symbol_shift<operand, dx, dy, dz>(what);
	}
	template <typename operand, int x, int y, int z>
	struct ExprTypeInfer<symbol_shift<operand, x, y, z> >{
		typedef typename ExprTypeInfer<operand>::R R;
	};
}
#endif /* __SHIFT_HPP__ */
