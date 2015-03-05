#ifndef __LINKER_HPP__

namespace SpatialOps{
	/* Link the code with the Library */
	template <typename Expr, int DeviceId> struct Executable;

	/**
	 * @brief Generate the executable code for the symbolic expression
	 **/
	template <int DeviceId, typename Expr>
	inline Executable<Expr, DeviceId> link(const Expr& symbolic_expr){
		return Executable<Expr, DeviceId>(InvokeDevicePP<DeviceId, Expr>::preprocess(symbolic_expr));
	}
	template <template <typename ,typename> class BinSym, typename Left, typename Right, int DeviceId>
	struct Executable<BinSym<Left, Right>, DeviceId>{
		typedef Executable<Left, DeviceId> OP1Type;
		typedef Executable<Right, DeviceId> OP2Type;
		typedef BinSym<Left, Right> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		OP2Type _2;
		inline Executable(const Symbol& _symbol): _1(_symbol.operand_l), _2(_symbol.operand_r){}
	};
	template <template <typename> class UniSym, typename Operand, int DeviceId>
	struct Executable<UniSym<Operand>, DeviceId>{
		typedef Executable<Operand, DeviceId> OP1Type;
		typedef UniSym<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		inline Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
	template <typename Expr, int DeviceId> 
	struct Executable{
		typedef Expr Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		Symbol _s;
		Executable(const Symbol& _symbol): _s(_symbol){}
	};
	template <typename T, int DeviceId>
	struct Executable<Field<T>, DeviceId>{
		typedef Field<T> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		T* _m;
		int lx, ly, lz, hx, hy, hz;
		Executable(const Symbol& _symbol):  _m(NULL){
			_symbol.get_range(lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename Operand, int DeviceId>
	struct Executable<REFSYM(window)<Operand>, DeviceId>{
		typedef REFSYM(window)<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		typedef typename ExprTypeInfer<Symbol>::R RetType;
		typedef Executable<Operand, DeviceId> OP1Type;
		int lx, ly, lz, hx, hy, hz;
		RetType defval;
		OP1Type _1;
		Executable(const Symbol& _symbol): _1(_symbol.operand)
		{
			typedef GetRange<Operand> RangeFinder;
			RangeFinder::get_range(_symbol.operand, lx, ly, lz, hx, hy, hz);
			defval = _symbol.defval;
		}
	};
	template <typename Dir, int DeviceId>
	struct Executable<symbol_coordinate<Dir>, DeviceId>{
		typedef symbol_coordinate<Dir> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		Executable(const Symbol& _symbol){}
	};
	template <typename Operand, int Dx, int Dy, int Dz, int DeviceId>
	struct Executable<symbol_shift<Operand, Dx, Dy, Dz>, DeviceId>{
		typedef symbol_shift<Operand, Dx, Dy, Dz> Symbol;
		typedef Executable<Operand, DeviceId> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
	template <typename Operand, typename Annotation, int DeviceId>
	struct Executable<symbol_annotation<Operand, Annotation>, DeviceId>: public Executable<Operand, DeviceId>{
		/* simply do nothing */
		typedef symbol_annotation<Operand, Annotation> Symbol;
		Executable(const Symbol& _symbol): Executable<Operand, DeviceId>(_symbol){}
	};

	template <typename T, int DeviceId>
	struct Executable<LValueScalar<T>, DeviceId>{
		typedef LValueScalar<T> Symbol;
		typedef Executable<typename Symbol::Operand, DeviceId> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
};
#endif
