#ifndef __LINKER_HPP__
namespace SpatialOps{
	/* Link the code with the Library */
	template <typename Expr, int DeviceId> struct Executable;

	/**
	 * @brief Generate the executable code for the symbolic expression
	 **/
	template <int DeviceId, typename Expr>
	inline Executable<Expr, DeviceId> link(const Expr& symbolic_expr){
		return Executable<Expr, DeviceId>(symbolic_expr);
	}
	template <template <typename ,typename> class BinSym, typename Left, typename Right, int DeviceId>
	struct Executable<BinSym<Left, Right>, DeviceId>{
		typedef Executable<Left, DeviceId> OP1Type;
		typedef Executable<Right, DeviceId> OP2Type;
		typedef BinSym<Left, Right> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		const Symbol& _s;
		OP1Type _1;
		OP2Type _2;
		inline Executable(const Symbol& _symbol): _s(_symbol), _1(_symbol.operand_l), _2(_symbol.operand_r){}
	};
	template <template <typename> class UniSym, typename Operand, int DeviceId>
	struct Executable<UniSym<Operand>, DeviceId>{
		typedef Executable<Operand, DeviceId> OP1Type;
		typedef UniSym<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		const Symbol& _s;
		OP1Type _1;
		inline Executable(const Symbol& _symbol): _s(_symbol), _1(_symbol.operand){}
	};
	template <typename Expr, int DeviceId> 
	struct Executable{
		typedef Expr Symbol;
		const Symbol& _s;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		Executable(const Symbol& _symbol): _s(_symbol){}
	};
	template <typename T, int DeviceId>
	struct Executable<Field<T>, DeviceId>{
		typedef Field<T> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		const Symbol& _s;
		T* _m;
		Executable(const Symbol& _symbol): _s(_symbol), _m(NULL){}
	};
	
	template <typename Dir, int DeviceId>
	struct Executable<symbol_coordinate<Dir>, DeviceId>{
		typedef symbol_coordinate<Dir> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		const Symbol& _s;
		Executable(const Symbol& _symbol): _s(_symbol){}
	};
	template <typename Operand, int Dx, int Dy, int Dz, int DeviceId>
	struct Executable<symbol_shift<Operand, Dx, Dy, Dz>, DeviceId>{
		typedef symbol_shift<Operand, Dx, Dy, Dz> Symbol;
		typedef Executable<Operand, DeviceId> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		const Symbol& _s;
		OP1Type _1;
		Executable(const Symbol& _symbol): _s(_symbol), _1(_s.operand){}
	};
};
#endif
