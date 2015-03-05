#ifndef __LINKER_HPP__

namespace SpatialOps{
	/* Link the code with the Library */
	template <typename Expr, int DeviceId, typename Env, int offset> struct Executable;

	/**
	 * @brief Generate the executable code for the symbolic expression
	 **/
	template <int DeviceId, typename Expr>
	inline Executable<Expr, DeviceId, EmptyEnv, 0> link(const Expr& symbolic_expr){
		return Executable<Expr, DeviceId, EmptyEnv, 0>(InvokeDevicePP<DeviceId, Expr>::preprocess(symbolic_expr));
	}
	template <template <typename ,typename> class BinSym, typename Left, typename Right, int DeviceId, typename Env, int offset>
	struct Executable<BinSym<Left, Right>, DeviceId, Env, offset>{
		typedef Executable<Left, DeviceId, Env, offset> OP1Type;
		typedef Executable<Right, DeviceId, Env, offset + sizeof(OP1Type)> OP2Type;
		typedef BinSym<Left, Right> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		OP2Type _2;
		inline Executable(const Symbol& _symbol): _1(_symbol.operand_l), _2(_symbol.operand_r){}
	};
	template <template <typename> class UniSym, typename Operand, int DeviceId, typename Env, int offset>
	struct Executable<UniSym<Operand>, DeviceId, Env, offset>{
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		typedef UniSym<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		inline Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
	template <typename Expr, int DeviceId, typename Env, int offset> 
	struct Executable{
		typedef Expr Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		Symbol _s;
		Executable(const Symbol& _symbol): _s(_symbol){}
	};
	template <typename T, int DeviceId, typename Env, int offset>
	struct Executable<Field<T>, DeviceId, Env, offset>{
		typedef Field<T> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		T* _m;
		int lx, ly, lz, hx, hy, hz;
		Executable(const Symbol& _symbol):  _m(NULL){
			_symbol.get_range(lx, ly, lz, hx, hy, hz);
		}
	};
	template <typename Operand, int DeviceId, typename Env, int offset>
	struct Executable<REFSYM(window)<Operand>, DeviceId, Env, offset>{
		typedef REFSYM(window)<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		typedef typename ExprTypeInfer<Symbol>::R RetType;
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		OP1Type _1;
		int lx, ly, lz, hx, hy, hz;
		RetType defval;
		Executable(const Symbol& _symbol): _1(_symbol.operand)
		{
			typedef GetRange<Operand, Env> RangeFinder;
			RangeFinder::get_range(_symbol.operand, lx, ly, lz, hx, hy, hz);
			defval = _symbol.defval;
		}
	};
	template <typename Dir, int DeviceId, typename Env, int offset>
	struct Executable<symbol_coordinate<Dir>, DeviceId, Env, offset>{
		typedef symbol_coordinate<Dir> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		Executable(const Symbol& _symbol){}
	};
	template <typename Operand, int Dx, int Dy, int Dz, int DeviceId, typename Env, int offset>
	struct Executable<symbol_shift<Operand, Dx, Dy, Dz>, DeviceId, Env, offset>{
		typedef symbol_shift<Operand, Dx, Dy, Dz> Symbol;
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
	template <typename Operand, typename Annotation, int DeviceId, typename Env, int offset>
	struct Executable<symbol_annotation<Operand, Annotation>, DeviceId, Env, offset>: public Executable<Operand, DeviceId, Env, offset>{
		/* simply do nothing */
		typedef symbol_annotation<Operand, Annotation> Symbol;
		Executable(const Symbol& _symbol): Executable<Operand, DeviceId, Env, offset>(_symbol){}
	};
	template <typename var, typename Op1, typename Op2, int DeviceId, typename Env, int offset>
	struct Executable<symbol_binding<var, Op1, Op2>, DeviceId, Env, offset>
	{
		typedef symbol_binding<var, Op1, Op2> Symbol;
		typedef Executable<Op1, DeviceId, Env, offset> OP1Type;
		typedef AppendEnv<Env, var, OP1Type, offset> NewEnv;
		typedef Executable<Op2, DeviceId, NewEnv, offset + sizeof(OP1Type)> OP2Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		OP2Type _2;
		Executable(const Symbol& _symbol): _1(_symbol.operand_l), _2(_symbol.operand_r){}
	};
	template <typename var, int DevId, typename env, int offset>
	struct Executable<symbol_ref<var>, DevId, env, offset>
	{
		typedef symbol_ref<var> Symbol;
		typedef GetEnv<var, env> EnvEntry;
		typedef typename EnvEntry::Expression Target;
		enum{Offset = (int)EnvEntry::Offset - offset};
		typedef typename InvokeDeviceLibrary<DevId, Symbol, Executable>::R CodeType;
		Executable(const Symbol& _symbol){}
	};
	template <typename T, int DeviceId, typename Env, int offset>
	struct Executable<LValueScalar<T>, DeviceId, Env, offset>{
		typedef LValueScalar<T> Symbol;
		typedef Executable<typename Symbol::Operand, DeviceId, Env, offset> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		OP1Type _1;
		Executable(const Symbol& _symbol): _1(_symbol.operand){}
	};
};
#endif
