#ifndef __LINKER_HPP__

#define PADDING 4
#define SIZE_PADDING(type) (((int)type::Size&~(PADDING-1)) + (int)(((int)type::Size & (PADDING-1)) == 0) * PADDING)

namespace SpatialOps{
	/* Link the code with the Library */
	template <typename Expr, int DeviceId, typename Env, int offset> struct Executable;

	/**
	 * @brief Generate the executable code for the symbolic expression
	 **/
	template <int DeviceId, typename Expr>
	struct Linker{
		typedef InvokeDevicePP<DeviceId, Expr> PP;
		typedef Executable<typename PP::RetType, DeviceId, EmptyEnv, 0> Exec;
		typedef char CodeType[Exec::Size];
		static inline void link(const Expr& symbolic_expr, CodeType& mem)
		{
			Exec::init(PP::preprocess(symbolic_expr), mem);
		}
	};
	template <template <typename ,typename> class BinSym, typename Left, typename Right, int DeviceId, typename Env, int offset>
	struct Executable<BinSym<Left, Right>, DeviceId, Env, offset>{
		struct Self{};
		typedef Executable<Left, DeviceId, Env, offset> OP1Type;
		typedef Executable<Right, DeviceId, Env, offset + SIZE_PADDING(OP1Type)> OP2Type;
		typedef BinSym<Left, Right> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = SIZE_PADDING(OP1Type) + (int)OP2Type::Size};
		/*OP1Type _1;
		OP2Type _2;*/
		static inline void init(const Symbol& _symbol, void* mem)
		{
			OP1Type::init(_symbol.operand_l, mem);
			OP2Type::init(_symbol.operand_r, ((char*)mem) + SIZE_PADDING(OP1Type));
		}
		enum{
			_1 = 0,   /* the offset of 1st parameter */
			_2 = SIZE_PADDING(OP1Type), /* the offset of 2nd parameter */
			_self = 0 /* offset of it self */
		};
	};
	template <template <typename> class UniSym, typename Operand, int DeviceId, typename Env, int offset>
	struct Executable<UniSym<Operand>, DeviceId, Env, offset>{
		struct Self{};
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		typedef UniSym<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = (int)OP1Type::Size};
		//OP1Type _1;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			OP1Type::init(_symbol.operand, mem);
		}
		enum{
			_1 = 0,
			_self = 0
		};
	};
	template <typename Expr, int DeviceId, typename Env, int offset> 
	struct Executable{
		typedef Expr Symbol;
		typedef Symbol Self;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = sizeof(Symbol)};
		//Symbol _s;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			//new (mem) Symbol(_symbol);
			*(Symbol*)mem = _symbol;
		}
		enum{
			_1 = 0,
			_self = 0
		};
	};
	template <typename T, int DeviceId, typename Env, int offset>
	struct Executable<Field<T>, DeviceId, Env, offset>{
		typedef Field<T> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		struct Self{
			T* _m;
			int lx, ly, lz, hx, hy, hz;
		};
		//Self
		enum{Size = sizeof(Self)};
		static inline void init(const Symbol& _symbol, void* mem){
			Self* self = (Self*) mem;
			_symbol.get_range(self->lx, self->ly, self->lz, self->hx, self->hy, self->hz);
			self->_m = NULL;
		}
		enum{
			_self = 0
		};
	};
	template <typename Operand, int DeviceId, typename Env, int offset>
	struct Executable<REFSYM(window)<Operand>, DeviceId, Env, offset>{
		typedef REFSYM(window)<Operand> Symbol;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		typedef typename ExprTypeInfer<Symbol>::R RetType;
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		struct Self{
			int lx, ly, lz, hx, hy, hz;
			RetType defval;
		};
		enum{Size = SIZE_PADDING(OP1Type) + sizeof(Self)};
		//OP1Type _1;
		//Self;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			Self* self = (Self*)(((char*)mem) + SIZE_PADDING(OP1Type)); 
			OP1Type::init(_symbol.operand, mem);
			typedef GetRange<Operand, Env> RangeFinder;
			RangeFinder::get_range(_symbol.operand, self->lx, self->ly, self->lz, self->hx, self->hy, self->hz);
			new (&self->defval) RetType(_symbol.defval);
		}
		enum{
			_1 = 0,
			_self = SIZE_PADDING(OP1Type)
		};
	};
	template <typename Dir, int DeviceId, typename Env, int offset>
	struct Executable<symbol_coordinate<Dir>, DeviceId, Env, offset>{
		struct Self{};
		typedef symbol_coordinate<Dir> Symbol;
		enum{Size = 0};
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		static inline void init(const Symbol& _symbol, void* mem){}
		enum{
			_self = 0
		};
	};
	template <typename Operand, int Dx, int Dy, int Dz, int DeviceId, typename Env, int offset>
	struct Executable<symbol_shift<Operand, Dx, Dy, Dz>, DeviceId, Env, offset>{
		struct Self{};
		typedef symbol_shift<Operand, Dx, Dy, Dz> Symbol;
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = (int)OP1Type::Size};
		//OP1Type _1;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			OP1Type::init(_symbol.operand, mem);
		}
		enum{
			_1 = 0,
			_self = 0
		};
	};
	template <typename Operand, typename Annotation, int DeviceId, typename Env, int offset>
	struct Executable<symbol_annotation<Operand, Annotation>, DeviceId, Env, offset>{
		struct Self{};
		typedef symbol_annotation<Operand, Annotation> Symbol;
		typedef Executable<Operand, DeviceId, Env, offset> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = 0};
		static inline void init (const Symbol& _symbol, void* mem)
		{
			Executable<Operand, DeviceId, Env, offset>::init(_symbol, mem);
		}
		enum{
			_1 = 0,
			_self = 0
		};
	};
	template <typename var, typename Op1, typename Op2, int DeviceId, typename Env, int offset>
	struct Executable<symbol_binding<var, Op1, Op2>, DeviceId, Env, offset>
	{
		struct Self{};
		typedef symbol_binding<var, Op1, Op2> Symbol;
		typedef Executable<Op1, DeviceId, Env, offset> OP1Type;
		typedef AppendEnv<Env, var, OP1Type, offset> NewEnv;
		typedef Executable<Op2, DeviceId, NewEnv, offset + SIZE_PADDING(OP1Type)> OP2Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = SIZE_PADDING(OP1Type) + (int)OP2Type::Size};
		//OP1Type _1;
		//OP2Type _2;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			OP1Type::init(_symbol.operand_l, mem);
			OP2Type::init(_symbol.operand_r, ((char*)mem) + SIZE_PADDING(OP1Type));
		}
		enum{
			_1 = 0,
			_2 = SIZE_PADDING(OP1Type),
			_self = 0
		};
	};
	template <typename var, int DevId, typename env, int offset>
	struct Executable<symbol_ref<var>, DevId, env, offset>
	{
		struct Self{};
		typedef symbol_ref<var> Symbol;
		typedef GetEnv<var, env> EnvEntry;
		typedef typename EnvEntry::Expression Target;
		enum{Offset = (int)EnvEntry::Offset - offset};
		typedef typename InvokeDeviceLibrary<DevId, Symbol, Executable>::R CodeType;
		enum{ Size = 0};
		static inline void init(const Symbol& _symbol, void* mem){}
		enum{
			_self = 0
		};
	};
	template <typename T, int DeviceId, typename Env, int offset>
	struct Executable<LValueScalar<T>, DeviceId, Env, offset>{
		struct Self{};
		typedef LValueScalar<T> Symbol;
		typedef Executable<typename Symbol::Operand, DeviceId, Env, offset> OP1Type;
		typedef typename InvokeDeviceLibrary<DeviceId, Symbol, Executable>::R CodeType;
		enum{Size = (int)OP1Type::Size};
		//OP1Type _1;
		static inline void init(const Symbol& _symbol, void* mem)
		{
			OP1Type::init(_symbol.operand, mem);
		}
		enum{
			_1 = 0,
			_self = 0
		};
	};
};
#endif
