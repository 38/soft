#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__
namespace SpatialOps{
	template <int DevId, typename Executable, int NumExpr>
	struct DataValidator{
		static inline bool validate(const void* e ,const typename Executable::Symbol& s)
		{
			return true;
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 3>{
		static inline bool validate(void* e, const typename Executable::Symbol& s)
		{
			return DataValidator<DevId,
			                     typename Executable::OP1Type,
			                     GetNumOperands<typename Executable::Symbol::Operand_1>::R
			                    >::validate(((char*)e) + (int)Executable::_1, s.operand_1) &&
			       DataValidator<DevId,
			                     typename Executable::OP2Type,
			                     GetNumOperands<typename Executable::Symbol::Operand_r>::R
			                    >::validate(((char*)e) + (int)Executable::_2, s.operand_2) &&
			       DataValidator<DevId,
			                        typename Executable::OP3Type,
			                     GetNumOperands<typename Executable::Symbol::Operand_2>::R
			                    >::validate(((char*)e) + (int)Executable::_3, s.operand_3);
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 2>{
		static inline bool validate(void* e, const typename Executable::Symbol& s)
		{
			return DataValidator<DevId,
			                     typename Executable::OP1Type,
			                     GetNumOperands<typename Executable::Symbol::Operand_l>::R
			                    >::validate(((char*)e) + (int)Executable::_1, s.operand_l) &&
			       DataValidator<DevId,
			                     typename Executable::OP2Type,
			                     GetNumOperands<typename Executable::Symbol::Operand_r>::R
			                    >::validate(((char*)e) + (int)Executable::_2, s.operand_r);
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 1>{
		static inline bool validate(void* e, const typename Executable::Symbol& s)
		{
			return DataValidator<DevId,
			                     typename Executable::OP1Type,
			                     GetNumOperands<typename Executable::Symbol::Operand>::R
			                    >::validate(((char*)e) + (int)Executable::_1, s.operand);
		}
	};
	template <int DeviceId, typename T, typename Env, int offset>
	struct DataValidator<DeviceId, Executable<Field<T>, DeviceId, Env, offset>, 0>{
		static inline bool validate(void* e, const Field<T>& s)
		{
			bool result = (NULL != (((typename Executable<Field<T>, DeviceId, Env, offset>::Self*)e)->_m = s.template get_memory<DeviceId>()));
			if(!result)
			{
				fprintf(stderr, "Can not allocate memory on device %d\n", DeviceId);
			}
			return result;
		}
	};
	template <int DevId, typename Executable>
	static inline bool data_validate(void* e, const typename Executable::Symbol& s)
	{
		return DataValidator<DevId, Executable, GetNumOperands<typename Executable::Symbol>::R>::validate(e, s);
	}
	template <int DevId, typename SymExpr>
	struct SymExprExecutor{
		static inline bool execute_symexpr(const SymExpr& expr){
			typedef Linker<DevId, SymExpr> LinkerType;
			typedef typename LinkerType::Exec Exec;
			typename LinkerType::CodeType e;
			LinkerType::link(expr, e);
			if(!data_validate<DevId, Exec>((void*)e, expr) || !GetDeviceRuntimeEnv<DevId>::R::template execute<Exec>(e, expr))
			    return SymExprExecutor<DevId + 1, SymExpr>::execute_symexpr(expr);
			return true;
		}
	};
	template <typename SymExpr>
	struct SymExprExecutor<NUM_DEVICE_TYPES, SymExpr>{
		static inline bool execute_symexpr(const SymExpr& expr){
			return false;
		}
	};
	

	/* execute the expression on the preferred device */
	template<int DevId, typename SymExpr>
	struct PreferredExecutor{
		static inline bool execute_symexpr(const SymExpr& expr){
			if(DevicePreference::get() == DevId)
			{
				typedef Linker<DevId, SymExpr> LinkerType;
				typedef typename LinkerType::Exec Exec;
				typename LinkerType::CodeType e;
				LinkerType::link(expr, e);
				return (!data_validate<DevId, Exec>((void*)e, expr) || !GetDeviceRuntimeEnv<DevId>::R::template execute<Exec>(e, expr));
			}
			return PreferredExecutor<DevId + 1, SymExpr>::execute_symexpr(expr);
		}
	};
	template <typename SymExpr>
	struct PreferredExecutor<NUM_DEVICE_TYPES, SymExpr>{
		static inline bool execute_symexpr(const SymExpr& expr)
		{
			return false;
		}
	};
	
}
#endif
