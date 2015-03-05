#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__
namespace SpatialOps{
	template <int DevId, typename Executable, int NumExpr>
	struct DataValidator{
		static inline bool validate(const Executable& e ,const typename Executable::Symbol& s)
		{
			return true;
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 2>{
		static inline bool validate(const Executable& e, const typename Executable::Symbol& s)
		{
			return DataValidator<DevId, typename Executable::OP1Type, GetNumOperands<typename Executable::Symbol::Operand_l>::R>::validate(e._1, s.operand_l) &&
			       DataValidator<DevId, typename Executable::OP2Type, GetNumOperands<typename Executable::Symbol::Operand_r>::R>::validate(e._2, s.operand_r);
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 1>{
		static inline bool validate(const Executable& e, const typename Executable::Symbol& s)
		{
			return DataValidator<DevId, typename Executable::OP1Type, GetNumOperands<typename Executable::Symbol::Operand>::R>::validate(e._1, s.operand);
		}
	};
	template <int DeviceId, typename T, typename Env, int offset>
	struct DataValidator<DeviceId, Executable<Field<T>, DeviceId, Env, offset>, 0>{
		static inline bool validate(const Executable<Field<T>, DeviceId, Env, offset>& e, const Field<T>& s)
		{
			return (NULL != (((Executable<Field<T>, DeviceId, Env, offset>*)&e)->_m = s.template get_memory<DeviceId>()));
		}
	};
	template <int DevId, typename Executable>
	static inline bool data_validate(const Executable& e, const typename Executable::Symbol& s)
	{
		return DataValidator<DevId, Executable, GetNumOperands<typename Executable::Symbol>::R>::validate(e, s);
	}
	template <int DevId, typename SymExpr>
	struct SymExprExecutor{
		static inline bool execute_symexpr(const SymExpr& expr){
			typedef typeof(link<DevId>(expr)) Exec;
			Exec e = link<DevId>(expr);

			printf("%zu\n", sizeof(e));
			
			if(!data_validate<DevId>(e, expr) || !GetDeviceRuntimeEnv<DevId>::R::execute(e, expr)) 
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
}
#endif
