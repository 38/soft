#ifndef __RUNTIME_HPP__
#define __RUNTIME_HPP__
namespace SpatialOps{
	template <int DevId, typename Executable, int NumExpr>
	struct DataValidator{
		static inline bool validate(const Executable& e)
		{
			return true;
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 2>{
		static inline bool validate(const Executable& e)
		{
			return DataValidator<DevId, typename Executable::OP1Type, GetNumOperands<typename Executable::Symbol::Operand_l>::R>::validate(e._1) &&
			       DataValidator<DevId, typename Executable::OP2Type, GetNumOperands<typename Executable::Symbol::Operand_r>::R>::validate(e._2);
		}
	};
	template <int DevId, typename Executable>
	struct DataValidator<DevId, Executable, 1>{
		static inline bool validate(const Executable& e)
		{
			return DataValidator<DevId, typename Executable::OP1Type, GetNumOperands<typename Executable::Symbol::Operand>::R>::validate(e._1);
		}
	};
	template <int DeviceId, typename T>
	struct DataValidator<DeviceId, Executable<Field<T>, DeviceId>, 0>{
		static inline bool validate(const Executable<Field<T>, DeviceId>& e)
		{
			return (NULL != (((Executable<Field<T>, DeviceId>*)&e)->_m = e._s.template get_memory<DeviceId>()));
		}
	};
	template <int DevId, typename Executable>
	static inline bool data_validate(const Executable& e)
	{
		return DataValidator<DevId, Executable, GetNumOperands<typename Executable::Symbol>::R>::validate(e);
	}
	template <int DevId, typename SymExpr>
	struct SymExprExecutor{
		static inline bool execute_symexpr(const SymExpr& expr){
			typedef typeof(link<DevId>(expr)) Exec;
			Exec e = link<DevId>(expr);
			if(!data_validate<DevId>(e) || !GetDeviceRuntimeEnv<DevId>::R::execute(e)) 
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
