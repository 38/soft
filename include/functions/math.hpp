#ifndef __FUNCTION_MATH_HPP__
#define TYPE(T) (*(T*)NULL)
namespace SpatialOps{
	template<typename OpT>
	struct squre_type_inf{
		typedef typeof(TYPE(OpT) * TYPE(OpT)) R;
	};
	template<typename OpT>
	typename squre_type_inf<OpT>::R square(const OpT& op)
	{
		return op * op;
	}
	template<typename OpT,typename Dir>
	struct D_type_inf{
		typedef typeof((TYPE(OpT) - shift<-(int)GetDirectVec<Dir>::X, -(int)GetDirectVec<Dir>::Y, -(int)GetDirectVec<Dir>::Z>(TYPE(OpT)))) R;
	};
	template<typename Dir, typename OpT>
	typename D_type_inf<OpT, Dir>::R D(const OpT& op)
	{
		 return (op - shift<-(int)GetDirectVec<Dir>::X, -(int)GetDirectVec<Dir>::Y, -(int)GetDirectVec<Dir>::Z>(op));
	}
}
#endif
