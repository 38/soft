#ifndef __FUNCTION_MATH_HPP__
#define TYPE(T) (*(T*)NULL)
namespace SpatialOps{
	template<typename OpT>
	struct squre_type_inf{
		struct X{typedef typename ExprTypeInfer<OpT>::R T;};
		typedef typeof(let<X>(TYPE(OpT), ref<X>() * ref<X>())) R;
	};
	template<typename OpT>
	typename squre_type_inf<OpT>::R square(const OpT& op)
	{
		typedef typename squre_type_inf<OpT>::X X;
		return let<X>(op, ref<X>() * ref<X>());
	}
	
	template<typename OpT,typename Dir>
	struct D_type_inf{
		 typedef GetDirectVec<Dir> D;
		struct X{typedef typename ExprTypeInfer<OpT>::R T;};
		typedef typeof(let<X>(TYPE(OpT), ref<X>() - shift<-(int)D::X, -(int)D::Y, -(int)D::Z>(ref<X>()))) R;
	};
	template<typename Dir, typename OpT>
	typename D_type_inf<OpT, Dir>::R Div(const OpT& op)
	{
		 typedef GetDirectVec<Dir> D;
		 typedef typename D_type_inf<OpT, Dir>::X X;
		 return let<X>(op, ref<X>() - shift<-(int)D::X, -(int)D::Y, -(int)D::Z>(ref<X>()));
	}
	
	template<typename OpT,typename Dir>
	struct interp_type_inf{
		 typedef GetDirectVec<Dir> D;
		struct X{typedef typename ExprTypeInfer<OpT>::R T;};
		typedef typeof(let<X>(TYPE(OpT), (ref<X>() + shift<-(int)D::X, -(int)D::Y, -(int)D::Z>(ref<X>()))/2)) R;
	};
	template<typename Dir, typename OpT>
	typename interp_type_inf<OpT, Dir>::R Interp(const OpT& op)
	{
		 typedef GetDirectVec<Dir> D;
		 typedef typename interp_type_inf<OpT, Dir>::X X;
		 return let<X>(op, (ref<X>() + shift<-(int)D::X, -(int)D::Y, -(int)D::Z>(ref<X>())) / 2);
	}
	
	
	template<typename OpT,typename Dir>
	struct DR_type_inf{
		 typedef GetDirectVec<Dir> D;
		struct X{typedef typename ExprTypeInfer<OpT>::R T;};
		typedef typeof(let<X>(TYPE(OpT), (shift<(int)D::X, (int)D::Y, (int)D::Z>(ref<X>())) - ref<X>())) R;
	};
	template<typename Dir, typename OpT>
	typename DR_type_inf<OpT, Dir>::R DivR(const OpT& op)
	{
		 typedef GetDirectVec<Dir> D;
		 typedef typename DR_type_inf<OpT, Dir>::X X;
		 return let<X>(op, shift<(int)D::X, (int)D::Y, (int)D::Z>(ref<X>()) - ref<X>());
	}
}
#endif
