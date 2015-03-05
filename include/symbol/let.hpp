#ifndef __SYMBOL_LET_HPP__
#define __SYMBOL_LET_HPP__
namespace SpatialOps{
	/**
	 * @brief reference to a let-binding variable
	 **/
	template <typename BindingId> struct symbol_ref;
	/**
	 * @brief the base class for a Let-Binding Id 
	 **/
	struct BindingId{
		template <typename T>
		static inline T& access(const T* v = NULL)
		{
			static T* t = NULL;
			if(NULL != v) t = v;
			return *t;
		}
		static inline symbol_ref<BindingId> ref()
		{
			return symbol_ref<BindingId>();
		}
	};
	/*{
		struct VarX:BindingId{};
		let<VarX>(x, VarX::ref() * VarX::ref())

		because we know the symbolic expression type of 
		each binding-variable. So that we can retrieve 
		the reference by it's type

		So the list needs to memorize 
			1. the BindingId
			2. the variable type


	}*/

}
#endif
