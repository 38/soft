#ifndef __UTIL_ENVIRON_HPP__
#define __UTIL_ENVIRON_HPP__
namespace SpatialOps{
	/** Environ */
	struct EmptyEnv{
		typedef int Id;
	};
	template <typename IdType, typename Expr, typename Rem>
	struct AppendEnv{
		typedef Expr Symbol;
		typedef IdType Id;
		typedef Rem Next;
	};
	template <typename Id, typename Env> struct GetEnv;
	template <typename TargetId, typename CurrentId, typename Env>
	struct _GetEnvImpl{
		typedef typename _GetEnvImpl<TargetId, typename Env::Next::Id, typename Env::Next>::R R;
	};
	template <typename TargetId>
	struct _GetEnvImpl<TargetId, int, EmptyEnv>{
		/* Nothing to define, so a compilation error raise here */
	};
	template <typename TargetId, typename Env>
	struct _GetEnvImpl<TargetId, TargetId, Env>{
		typedef typename Env::Symbol R;
	};
	template <typename Id, typename Env>
	struct GetEnv{
		typedef typename _GetEnvImpl<Id, typename Env::Id, Env>::R R;
	};
}
#endif
