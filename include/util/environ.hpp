#ifndef __UTIL_ENVIRON_HPP__
#define __UTIL_ENVIRON_HPP__
#include <stdlib.h>
namespace SpatialOps{
	/** Environ */
	struct EmptyEnv{
		typedef int Id;
	};
	template <typename Rem, typename Sym, typename Expr, int offset>
	struct AppendEnv{
		enum{
			Offset = offset
		};
		typedef Expr Expression;
		typedef Sym  Var;
		typedef Rem Next;
		static inline const Expression& get(const Expression* e)
		{
			static const Expression* inst = NULL;
			if(NULL != e) inst = e;
			return *inst;
		}
	};
	template <typename Id, typename Env> struct GetEnv;
	template <typename TargetId, typename CurrentId, typename Env>
	struct _GetEnvImpl{
		typedef _GetEnvImpl<TargetId, typename Env::Next::Var, typename Env::Next> _Next;
		typedef typename _Next::Expression Expression;
		enum{
			Offset = _Next::Offset
		};
		typedef typename _Next::Environ Environ;
		static inline const Expression& get(const Expression* e)
		{
			return _Next::get(e);
		}
	};
	template <typename TargetId>
	struct _GetEnvImpl<TargetId, int, EmptyEnv>{
		/* Nothing to define, so a compilation error raise here */
	};
	template <typename TargetId, typename Env>
	struct _GetEnvImpl<TargetId, TargetId, Env>{
		typedef typename Env::Expression Expression;
		enum{
			Offset = Env::Offset
		};
		typedef typename Env::Next Environ;
		static inline const Expression& get(const Expression* e)
		{
			return Env::get(e);
		}
	};
	template <typename Id, typename Env>
	struct GetEnv{
		typedef _GetEnvImpl<Id, typename Env::Var, Env> _R;
		typedef typename _R::Expression Expression;
		typedef typename _R::Environ Environ;
		enum{
			Offset = _R::Offset
		};
		static inline const Expression& get(const Expression* e)
		{
			return _R::get(e);
		}
	};
}
#endif
