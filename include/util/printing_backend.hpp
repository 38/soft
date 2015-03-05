#ifndef __PRINT_BACKEND_HPP__
#define __PRINT_BACKEND_HPP__
/* printing backend */
#include <typeinfo>
namespace SpatialOps{
	template <typename Expr, int NOperands>
	struct Print_Implemenation;
	static int _level;
	template <typename Expr>
	struct Print{
		typedef Print_Implemenation<Expr, GetNumOperands<Expr>::R > writer_t;
		Print(const Expr& e){
			writer_t writer(e);
		}
	};
	static inline void print_indentation()
	{
		for(int i = 0; i < _level; i ++, putchar('\t'));
	}
	template < >
	struct Print_Implemenation<double, 0>{
		Print_Implemenation(const double& e){
			print_indentation();
			printf("%lf", e);
		}
	};
	template < >
	struct Print_Implemenation<int, 0>{
		Print_Implemenation(const int& e){
			print_indentation();
			printf("%d", e);
		}
	};
	template <typename Expr>
	struct Print_Implemenation<Expr, 0>{
		Print_Implemenation(const Expr& e){
			print_indentation();
			printf("%s", e.name());
		}
	};
	template <typename Expr>
	struct Print_Implemenation<Expr, 1>{
		Print_Implemenation(const Expr& e){
			print_indentation();
			printf("%s {\n", e.name());
			_level ++;
			Print_Implemenation<typename Expr::Operand, GetNumOperands<typename Expr::Operand>::R> writer(e.operand);
			_level --;
			puts("");
			print_indentation();
			printf("} /* %s */", e.name());
		}
	};
	template <typename Expr>
	struct Print_Implemenation<Expr, 2>{
		Print_Implemenation(const Expr& e){
			print_indentation();
			printf("%s {\n", e.name());
			_level ++;
			Print_Implemenation<typename Expr::Operand_l, GetNumOperands<typename Expr::Operand_l>::R> writer1(e.operand_l);
			printf(",\n");
			Print_Implemenation<typename Expr::Operand_r, GetNumOperands<typename Expr::Operand_r>::R> writer2(e.operand_r);
			_level --;
			puts("");
			print_indentation();
			printf("} /* %s */", e.name());
		}
	};
	template <typename Expr>
	static inline void print_expr(const Expr& expr)
	{
		Print<Expr> writer(expr);
		int lx, ly, lz, hx, hy, hz;
		GetRange<Expr, EmptyEnv>::get_range(expr, lx, ly, lz, hx, hy, hz);
		printf("\n[from (%d, %d, %d) to (%d, %d, %d)] type=%s\n", lx, ly, lz, hx, hy, hz, typeid(typename ExprTypeInfer<Expr>::R).name());
	}
};
#endif
