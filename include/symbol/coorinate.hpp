#ifndef __COORDINATE_HPP__
#define __COORDINATE_HPP__
namespace SpatialOps{
	/* Symbol For Coordinate */
	template<typename Dir>
	struct symbol_coordinate{
		enum{
			X = GetDirectVec<Dir>::X,
			Y = GetDirectVec<Dir>::Y,
			Z = GetDirectVec<Dir>::Z
		};
		const inline char* name() const{
			return "Coordinate";
		}
	};
	template<typename Dir>
	symbol_coordinate<Dir> coordinate()
	{
		return symbol_coordinate<Dir>();
	}
	/* The coordinate operator always return a interger value */
	template<typename Dir, typename Env = EmptryEnv>
	struct ExprTypeInfer<symbol_coordinate<Dir> >{
		typedef int R;
	};
}
#endif /* __COORDINATE_HPP__ */
