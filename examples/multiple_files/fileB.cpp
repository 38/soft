#include <spatialops.hpp>
using namespace SpatialOps;
Field<double> get_field_B()
{
	Field<double> r(0,0,0,2,2,2);
	print_expr(r <<= 2);
	r <<= 2;
	return r;
}

