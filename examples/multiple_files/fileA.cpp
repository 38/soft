#include <spatialops.hpp>
using namespace SpatialOps;
Field<double> get_field_A()
{
	Field<double> r(0,0,0,2,2,2);
	print_expr(r <<= 1);
	r <<= 1;
	return r;
}

