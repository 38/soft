#include <spatialops.hpp>
using namespace SpatialOps;
extern Field<double> get_field_A();
extern Field<double> get_field_B();
int main()
{
	Field<double> A = get_field_A();
	Field<double> B = get_field_B();
	Field<double> C(0,0,0,2,2,2);
	C <<= A + B;
	print_expr(C <<= A + B);
	C.dump(std::cout);
	return 0;
}
