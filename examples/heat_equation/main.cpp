#include <spatialops.hpp>
using namespace SpatialOps;
#define SZ 6
const double deltaX = 1.0/SZ;
const double deltaY = 1.0/SZ;
const double sqrdDeltaX = deltaX * deltaX;
const double sqrdDeltaY = deltaY * deltaY;
const double sqrdDeltaXYmult = sqrdDeltaX * sqrdDeltaY;
const double sqrdDeltaXYplus = sqrdDeltaX + sqrdDeltaY;

struct X{typedef double T;};
struct Y{typedef double T;};
void set_boundary(Field<double>& phi)
{
	phi <<= window(5.0, 0.0, -1, -1, 0, SZ + 1,  0, 1);
	phi <<= window(5.0, 0.0, -1,  SZ, 0, SZ + 1,  SZ + 1, 1);
	phi <<= window(10.0, 0.0, -1, -1,0 , 0, SZ + 1, 1);
	phi <<= window(0.0, 0.0, SZ, -1, 0, SZ + 1,  SZ + 1, 1);
}
int main()
{
	Field<double> phi(-1,-1,0,SZ + 1,SZ + 1,1);
	Field<double> rhs(-1,-1,0,SZ + 1,SZ + 1,1);
	LValueScalar<double> deltaT;
	DEFINE_FORMULA(alpha, 1);   /* For some constant field, we don't need to allocate a field, we can compute it on the fly */
	
	/* Reduction */
	deltaT <<= INT_MAX;
	deltaT <<= min(deltaT, alpha);
	deltaT <<= 0.25 * sqrdDeltaXYmult / (sqrdDeltaXYplus  * deltaT);
	
	phi <<= 5.0;
	set_boundary(phi);
	
	/* initialize the RHS, since there might be some uninitialized values */
	rhs <<= 0;

	/* Do the iteration! */
	int nSteps = 40;
	for(int i = 0; i < nSteps; i ++)
	{
		/* Instead of the Stencil in Nebo, we use the let binding to solve the same problem */
		rhs <<= let<X>(alpha,let<Y>(phi,(DivR<XDir>( Interp<XDir>(ref<X>()) * Div<XDir>(ref<Y>())) +
		                                 DivR<YDir>( Interp<YDir>(ref<X>()) * Div<YDir>(ref<Y>()))) * SZ * SZ));
		phi <<= phi + deltaT * rhs;
		set_boundary(phi);
	}
	
	phi.print();
	
	return 0;
}
