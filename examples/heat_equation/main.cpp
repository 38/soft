#include <spatialops.hpp>
using namespace SpatialOps;
#define SZ 600
const double deltaX = 1.0/SZ;
const double deltaY = 1.0/SZ;
const double sqrdDeltaX = deltaX * deltaX;
const double sqrdDeltaY = deltaY * deltaY;
const double sqrdDeltaXYmult = sqrdDeltaX * sqrdDeltaY;
const double sqrdDeltaXYplus = sqrdDeltaX + sqrdDeltaY;

/* Define "variables" for let binding */
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
	Field<double> rhs(0, 0 ,0, SZ, SZ, 1);
	LValueScalar<double> deltaT;
	
	/* For some constant field, we don't need to allocate a field, we can evaluate on the fly */
	DEFINE_EXPRESSION(alpha, 1);
	
	/* Reduction */
	deltaT <<= INT_MAX;
	deltaT <<= min(deltaT, alpha);
	deltaT <<= 0.25 * sqrdDeltaXYmult / (sqrdDeltaXYplus  * deltaT);
	
	phi <<= 5.0;
	set_boundary(phi);
	
	/* Do the iteration! */
	int nSteps = 1000;
	for(int i = 0; i < nSteps; i ++)
	{
		rhs <<= let<X>(alpha,let<Y>(phi,(DivR<XDir>( Interp<XDir>(ref<X>()) * Div<XDir>(ref<Y>())) +
		                                 DivR<YDir>( Interp<YDir>(ref<X>()) * Div<YDir>(ref<Y>()))) * SZ * SZ));
		phi <<= phi + deltaT * rhs;
		set_boundary(phi);
	}
	
	printf("The result temperature field after %d iterations:\n\n", nSteps);
	//phi.print(std::cout);
	
	return 0;
}
