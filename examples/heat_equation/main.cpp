#include <spatialops.hpp>
using namespace SpatialOps;

const double deltaX = 1/6.0;
const double deltaY = 1/6.0;
const double sqrdDeltaX = deltaX * deltaX;
const double sqrdDeltaY = deltaY * deltaY;
const double sqrdDeltaXYmult = sqrdDeltaX * sqrdDeltaY;
const double sqrdDeltaXYplus = sqrdDeltaX + sqrdDeltaY;

struct X{typedef double T;};
struct Y{typedef double T;};
int main()
{
	Field<double> phi(-1,-1,0,7,7,1);
	Field<double> rhs(-1,-1,0,7,7,1);
	Field<double> alpha(-1,-1,0,7,7,1);

	alpha <<= 1;

	/* Reduction */
	LValueScalar<double> deltaT;
	deltaT <<= INT_MAX;
	deltaT <<= min(deltaT, alpha);
	deltaT <<= 0.25 * sqrdDeltaXYmult / (sqrdDeltaXYplus  * deltaT);
	
	rhs <<= 0;
	
	phi <<= 5.0;
	phi <<= window(10.0, 0.0, -1, -1,0 , 0, 7, 1);
	phi <<= window(0.0, 0.0, 6, -1, 0, 7, 7 , 1);

	int nSteps = 40;
	for(int i = 0; i < nSteps; i ++)
	{
		rhs <<= let<X>(alpha,let<Y>(phi,(DivR<XDir>( Interp<XDir>(ref<X>()) * Div<XDir>(ref<Y>())) +
		                                 DivR<YDir>( Interp<YDir>(ref<X>()) * Div<YDir>(ref<Y>()))) * 36.0));
		phi <<= phi + deltaT * rhs;

		phi <<= phi;
		
		phi <<= window(5.0, 0.0, -1, -1, 0, 
		                          7,  0, 1);
		phi <<= window(5.0, 0.0, -1,  6, 0, 
		                          7,  7, 1);
		
		phi <<= window(10.0, 0.0, -1, -1,0 , 
		                           0, 7, 1);
		phi <<= window(0.0, 0.0, 6, -1, 0, 
		                         7,  7, 1);
		phi.print();
	}
	return 0;
}
