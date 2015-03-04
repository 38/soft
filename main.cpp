#include <spatialops.hpp>
using namespace SpatialOps;

const double deltaX = 1/6.0;
const double deltaY = 1/6.0;
const double sqrdDeltaX = deltaX * deltaX;
const double sqrdDeltaY = deltaY * deltaY;
const double sqrdDeltaXYmult = sqrdDeltaX * sqrdDeltaY;
const double sqrdDeltaXYplus = sqrdDeltaX + sqrdDeltaY;

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
		
		rhs <<= DivR<XDir>( Interp<XDir>(alpha) * Div<XDir>(phi) * 6.0) * 6.0 +
				DivR<YDir>( Interp<YDir>(alpha) * Div<YDir>(phi) * 6.0) * 6.0;
		phi <<= phi + deltaT * rhs;
		
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
	deltaT.operand.print();
	return 0;
}
