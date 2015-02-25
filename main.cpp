#include <spatialops.hpp>
using namespace SpatialOps;
int main()
{
	/*Field<double> f(0,0,0,100,1,1);
	Field<double> g(0,0,0,100,1,1);

	f <<= square(coordinate<XDir>() * 0.000001);
	g <<= Div<XDir>(f) / 0.000001 - (coordinate<XDir>() * 0.000002);

	Field<bool> k(0,0,0,10,10,10);
	print_expr(k <<= (coordinate<XDir>() + coordinate<YDir>() + coordinate<ZDir>() == 10));*/

	Field<double> phi(-1,-1,0,7,7,1);
	Field<double> rhs(-1,-1,0,7,7,1);
	Field<double> alpha(-1,-1,0,7,7,1);


	alpha <<= 1.0;

	phi <<= 5.0;
	phi <<= window(10.0, 0.0, -1, -1,0 , 0, 7, 1);
	phi <<= window(0.0, 0.0, 6, -1, 0, 7, 7 , 1);

	int nSteps = 5;
	for(int i = 0; i < nSteps; i ++)
	{
		rhs <<= Div<XDir>( Interp<XDir>(alpha) * Div<XDir>(phi) / 6.0) +
				Div<YDir>( Interp<YDir>(alpha) * Div<YDir>(phi) / 6.0);
		phi <<= phi + 0.3 * rhs;
		
		phi <<= window(10.0, 0.0, -1, -1,0 , 
				                   0, 7, 1);
		phi <<= window(0.0, 0.0, 6, -1, 0, 
				                 7,  7, 1);

		phi <<= window(5.0, 0.0, -1, -1, 0, 
				                  7,  0, 1);
		phi <<= window(5.0, 0.0, -1,  6, 0, 
				                  7,  7, 1);
	}
	phi.print();
	//typedef typeof(link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10 + coordinate<YDir>())) T;
	//T t = link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10  + coordinate<YDir>()); 
	//GetExecutor<DEVICE_TYPE_CPU>::execute(1,2,4,t);
	//DataValidator<DEVICE_TYPE_CPU, T, GetNumOperands<T::Symbol>::R> init(t);

	return 0;
}
