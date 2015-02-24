#include <spatialops.hpp>
using namespace SpatialOps;
int main()
{
	Field<double> f(0,0,0,10,10,10);

	f <<= (2 < window(3, 0, 0,0,0, 5,5,5));


	f.print<DEVICE_TYPE_CPU>();
	
	//typedef typeof(link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10 + coordinate<YDir>())) T;
	//T t = link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10  + coordinate<YDir>()); 
	//GetExecutor<DEVICE_TYPE_CPU>::execute(1,2,4,t);
	//DataValidator<DEVICE_TYPE_CPU, T, GetNumOperands<T::Symbol>::R> init(t);

	return 0;
}
