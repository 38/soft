#include <spatialops.hpp>
using namespace SpatialOps;
int main()
{
	Field<double> f(0,0,0,20000,1,1);
	Field<double> g(0,0,0,20000,1,1);

	f <<= (coordinate<XDir>() * 0.0001) * (coordinate<XDir>() * 0.0001);

	g <<= (f - shift<-1,0,0>(f)) / 0.0001 - (coordinate<XDir>() * 0.0002);

	f.print();
	g.print();
	
	//typedef typeof(link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10 + coordinate<YDir>())) T;
	//T t = link<DEVICE_TYPE_CPU>(f <<= -(f + 1) + coordinate<XDir>() * 10  + coordinate<YDir>()); 
	//GetExecutor<DEVICE_TYPE_CPU>::execute(1,2,4,t);
	//DataValidator<DEVICE_TYPE_CPU, T, GetNumOperands<T::Symbol>::R> init(t);

	return 0;
}
