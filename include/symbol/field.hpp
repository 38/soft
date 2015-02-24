#ifndef __PRIMITIVES_HPP__
#define __PRIMITIVES_HPP__
#include <iostream>
#include <stdlib.h>
namespace SpatialOps{
	/**
	 * @brief the type for a spatial field
	 * @detials This actually a field of symbol, DO NOT ALLOCATE ACTUALL MEMORY
	 *          ON ANY DEVICE
	 * @param T element type
	 **/
	template <typename T>
	struct Field{
		typedef T element_type;   /*!< the element type */
		/**
		 * @brief construct a field within the given range
		 **/
		Field(int lx, int ly, int lz,
			  int hx, int hy, int hz)
		{
			_high[0] = hx;
			_high[1] = hy;
			_high[2] = hz;
			_low[0] = lx;
			_low[1] = ly;
			_low[2] = lz;
			static unsigned next_id = 0;
			_identifier = (next_id++);
		}
		inline const char* name() const
		{
			static char _buf[100];
			snprintf(_buf, 100, "Field_%d", _identifier);
			return _buf;
		}
		void inline get_range(int& lx, int& ly, int&lz,
							  int& hx, int& hy, int&hz) const
		{
			lx = _low[0];
			ly = _low[1];
			lz = _low[2];
			hx = _high[0];
			hy = _high[1];
			hz = _high[2];
		}
		~Field()
		{
			InvokeDeviceDestructor<>::notify(_identifier);
		}
		template <int DeviceId>
		T* get_memory() const
		{
			size_t size = (_high[0] - _low[0]) *
						  (_high[1] - _low[1]) *
						  (_high[2] - _low[2]) *
						  sizeof(T);
			return (T*)GetDeviceRuntimeEnv<DeviceId>::R::allocate(_identifier, size);
		}
		inline int getid(){return _identifier;}

		template <int DeviceId>
		inline const void print() const
		{
			size_t size = (_high[0] - _low[0]) *
						  (_high[1] - _low[1]) *
						  (_high[2] - _low[2]) *
						  sizeof(T);
			T* _device_memory = (T*)GetDeviceRuntimeEnv<DeviceId>::R::allocate(_identifier, size);
			for(int z = _low[2]; z < _high[2]; z ++)
			{
				for(int y = _low[1]; y < _high[1]; y ++)
				{
					for(int x = _low[0]; x < _high[0]; x ++)
						std::cout << _device_memory[x + (_high[0] - _low[0]) * y + (_high[0] - _low[0]) * (_high[1] - _low[1]) * z] << ' ';
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}

		private:
			int _high[3];  /**!< the high bound of the field **/
			int _low[3];   /**!< the low bound of the field **/
			unsigned _identifier; /**!< the unique identifier of this field */
	};
	template <typename T>
	struct GetRange<Field<T> >{
		static const inline void get_range(const Field<T>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)\
		{
			e.get_range(lx, ly, lz, hx, hy, hz);
		}
	};
	/* # of operand is 0, just use default */

	/* Type inference class */
	template <typename T>
	struct ExprTypeInfer<Field<T> >{
		typedef T& R;
	};
};
#endif
