#ifndef __PRIMITIVES_HPP__
#define __PRIMITIVES_HPP__
#include <iostream>
#include <stdlib.h>
namespace SpatialOps{
	static unsigned next_id = 0;
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
			_identifier = (next_id++);
			size_t size = (_high[0] - _low[0]) *
						  (_high[1] - _low[1]) *
						  (_high[2] - _low[2]) *
						  sizeof(T);
			InvokeDeviceMM<>::notify_construct(_identifier, size);
		}
		Field(const Field& f) : _identifier(f._identifier)
		{
			_high[0] = f._high[0];
			_high[1] = f._high[1];
			_high[2] = f._high[2];
			_low[0] = f._low[0];
			_low[1] = f._low[1];
			_low[2] = f._low[2];
			size_t size = (_high[0] - _low[0]) *
						  (_high[1] - _low[1]) *
						  (_high[2] - _low[2]) *
						  sizeof(T);
			InvokeDeviceMM<>::notify_construct(_identifier, size);
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
			InvokeDeviceMM<>::notify_destruct(_identifier);
		}
		template <int DeviceId>
		T* get_memory() const
		{
			int _timestamp = -1;
			T* ret = (T*)InvokeDeviceMM<DeviceId>::get_memory(_identifier);
			int last_valid = InvokeDeviceMM<>::find_up_to_dated_copy(_identifier, _timestamp);
			InvokeDeviceMM<DeviceId>::synchronize_device(_identifier, last_valid);
			InvokeDeviceMM<DeviceId>::set_timestamp(_identifier, ++_timestamp);
			return ret;
		}
		inline int getid(){return _identifier;}

		inline void print() const
		{
			T* _device_memory = get_memory<DEVICE_TYPE_CPU>();
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
		static inline void get_range(const Field<T>& e, int& lx, int& ly, int& lz, int& hx, int& hy, int& hz)\
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
