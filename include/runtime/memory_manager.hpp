#include <map>
#include <stdio.h>
#ifndef __UTIL_MEMORY_MANAGER_HPP__
#define __UTIL_MEMORY_MANAGER_HPP__
namespace SpatialOps {
	template <int DeviceType = 0>
	struct InvokeDeviceMM{
		struct MemoryMapValue{
			typedef typename GetDeviceRuntimeEnv<DeviceType>::R::DeviceMemory dev_mem_ptr_t;
			unsigned ref_cnt;
			unsigned size;
			dev_mem_ptr_t mem;
			int timestamp;
			inline MemoryMapValue(unsigned sz): ref_cnt(0),  size(sz), mem(GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer()), timestamp(-1){}
			inline void incref()
			{
				ref_cnt ++;
			}
			inline bool decref()
			{
				if(ref_cnt > 0) ref_cnt --;
				if(ref_cnt == 0 && GetDeviceRuntimeEnv<DeviceType>::R::is_valid_pointer(mem))
				{
					if(!GetDeviceRuntimeEnv<DeviceType>::R::deallocate(mem))
					{
						fprintf(stderr, "Warning: memory deallocation failed\n");
						return false;
					}
					mem = GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer();
					return true;
				}
				return false;
			}
			inline void update_ts(int ts)
			{
				timestamp = ts;
			}
		};
		
		typedef StaticVar<InvokeDeviceMM, std::map<int, MemoryMapValue*> > memory_map;
		
		#ifdef AUTO_MEMORY_COPY_SUPPRESS
		typedef StaticVar<InvokeDeviceMM, bool> disable_memory_copy;
		
		static inline void disable_automatic_memcpy()
		{
			disable_memory_copy::get() = false;
		}
		
		static inline void enable_automatic_memcpy()
		{
			disable_memory_copy::get() = true;
		}
		#endif
		
		static inline void notify_construct(int id, unsigned size)
		{
			if(memory_map::get().find(id) == memory_map::get().end())
			    memory_map::get()[id] = new MemoryMapValue(size);
			memory_map::get()[id]->incref();
			InvokeDeviceMM<DeviceType + 1>::notify_construct(id, size);
		}
		static inline void notify_destruct(int id)
		{
			if(memory_map::get().find(id) != memory_map::get().end() && memory_map::get()[id]->decref())
			{
				delete memory_map::get()[id];
				memory_map::get().erase(memory_map::get().find(id));
			}
			InvokeDeviceMM<DeviceType + 1>::notify_destruct(id);
		}
		
		static inline void* get_memory(int id)
		{
			/*  If the memory entry is not found that means this field has been deallocated already */
			if(memory_map::get().find(id) == memory_map::get().end())
			    return GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer();
			if(!GetDeviceRuntimeEnv<DeviceType>::R::is_valid_pointer(memory_map::get()[id]->mem))
			    memory_map::get()[id]->mem = GetDeviceRuntimeEnv<DeviceType>::R::allocate(memory_map::get()[id]->size);
			return (void*)memory_map::get()[id]->mem;
		}
		
		static inline void set_timestamp(int id, int ts)
		{
			memory_map::get()[id]->update_ts(ts);
		}
		
		static inline int find_up_to_dated_copy(int id, int& max_ts)
		{
			if(memory_map::get()[id]->timestamp > max_ts) max_ts = memory_map::get()[id]->timestamp;
			int r = InvokeDeviceMM<DeviceType + 1>::find_up_to_dated_copy(id, max_ts);
			if(r == -1 && memory_map::get()[id]->timestamp == max_ts) return DeviceType;
			return -1;
		}
		
		static inline bool copy_to_host(int id, int device, void* dest)
		{
			if(device == DeviceType)
			    return GetDeviceRuntimeEnv<DeviceType>::R::copy_to_host(dest, memory_map::get()[id]->mem, memory_map::get()[id]->size);
			else
			    return InvokeDeviceMM<DeviceType + 1>::copy_to_host(id, device, dest);
		}
		
		static inline bool synchronize_device(int id, int uptodate_dev)
		{
			/* if the device is already updated */
			if(uptodate_dev == DeviceType) return true;
			#ifdef AUTO_MEMORY_COPY_SUPPRESS
			/* if the auto copy is disabled on this device */
			if(disable_memory_copy::get()) return false;
			#endif
			/* get the CPU memory as buffer for transfer */
			void* host_mem = InvokeDeviceMM<DEVICE_TYPE_CPU>::get_memory(id);
			
			/* then copy the device memory to CPU */
			if(uptodate_dev != (int)DEVICE_TYPE_CPU && !InvokeDeviceMM<0>::copy_to_host(id, uptodate_dev, host_mem))
			    return false;
			
			/* finally copy the host memory to target device */
			if(DeviceType != (int)DEVICE_TYPE_CPU &&
			   !GetDeviceRuntimeEnv<DeviceType>::R::copy_from_host(memory_map::get()[id]->mem, host_mem, memory_map::get()[id]->size))
			    return false;
			
			return true;
		}
	};
	template <>
	struct InvokeDeviceMM<NUM_DEVICE_TYPES>{
		static inline void notify_destruct(int id){}
		static inline void notify_construct(int id, int size){}
		static inline int find_up_to_dated_copy(int id, int target_ts){return -1;}
		static inline bool copy_to_host(int id, int device, void* dest){return false;}
	};
}
#endif
