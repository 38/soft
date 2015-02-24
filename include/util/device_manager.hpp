#include <map>
#ifndef __DEVICE_MANAGER_HPP__
#define __DEVICE_MANAGER_HPP__
namespace SpatialOps{

	/* template <typename Expr, typename Exec> Library */
	/** @brief Get the runtime environment for this device */
	template <int DeviceType> struct GetDeviceRuntimeEnv;  
	/** @brief Get the linker library for this device */
	template <int DeviceType, typename Expr, typename Executable> struct InvokeDeviceLibrary;
	/** @brief Get the execute method for this device */
	template <int DeviceType> struct GetExecutor;

	template <int DeviceType = 0> 
	struct InvokeDeviceMM{
		struct MemoryMapValue{
			typedef typename GetDeviceRuntimeEnv<DeviceType>::R::DeviceMemory dev_mem_ptr_t;
			unsigned ref_cnt;
			unsigned size;
			dev_mem_ptr_t mem;
			int timestamp;
			inline MemoryMapValue(unsigned sz): ref_cnt(0), mem(GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer()), size(sz), timestamp(0){}
			inline void incref()
			{
				ref_cnt ++;
			}
			inline bool decref()
			{
				if(ref_cnt > 0) ref_cnt --;
				if(ref_cnt == 0 && GetDeviceRuntimeEnv<DeviceType>::R::is_valid_pointer(mem)) 
				{
					GetDeviceRuntimeEnv<DeviceType>::R::deallocate(mem);
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
		
		static inline std::map<int, MemoryMapValue*>& get_memory_map()
		{
			static std::map<int, MemoryMapValue*> _memory_map;
			return _memory_map;
		}

		static inline void notify_construct(int id, unsigned size)
		{
			if(get_memory_map().find(id) == get_memory_map().end())
				get_memory_map()[id] = new MemoryMapValue(size);
			get_memory_map()[id]->incref();
			InvokeDeviceMM<DeviceType + 1>::notify_construct(id);
		}
		static inline void notify_destruct(int id)
		{
			if(get_memory_map().find(id) != get_memory_map().end() && get_memory_map()[id]->decref())
			{
				delete get_memory_map()[id];
				get_memory_map().erase(get_memory_map().find(id));
			}
			InvokeDeviceMM<DeviceType + 1>::notify_destruct(id);
		}

		static inline void* get_memory(int id)
		{
			if(get_memory_map().find(id) == get_memory_map().end()) return GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer();
			if(get_memory_map()[id]->mem == GetDeviceRuntimeEnv<DeviceType>::R::get_null_pointer()) 
				get_memory_map()[id]->mem = GetDeviceRuntimeEnv<DeviceType>::R::allocate(get_memory_map()[id]->size);
			return (void*)get_memory_map()[id]->mem;
		}

		static inline void set_timestamp(int id, int ts)
		{
			get_memory_map()[id]->update_ts(ts);
		}

		static inline int find_up_to_dated_copy(int id, int target_ts)
		{
			if(get_memory_map()[id]->timestamp == target_ts) return DeviceType;
			return InvokeDeviceMM<DeviceType + 1>::find_up_to_dated_copy(id, target_ts);
		}

		static inline void copy_to_host(int id, int device, void* dest)
		{
			if(device == DeviceType) 
				GetDeviceRuntimeEnv<DeviceType>::R::copy_to_host(dest, get_memory_map()[id]->mem, get_memory_map()[id]->size);
			else
				InvokeDeviceMM<DeviceType + 1>::copy_to_host(id, device, dest);
		}
		
		static inline void synchronize_device(int id, int uptodate_dev)
		{
			/* if the device is already updated */
			if(uptodate_dev == DeviceType) return;
			/* get the CPU memory as buffer for transfer */
			void* host_mem = InvokeDeviceMM<DEVICE_TYPE_CPU>::get_memory(id);
			
			/* then copy the device memory to cpu */
			if(uptodate_dev != DEVICE_TYPE_CPU) 
				InvokeDeviceMM<0>::copy_to_host(id, uptodate_dev, host_mem);
			
			/* finally copy the host memory to target device */
			if(DeviceType != DEVICE_TYPE_CPU)
				GetDeviceRuntimeEnv<DeviceType>::R::copy_from_host(get_memory_map()[id]->mem, host_mem, get_memory_map()[id]->size); 
		}
	};
	template <>
	struct InvokeDeviceMM<NUM_DEVICE_TYPES>{
		static inline void notify_destruct(int id){}
		static inline void notify_construct(int id){}
		static inline int find_up_to_dated_copy(int id, int target_ts){return -1;}
		static inline void copy_to_host(int id, int device, void* dest){}
	};

}
#endif
