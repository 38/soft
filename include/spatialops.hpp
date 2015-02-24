/**
 * @brief The top-level header file
 **/
#ifndef __SPATIALOPS_HPP__
#define __SPATIALOPS_HPP__


/* configuration */
#include <configure/devices.hpp>

/* device management */
#include <util/device_manager.hpp>

/* symbol definitions */
#include <symbol/interface.hpp>
#include <symbol/field.hpp>
#include <symbol/mathfunc.hpp>
#include <symbol/shift.hpp>
#include <symbol/window.hpp>
#include <symbol/coorinate.hpp>
#include <symbol/basicops.hpp>

/* the printing backend for debugging */
#include <util/printing_backend.hpp>

/* the linker templates */
#include <linker/linker.hpp>

/* the CPU Library */
#include <lib/CPU.hpp>

#ifdef __CUDACC__
#include <lib/GPU.hpp>
#endif

/* runtime environment */
#include <runtime/runtime.hpp>
#include <runtime/CPU.hpp>
#endif
