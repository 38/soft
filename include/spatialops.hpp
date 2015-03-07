/**
 * @brief The top-level header file
 **/
#ifndef __SPATIALOPS_HPP__
#define __SPATIALOPS_HPP__


/* configuration */
#include <configure/devices.hpp>

/* device management */
#include <util/device_manager.hpp>

/* memory management */
#include <util/memory_manager.hpp>

/* environ uitls */
#include <util/environ.hpp>

/* symbol definitions */
#include <symbol/interface.hpp>
#include <symbol/field.hpp>
#include <symbol/mathfunc.hpp>
#include <symbol/shift.hpp>
#include <symbol/window.hpp>
#include <symbol/coorinate.hpp>
#include <symbol/basicops.hpp>
#include <symbol/lvalue.hpp>
#include <symbol/annotation.hpp>
#include <symbol/let.hpp>

/* define functions */
#include <functions/math.hpp>

/* the printing backend for debugging */
#include <util/printing_backend.hpp>

/* the linker templates */
#include <linker/linker.hpp>

/* the CPU Library */
#include <lib/CPU.hpp>

/* the CUDA Library */
#ifdef __CUDACC__
#include <lib/GPU.hpp>
#endif

/* runtime environment */
#include <runtime/runtime.hpp>
#include <runtime/CPU.hpp>

#ifdef __CUDACC__
#include <runtime/GPU.hpp>
#endif

#define DEFINE_FORMULA(name, expr) typeof(expr) name = expr
#endif
