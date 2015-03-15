/**
 * @brief The top-level header file
 **/
#ifndef __SPATIALOPS_HPP__
#define __SPATIALOPS_HPP__

/* helper types */
#include <util/static_vars.hpp>

/* configuration */
#include <configure/devices.hpp>

/* device management */
#include <runtime/device_manager.hpp>

/* memory management */
#include <runtime/memory_manager.hpp>

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
#include <lib/CPU/lib.hpp>

/* the CUDA Library */
#ifdef __CUDACC__
#include <lib/CUDA/lib.hpp>
#endif

/* basic types for runtime environment */
#include <runtime/runtime.hpp>

/* set up the CPU runtime */
#include <runtime/CPU/runtime.hpp>

#ifdef __CUDACC__
/* set up the GPU runtime */
#include <runtime/CUDA/runtime.hpp>
#endif

/* Define helper macros */
/**
 * @brief Define a named symbolic expression, this is useful when you
 *        want to use a result of an expression again and again, instead
 *        of allocating a memory for the result of this epxression,
 *        you can make it a runtime generated value
 * @param name the name of this expression
 * @param expr the expression
 **/
#define DEFINE_EXPRESSION(name, expr) typeof(expr) name = expr
#endif
