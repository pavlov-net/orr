// orr_assert(cond, message, args...): with `ASSERTS` linked, an invocation for which `cond`
// is false prints `<file>:<line>: <message>` with `args` (printf format) and aborts with the
// same text, without the arguments, which loses the device. Otherwise the check compiles
// out. `message` is a string literal: `abort` takes only literals. Macros do not cross
// `import`, so kernels include this file: `#include "orr/debug.h"`.
#pragma once
import orr.debug;

#define ORR_STRINGIFY_(x) #x
#define ORR_STRINGIFY(x) ORR_STRINGIFY_(x)
#define ORR_HERE __FILE__ ":" ORR_STRINGIFY(__LINE__) ": "
#define ORR_FIRST_(message, ...) message
#define orr_assert(cond, ...)                                                              \
    do                                                                                     \
    {                                                                                      \
        if (ASSERTS && !(cond))                                                            \
        {                                                                                  \
            __target_switch                                                                \
            {                                                                              \
            case metal:                                                                    \
                break;                                                                     \
            default:                                                                       \
                printf(ORR_HERE __VA_ARGS__);                                              \
                abort(ORR_HERE ORR_FIRST_(__VA_ARGS__, 0));                                \
            }                                                                              \
        }                                                                                  \
    } while (false)
