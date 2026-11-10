/* include/numkit/control/c_api.h
 *
 * Self-contained C-ABI for the `numkit/control` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_control_resolve` + `numkit_control_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_control_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_CONTROL_C_API_H
#define NUMKIT_CONTROL_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_CONTROL_API __declspec(dllexport)
#  elif defined(NUMKIT_CONTROL_STATIC)
#    define NUMKIT_CONTROL_API
#  else
#    define NUMKIT_CONTROL_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_CONTROL_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_CONTROL_API
#  endif
#endif

/* Opaque isolated `control` toolbox context */
typedef struct numkit_control_ctx_s numkit_control_ctx;

/* Create an isolated `control` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_CONTROL_API numkit_control_ctx *numkit_control_ctx_create(numkit_error *err);

/* Destroy `control` toolbox context and release its resources. */
NUMKIT_CONTROL_API void numkit_control_ctx_destroy(numkit_control_ctx *ctx);

/* Seed the `control` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_CONTROL_API void numkit_control_seed_rng(numkit_control_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `control`, 0 otherwise. */
NUMKIT_CONTROL_API int numkit_control_has_function(const numkit_control_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_CONTROL_API int numkit_control_register_fn(numkit_control_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_CONTROL_API const numkit_fn *numkit_control_resolve(const numkit_control_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_CONTROL_API int numkit_control_invoke(numkit_control_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `control` function or object method `name`. */
NUMKIT_CONTROL_API int numkit_control_call(numkit_control_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `control` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_CONTROL_API size_t numkit_control_format(const numkit_control_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_CONTROL_C_API_H */
