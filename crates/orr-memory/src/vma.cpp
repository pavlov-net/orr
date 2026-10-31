// VMA's implementation. Failed checks go to Rust, which logs and aborts (checks) or logs
// (leaks), so no C runtime dialog opens.
extern "C" void orr_memory_check_failed(const char* expr, const char* file, int line);
extern "C" void orr_memory_leak(const char* expr);

#ifdef ORR_MEMORY_CHECKS
#define VMA_ASSERT(expr) ((expr) ? (void)0 : orr_memory_check_failed(#expr, __FILE__, __LINE__))
#else
#define VMA_ASSERT(expr)
#endif
#define VMA_ASSERT_LEAK(expr) ((expr) ? (void)0 : orr_memory_leak(#expr))

#define VMA_IMPLEMENTATION
#include "vma.h"

static_assert(VMA_VERSION == VK_MAKE_VERSION(3, 5, 0), "src/bindings.rs is generated from VMA 3.5.0");
