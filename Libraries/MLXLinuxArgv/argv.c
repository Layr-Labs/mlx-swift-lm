// Linux: exact CommandLine.arguments in every executable and test that links
// MLXLMCommon.
//
// Swift on Linux finds argv by walking back from `environ` on the process
// stack (Swift stdlib, stdlib/public/CommandLineSupport/CommandLine.cpp,
// ArgvGrabber). A shared library that calls setenv while it loads makes glibc
// move `environ` to the heap first; librocblas (ROCm) does this. Swift then
// finds no argv, and CommandLine.arguments is empty.
//
// glibc passes (argc, argv, envp) to ELF constructors, and that argv is the
// original vector on the stack, so it is exact, empty arguments included. The
// constructor gives it to the Swift runtime override, which
// _swift_stdlib_getUnsafeArgvArgc reads before the platform lookup.
//
// _swift_stdlib_overrideUnsafeArgvArgc is an exported but underscored runtime
// hook (Swift 6.3.3 CommandLine.cpp:48-52), not public API. It is declared
// weak, so a runtime without it leaves CommandLine unchanged.

#if defined(__linux__)

#include <stddef.h>

extern void _swift_stdlib_overrideUnsafeArgvArgc(char **argv, int argc)
    __attribute__((swiftcall, weak));

__attribute__((constructor)) static void mlx_linux_argv(int argc, char **argv, char **envp) {
  (void)envp;
  if (_swift_stdlib_overrideUnsafeArgvArgc != NULL && argv != NULL && argc > 0) {
    _swift_stdlib_overrideUnsafeArgvArgc(argv, argc);
  }
}

#endif
