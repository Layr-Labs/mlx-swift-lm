// Linux only. Glibc declares `stdout` as a plain mutable global, and Swift 6
// rejects a reference to it unless it is found through a `@preconcurrency`
// import. Foundation re-exports Glibc without `@preconcurrency`, so in a file
// that imports Foundation the reference can resolve through Foundation and fail.
// This file imports only Glibc, so `stdout` has one import path.

#if canImport(Glibc)
    @preconcurrency import Glibc

    /// Makes stdout unbuffered, so a trap cannot lose buffered results.
    func makeStdoutUnbuffered() {
        setvbuf(stdout, nil, _IONBF, 0)
    }
#endif
