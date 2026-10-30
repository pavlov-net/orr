// Build-time debug switches. `orr_assert` is a macro in `orr/debug.h`.
module debug;

// True in builds made with `Build::asserts(true)`, which link a module exporting it.
// Checks guarded by it compile out otherwise.
public extern static const bool ASSERTS = false;
