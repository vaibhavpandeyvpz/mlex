use std::cell::RefCell;
use std::sync::Once;

use crate::error::install_error_handler;

struct StreamHolder(crate::sys::mlx_stream);

// MLX's GPU backend binds a stream's command-buffer machinery to whatever
// OS thread created it ("no Stream(gpu, 0) in current thread" if used from
// another thread) - so each thread gets its own lazily-created stream
// rather than one process-wide stream shared (and potentially used) across
// threads. This matters once test binaries run multiple `#[test]`s (each
// on its own thread) that touch array ops, not just the single-threaded
// CLI examples this originally targeted.
thread_local! {
    static DEFAULT_STREAM: RefCell<Option<StreamHolder>> = const { RefCell::new(None) };
    static CPU_STREAM: RefCell<Option<StreamHolder>> = const { RefCell::new(None) };
}

static INSTALL_ERROR_HANDLER_ONCE: Once = Once::new();

fn ensure_error_handler() {
    INSTALL_ERROR_HANDLER_ONCE.call_once(install_error_handler);
}

/// The current thread's default stream (GPU on Apple Silicon, CPU otherwise).
pub(crate) fn stream() -> crate::sys::mlx_stream {
    ensure_error_handler();
    DEFAULT_STREAM.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(unsafe {
                let mut dev = crate::sys::mlx_device_new();
                crate::sys::mlx_get_default_device(&mut dev);
                let mut s = crate::sys::mlx_stream_new();
                crate::sys::mlx_get_default_stream(&mut s, dev);
                crate::sys::mlx_device_free(dev);
                StreamHolder(s)
            });
        }
        slot.as_ref().unwrap().0
    })
}

/// A stream pinned to the CPU device. `Load` (safetensors I/O) only has a
/// CPU eval kernel, so checkpoint loading must run here rather than on the
/// default (GPU) stream.
pub(crate) fn cpu_stream() -> crate::sys::mlx_stream {
    ensure_error_handler();
    CPU_STREAM.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = Some(unsafe { StreamHolder(crate::sys::mlx_default_cpu_stream_new()) });
        }
        slot.as_ref().unwrap().0
    })
}
