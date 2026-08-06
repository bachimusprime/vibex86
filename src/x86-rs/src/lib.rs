//! x86-rs — a low-level i386 PC core.
//!
//! Provides:
//! - A full real-mode / 16-bit + 32-bit protected-mode decoder (386 subset: all
//!   general-purpose integer instructions, string ops, port I/O, segment/descriptor
//!   machinery, protected-mode interrupts and paging).
//! - Two execution engines over the same decoder and state:
//!   * `Interpreter` — a conventional decode-execute loop.
//!   * `Jit` (feature `jit`, default on) — a Cranelift-based native-code engine
//!     that compiles hot basic blocks.
//!
//! The machine (chipset, PIC/PIT/VGA/IDE/... ) lives in the `vibex86` binary
//! crate; this crate only depends on a `Mem`/device interface.
//!
//! Real firmware (Bochs `BIOS-bochs-latest` and `VGABIOS-lgpl-latest`) is loaded
//! by the machine crate; nothing in this crate is firmware.

pub mod decode;
pub mod mmu;
pub mod sem;

mod cpu;
mod interp;
pub mod mem;

#[cfg(feature = "jit")]
pub mod jit;

pub use cpu::{
    AccessKind, Desc, DescProt, DescType, Error, Mode, Reg, Seg, StepOut, Trip, Tss, X86,
};
pub use interp::Interpreter;
#[cfg(feature = "jit")]
pub use jit::Jit;
pub use mem::{Mem, Ram, VgaMem};

/// Version of the emulated CPU.
pub const CPU_STRING: &str = "i386 (486-class subset)";

/// Convenience test helper: build a core with 16 MB RAM, no devices,
/// firmware-less (test firmware installed by the caller).
pub fn new_core() -> X86 {
    X86::new()
}
