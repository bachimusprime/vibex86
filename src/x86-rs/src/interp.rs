//! Interpreter: a conventional fetch-decode-execute loop over the shared
//! `sem` layer. This is also the fallback engine used by the JIT for any
//! instruction or mode it declines to compile.

use crate::cpu::{Error, StepOut, X86, flag};
use crate::decode;
use crate::sem;

pub use crate::sem::dispatch_interrupt;

/// Interpreter engine handle.
#[derive(Debug, Default, Clone, Copy)]
pub struct Interpreter;

/// Execute one instruction (including interrupt delivery between instructions).
pub fn step(cpu: &mut X86) -> StepOut {
    // Pending maskable interrupt: deliver before the next instruction if IF set.
    if let Some(vec) = cpu.pending_irq.take() {
        if cpu.eflags & flag::IF != 0 {
            return dispatch_interrupt(cpu, vec, false, 0);
        }
        // IF cleared: keep it pending and continue.
        cpu.pending_irq = Some(vec);
    }

    cpu.cycles = cpu.cycles.wrapping_add(1);
    let out = sem::step(cpu);
    match out {
        StepOut::Ok => {
            cpu.mem.tick_device(1);
            StepOut::Ok
        }
        other => other,
    }
}

/// Run `n` instructions (or until an `Error`); returns the number executed.
pub fn run(cpu: &mut X86, n: u64) -> Result<u64, Error> {
    let mut count = 0u64;
    while count < n {
        match step(cpu) {
            StepOut::Ok | StepOut::Interrupt => count += 1,
            StepOut::Error(e) => return Err(e),
        }
    }
    Ok(count)
}

/// Decode one instruction without executing (used by debugger/monitor).
pub fn decode(cpu: &X86) -> Result<decode::Decoded, String> {
    decode::fetch(cpu)
}
