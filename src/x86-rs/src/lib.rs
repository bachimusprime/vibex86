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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{SegVal, flag};

    fn core_with(code: &[u8]) -> X86 {
        let mut cpu = X86::new();
        cpu.eip = 0;
        cpu.seg[Seg::Cs as usize] = SegVal::real(0);
        cpu.seg[Seg::Ss as usize] = SegVal::real(0);
        cpu.mem.phys_write_block(0, code);
        cpu
    }

    fn step_ok(cpu: &mut X86) {
        assert_eq!(cpu.step(), StepOut::Ok);
    }

    #[test]
    fn byte_registers_address_ah_ch_dh_bh_not_esp_family() {
        let mut cpu = core_with(&[
            0xB4, 0x12, // mov ah,12h
            0xB5, 0x34, // mov ch,34h
            0xB6, 0x56, // mov dh,56h
            0xB7, 0x78, // mov bh,78h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0xAAAA_AAAA;
        cpu.gpr[Reg::Ebp as usize] = 0xBBBB_BBBB;
        cpu.gpr[Reg::Esi as usize] = 0xCCCC_CCCC;
        cpu.gpr[Reg::Edi as usize] = 0xDDDD_DDDD;

        for _ in 0..4 {
            step_ok(&mut cpu);
        }

        assert_eq!(cpu.gpr[Reg::Eax as usize], 0x0000_1200);
        assert_eq!(cpu.gpr[Reg::Ecx as usize], 0x0000_3400);
        assert_eq!(cpu.gpr[Reg::Edx as usize], 0x0000_5600);
        assert_eq!(cpu.gpr[Reg::Ebx as usize], 0x0000_7800);
        assert_eq!(cpu.gpr[Reg::Esp as usize], 0xAAAA_AAAA);
        assert_eq!(cpu.gpr[Reg::Ebp as usize], 0xBBBB_BBBB);
        assert_eq!(cpu.gpr[Reg::Esi as usize], 0xCCCC_CCCC);
        assert_eq!(cpu.gpr[Reg::Edi as usize], 0xDDDD_DDDD);
    }

    #[test]
    fn arithmetic_overflow_flags_are_width_correct() {
        let mut cpu = core_with(&[
            0xB0, 0x7F, // mov al,7fh
            0x04, 0x01, // add al,1
            0xFE, 0xC8, // dec al
            0x2C, 0xFF, // sub al,0ffh
        ]);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        assert_eq!(cpu.reg8(0), 0x80);
        assert_ne!(cpu.eflags & flag::OF, 0);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg8(0), 0x7F);
        assert_ne!(cpu.eflags & flag::OF, 0);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg8(0), 0x80);
        assert_ne!(cpu.eflags & flag::OF, 0);
    }

    #[test]
    fn div_and_idiv_use_operand_widths_and_signed_values() {
        let mut cpu = core_with(&[
            0xB8, 0x23, 0x01, // mov ax,0123h
            0xB3, 0x12, // mov bl,12h
            0xF6, 0xF3, // div bl
            0xB8, 0xFC, 0xFF, // mov ax,-4
            0xB3, 0xFE, // mov bl,-2
            0xF6, 0xFB, // idiv bl
        ]);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        step_ok(&mut cpu);
        assert_eq!(cpu.reg8(0), 0x10);
        assert_eq!(cpu.reg8(4), 0x03);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        step_ok(&mut cpu);
        assert_eq!(cpu.reg8(0), 0x02);
        assert_eq!(cpu.reg8(4), 0x00);
    }

    #[test]
    fn pushfd_and_popfd_preserve_32_bit_flags() {
        let mut cpu = core_with(&[
            0x66, 0x9C, // pushfd
            0x66, 0x9D, // popfd
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags = 0x0004_0203;

        step_ok(&mut cpu);
        cpu.eflags = 0x2;
        step_ok(&mut cpu);

        assert_eq!(cpu.eflags, 0x0004_0203);
        assert_eq!(cpu.gpr[Reg::Esp as usize], 0x1000);
    }

    #[test]
    fn register_bit_test_uses_bit_index_modulo_operand_width() {
        let mut cpu = core_with(&[
            0x0F, 0xBA, 0xE0, 0x10, // bt ax,16
        ]);
        cpu.gpr[Reg::Eax as usize] = 1;

        step_ok(&mut cpu);

        assert_ne!(cpu.eflags & flag::CF, 0);
    }

    #[test]
    fn bit_scan_reads_source_and_writes_destination() {
        let mut cpu = core_with(&[
            0x0F, 0xBC, 0xC8, // bsf cx,ax
        ]);
        cpu.gpr[Reg::Eax as usize] = 0x0010;
        cpu.gpr[Reg::Ecx as usize] = 0x7777;

        step_ok(&mut cpu);

        assert_eq!(cpu.reg16(1), 4);
        assert_eq!(cpu.eflags & flag::ZF, 0);
    }

    #[test]
    fn cbw_cwd_obey_operand_size() {
        let mut cpu = core_with(&[
            0x66, 0x98, // cwde
            0x66, 0x99, // cdq
        ]);
        cpu.gpr[Reg::Eax as usize] = 0x0000_8001;

        step_ok(&mut cpu);
        assert_eq!(cpu.gpr[Reg::Eax as usize], 0xFFFF_8001);

        step_ok(&mut cpu);
        assert_eq!(cpu.gpr[Reg::Edx as usize], 0xFFFF_FFFF);
    }

    #[test]
    fn pop_segment_uses_operand_stack_width() {
        let mut cpu = core_with(&[
            0x66, 0x1F, // pop ds with 32-bit operand size
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.mem.phys_write32(0x1000, 0x1234);

        step_ok(&mut cpu);

        assert_eq!(cpu.seg[Seg::Ds as usize].sel, 0x1234);
        assert_eq!(cpu.gpr[Reg::Esp as usize], 0x1004);
    }

    #[test]
    fn lds_reads_32_bit_offset_when_operand_size_is_32() {
        let mut cpu = core_with(&[
            0x66, 0xC5, 0x06, 0x00, 0x01, // lds eax,[0100h]
        ]);
        cpu.mem.phys_write32(0x0100, 0x1234_5678);
        cpu.mem.phys_write16(0x0104, 0x0000);

        step_ok(&mut cpu);

        assert_eq!(cpu.gpr[Reg::Eax as usize], 0x1234_5678);
        assert_eq!(cpu.seg[Seg::Ds as usize].sel, 0);
    }

    #[test]
    fn repne_scasb_stops_on_match_and_updates_count() {
        let mut cpu = core_with(&[
            0xF2, 0xAE, // repne scasb
        ]);
        cpu.set_reg8(0, 5);
        cpu.gpr[Reg::Ecx as usize] = 3;
        cpu.gpr[Reg::Edi as usize] = 0x0200;
        cpu.mem.phys_write8(0x0200, 1);
        cpu.mem.phys_write8(0x0201, 2);
        cpu.mem.phys_write8(0x0202, 5);

        step_ok(&mut cpu);

        assert_eq!(cpu.reg16(Reg::Ecx as i8), 0);
        assert_eq!(cpu.reg16(Reg::Edi as i8), 0x0203);
        assert_ne!(cpu.eflags & flag::ZF, 0);
    }

    #[test]
    fn string_ops_use_16_bit_address_wrap_without_clobbering_high_half() {
        let mut cpu = core_with(&[
            0xAC, // lodsb
        ]);
        cpu.gpr[Reg::Esi as usize] = 0x1234_FFFF;
        cpu.mem.phys_write8(0xFFFF, 0xA5);

        step_ok(&mut cpu);

        assert_eq!(cpu.reg8(0), 0xA5);
        assert_eq!(cpu.gpr[Reg::Esi as usize], 0x1234_0000);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_helper_calls_and_high_byte_registers_are_correct() {
        let mut cpu = core_with(&[
            0xB4, 0x12, // mov ah,12h
            0xB0, 0x7F, // mov al,7fh
            0x04, 0x01, // add al,1
            0xEB, 0x00, // jmp next
        ]);
        cpu.gpr[Reg::Esp as usize] = 0xAAAA_AAAA;
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 4), Ok(4));

        assert_eq!(cpu.gpr[Reg::Eax as usize], 0x0000_1280);
        assert_eq!(cpu.gpr[Reg::Esp as usize], 0xAAAA_AAAA);
        assert_ne!(cpu.eflags & flag::OF, 0);
        assert_eq!(cpu.eip, 8);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_xlat_uses_bx_plus_al_offset() {
        let mut cpu = core_with(&[
            0xD7, // xlat
            0xEB, 0x00, // jmp next
        ]);
        cpu.gpr[Reg::Ebx as usize] = 0x0200;
        cpu.set_reg8(0, 3);
        cpu.mem.phys_write8(0x0203, 0x5A);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 2), Ok(2));

        assert_eq!(cpu.reg8(0), 0x5A);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_run_limit_does_not_execute_past_requested_count() {
        let mut cpu = core_with(&[
            0xB8, 0x34, 0x12, // mov ax,1234h
            0xB9, 0x78, 0x56, // mov cx,5678h
        ]);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 1), Ok(1));

        assert_eq!(cpu.reg16(0), 0x1234);
        assert_eq!(cpu.reg16(1), 0);
    }
}
