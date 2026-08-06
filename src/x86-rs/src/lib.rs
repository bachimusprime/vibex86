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
    use crate::mem::Device;
    use std::any::Any;

    struct FakeIrqDevice {
        irq: Option<u8>,
        ticks: u64,
    }

    impl Device for FakeIrqDevice {
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn io_read(&mut self, _port: u16, _size: u8) -> u32 {
            0xFF
        }

        fn io_write(&mut self, _port: u16, _size: u8, _data: u32) {}

        fn ack_irq(&mut self) -> Option<u8> {
            self.irq.take()
        }

        fn tick(&mut self, cycles: u64) {
            self.ticks += cycles;
        }
    }

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
    fn sign_flag_uses_only_the_operand_sign_bit() {
        let mut cpu = core_with(&[
            0xB8, 0x06, 0x00, // mov ax,6
            0x83, 0xF8, 0x00, // cmp ax,0
            0x7C, 0x03, // jl skipped
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        assert_eq!(cpu.eflags & flag::SF, 0);

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 8);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg16(0), 0x1234);
    }

    #[test]
    fn one_byte_xchg_opcodes_address_the_right_register() {
        let mut cpu = core_with(&[
            0xB8, 0x11, 0x11, // mov ax,1111h
            0xB9, 0x22, 0x22, // mov cx,2222h
            0xBA, 0x33, 0x33, // mov dx,3333h
            0x92, // xchg ax,dx
        ]);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        step_ok(&mut cpu);
        step_ok(&mut cpu);

        assert_eq!(cpu.reg16(0), 0x3333);
        assert_eq!(cpu.reg16(1), 0x2222);
        assert_eq!(cpu.reg16(2), 0x1111);
    }

    #[test]
    fn bound_checks_signed_memory_limits() {
        let mut cpu = core_with(&[
            0xB8, 0x05, 0x00, // mov ax,5
            0x62, 0x06, 0x00, 0x01, // bound ax,[0100h]
            0xB8, 0x20, 0x00, // mov ax,20h
            0x62, 0x06, 0x00, 0x01, // bound ax,[0100h]
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.mem.phys_write16(0x0100, 0x0001);
        cpu.mem.phys_write16(0x0102, 0x0008);
        cpu.mem.phys_write16(5 * 4, 0x0200);
        cpu.mem.phys_write16(5 * 4 + 2, 0x0000);

        step_ok(&mut cpu);
        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 7);
        step_ok(&mut cpu);

        assert_eq!(cpu.step(), StepOut::Interrupt);
        assert_eq!(cpu.eip, 0x0200);
        assert_eq!(cpu.mem.phys_read16(0x0FFA), 0x000A);
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
    fn far_indirect_jump_reads_memory_pointer() {
        let mut cpu = core_with(&[
            0xFF, 0x2E, 0x00, 0x01, // ljmp far [0100h]
        ]);
        cpu.mem.phys_write16(0x0100, 0x0020);
        cpu.mem.phys_write16(0x0102, 0x1234);

        step_ok(&mut cpu);

        assert_eq!(cpu.seg[Seg::Cs as usize].sel, 0x1234);
        assert_eq!(cpu.eip, 0x0020);
    }

    #[test]
    fn near_indirect_jump_reads_memory_pointer() {
        let mut cpu = core_with(&[
            0xBB, 0x02, 0x00, // mov bx,2
            0xFF, 0xA7, 0x00, 0x01, // jmp word [bx+0100h]
        ]);
        cpu.mem.phys_write16(0x0102, 0x0040);

        step_ok(&mut cpu);
        step_ok(&mut cpu);

        assert_eq!(cpu.eip, 0x0040);
    }

    #[test]
    fn protected_far_jump_loads_descriptor_base_and_limit() {
        let mut cpu = core_with(&[
            0x66, 0xEA, 0x78, 0x56, 0x34, 0x12, 0x08, 0x00, // jmp 8:12345678h
        ]);
        cpu.cr[0] = 1;
        cpu.gdtr.base = 0x0200;
        cpu.gdtr.limit = 0x0010;
        cpu.mem.phys_write32(0x0208, 0x0000_FFFF);
        cpu.mem.phys_write32(0x020C, 0x00CF_9A12);

        step_ok(&mut cpu);

        assert_eq!(cpu.seg[Seg::Cs as usize].sel, 0x0008);
        assert_eq!(cpu.seg[Seg::Cs as usize].desc.base, 0x0012_0000);
        assert_eq!(cpu.seg[Seg::Cs as usize].desc.limit, 0x000F_FFFF);
        assert_eq!(cpu.seg[Seg::Cs as usize].desc.eff_limit(), 0xFFFF_FFFF);
        assert_eq!(cpu.eip, 0x1234_5678);
    }

    #[test]
    fn protected_data_segments_accept_null_selector() {
        let mut cpu = core_with(&[
            0x8E, 0xD8, // mov ds,ax
        ]);
        cpu.cr[0] = 1;
        cpu.gpr[Reg::Eax as usize] = 0;

        step_ok(&mut cpu);

        assert_eq!(cpu.seg[Seg::Ds as usize].sel, 0);
        assert!(!cpu.seg[Seg::Ds as usize].desc.p);
    }

    #[test]
    fn test_accumulator_immediate_opcodes_decode_and_set_flags() {
        let mut cpu = core_with(&[
            0xA8, 0x10, // test al,10h
            0xA9, 0x00, 0x80, // test ax,8000h
        ]);
        cpu.set_reg8(0, 0x10);

        step_ok(&mut cpu);
        assert_eq!(cpu.eflags & flag::ZF, 0);

        cpu.set_reg16(0, 0x7FFF);
        step_ok(&mut cpu);
        assert_ne!(cpu.eflags & flag::ZF, 0);
    }

    #[test]
    fn software_interrupt_returns_to_following_instruction() {
        let mut cpu = core_with(&[
            0xCD, 0x10, // int 10h
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags |= flag::IF;
        cpu.mem.phys_write16(0x10 * 4, 0x0200);
        cpu.mem.phys_write16(0x10 * 4 + 2, 0x0000);
        cpu.mem.phys_write8(0x0200, 0xCF); // iret

        assert_eq!(cpu.step(), StepOut::Interrupt);
        assert_eq!(cpu.eip, 0x0200);
        assert_eq!(cpu.mem.phys_read16(0x0FFA), 0x0002);

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 0x0002);
        assert_ne!(cpu.eflags & flag::IF, 0);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg16(0), 0x1234);
    }

    #[test]
    fn interpreter_acknowledges_device_irq_when_if_is_set() {
        let mut cpu = core_with(&[
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags |= flag::IF;
        cpu.mem.phys_write16(0x08 * 4, 0x0200);
        cpu.mem.phys_write16(0x08 * 4 + 2, 0x0000);
        cpu.mem.phys_write8(0x0200, 0xCF); // iret
        cpu.mem.install_device(Box::new(FakeIrqDevice {
            irq: Some(0x08),
            ticks: 0,
        }));

        assert_eq!(cpu.step(), StepOut::Interrupt);
        assert_eq!(cpu.eip, 0x0200);
        assert_eq!(cpu.mem.phys_read16(0x0FFA), 0x0000);

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 0x0000);
        assert_ne!(cpu.eflags & flag::IF, 0);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg16(0), 0x1234);
    }

    #[test]
    fn nested_real_mode_interrupt_restores_outer_flags_after_cli() {
        let mut cpu = core_with(&[
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags |= flag::IF;
        cpu.mem.phys_write16(0x08 * 4, 0x0200);
        cpu.mem.phys_write16(0x08 * 4 + 2, 0x0000);
        cpu.mem.phys_write16(0x1C * 4, 0x0300);
        cpu.mem.phys_write16(0x1C * 4 + 2, 0x0000);
        cpu.mem.phys_write_block(
            0x0200,
            &[
                0xFB, // sti
                0xCD, 0x1C, // int 1ch
                0xFA, // cli
                0xCF, // iret
            ],
        );
        cpu.mem.phys_write8(0x0300, 0xCF); // iret
        cpu.mem.install_device(Box::new(FakeIrqDevice {
            irq: Some(0x08),
            ticks: 0,
        }));

        assert_eq!(cpu.step(), StepOut::Interrupt);
        step_ok(&mut cpu);
        assert_eq!(cpu.step(), StepOut::Interrupt);
        step_ok(&mut cpu);
        step_ok(&mut cpu);
        step_ok(&mut cpu);

        assert_eq!(cpu.eip, 0x0000);
        assert_ne!(cpu.eflags & flag::IF, 0);
    }

    #[test]
    fn hlt_waits_at_next_eip_until_irq_arrives() {
        let mut cpu = core_with(&[
            0xF4, // hlt
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags |= flag::IF;
        cpu.mem.phys_write16(0x08 * 4, 0x0200);
        cpu.mem.phys_write16(0x08 * 4 + 2, 0x0000);
        cpu.mem.phys_write8(0x0200, 0xCF); // iret

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 1);
        assert!(cpu.halted);

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 1);
        assert!(cpu.halted);

        cpu.pending_irq = Some(0x08);
        assert_eq!(cpu.step(), StepOut::Interrupt);
        assert!(!cpu.halted);
        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 1);

        step_ok(&mut cpu);
        assert_eq!(cpu.reg16(0), 0x1234);
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
    fn jit_memory_store_does_not_fall_back_as_fault() {
        let mut cpu = core_with(&[
            0xC7, 0x06, 0x00, 0x02, 0x34, 0x12, // mov word [0200h],1234h
            0xEB, 0x00, // jmp next
        ]);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 2), Ok(2));

        assert_eq!(cpu.eip, 8);
        assert_eq!(cpu.mem.phys_read16(0x0200), 0x1234);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_rep_stos_advances_to_next_eip() {
        let mut cpu = core_with(&[
            0xF3, 0xAB, // rep stosw
            0xEB, 0x00, // jmp next
        ]);
        cpu.gpr[Reg::Ecx as usize] = 2;
        cpu.gpr[Reg::Edi as usize] = 0x0200;
        cpu.set_reg16(Reg::Eax as i8, 0xA55A);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 2), Ok(2));

        assert_eq!(cpu.eip, 4);
        assert_eq!(cpu.reg16(Reg::Ecx as i8), 0);
        assert_eq!(cpu.reg16(Reg::Edi as i8), 0x0204);
        assert_eq!(cpu.mem.phys_read16(0x0200), 0xA55A);
        assert_eq!(cpu.mem.phys_read16(0x0202), 0xA55A);
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

    #[cfg(feature = "jit")]
    #[test]
    fn jit_test_accumulator_immediate_runs_natively() {
        let mut cpu = core_with(&[
            0xA8, 0x10, // test al,10h
            0xEB, 0x00, // jmp next
        ]);
        cpu.set_reg8(0, 0x10);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 2), Ok(2));

        assert_eq!(cpu.eflags & flag::ZF, 0);
        assert_eq!(cpu.eip, 4);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_near_indirect_jump_reads_memory_pointer() {
        let mut cpu = core_with(&[
            0xBB, 0x02, 0x00, // mov bx,2
            0xFF, 0xA7, 0x00, 0x01, // jmp word [bx+0100h]
        ]);
        cpu.mem.phys_write16(0x0102, 0x0040);
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 2), Ok(2));

        assert_eq!(cpu.eip, 0x0040);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_near_call_and_ret_keep_stack_pointer() {
        let mut cpu = core_with(&[
            0xE8, 0x01, 0x00, // call 0004h
            0x90, // nop
            0xC3, // ret
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 3), Ok(3));

        assert_eq!(cpu.eip, 0x0004);
        assert_eq!(cpu.reg16(Reg::Esp as i8), 0x1000);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn jit_acknowledges_device_irq_when_if_is_set() {
        let mut cpu = core_with(&[
            0xB8, 0x34, 0x12, // mov ax,1234h
        ]);
        cpu.gpr[Reg::Esp as usize] = 0x1000;
        cpu.eflags |= flag::IF;
        cpu.mem.phys_write16(0x08 * 4, 0x0200);
        cpu.mem.phys_write16(0x08 * 4 + 2, 0x0000);
        cpu.mem.phys_write8(0x0200, 0xCF); // iret
        cpu.mem.install_device(Box::new(FakeIrqDevice {
            irq: Some(0x08),
            ticks: 0,
        }));
        let mut jit = Jit::new().unwrap();

        assert_eq!(jit.run(&mut cpu, 1), Ok(1));
        assert_eq!(cpu.eip, 0x0200);

        step_ok(&mut cpu);
        assert_eq!(cpu.eip, 0x0000);
        assert_ne!(cpu.eflags & flag::IF, 0);

        assert_eq!(jit.run(&mut cpu, 1), Ok(1));
        assert_eq!(cpu.reg16(0), 0x1234);
    }
}
