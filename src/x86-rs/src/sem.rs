//! Instruction semantics: the reference implementation shared by the
//! interpreter and (as the slow path) the JIT.
//!
//! Everything here is pure CPU + `Mem` state; no firmware or device logic.

use crate::cpu::{
    AccessKind, Desc, Error, Mode, Reg, Seg, StepOut, X86,
    flag::{self, AF, CF, OF, PF, SF, ZF},
};
use crate::decode::{AluOp, BitOp, Bits, Cond, Decoded, MemRef, Op, Opnd, Rep, ShiftBy, ShiftKind};
use crate::mmu;

// ---------------------------------------------------------------------------
// Flags helpers
// ---------------------------------------------------------------------------

const MASK8: u32 = 0xFF;
const MASK16: u32 = 0xFFFF;

#[inline]
fn mask_for(sz: Bits) -> u32 {
    match sz {
        Bits::B8 => MASK8,
        Bits::B16 => MASK16,
        Bits::B32 => u32::MAX,
    }
}

#[inline]
fn sign_bit_for(sz: Bits) -> u32 {
    match sz {
        Bits::B8 => 0x80,
        Bits::B16 => 0x8000,
        Bits::B32 => 0x8000_0000,
    }
}

#[inline]
pub fn set_flags_zsp(cpu: &mut X86, val: u32, mask: u32) {
    let z = val & mask == 0;
    let s = val & ((mask >> 1).wrapping_add(1)) != 0;
    let p = (val as u8).count_ones() % 2 == 0;
    if z {
        cpu.eflags |= ZF;
    } else {
        cpu.eflags &= !ZF;
    }
    if s {
        cpu.eflags |= SF;
    } else {
        cpu.eflags &= !SF;
    }
    if p {
        cpu.eflags |= PF;
    } else {
        cpu.eflags &= !PF;
    }
}

#[inline]
pub fn set_cf(cpu: &mut X86, c: bool) {
    if c {
        cpu.eflags |= CF;
    } else {
        cpu.eflags &= !CF;
    }
}

#[inline]
pub fn set_of(cpu: &mut X86, o: bool) {
    if o {
        cpu.eflags |= OF;
    } else {
        cpu.eflags &= !OF;
    }
}

#[inline]
pub fn set_af(cpu: &mut X86, a: bool) {
    if a {
        cpu.eflags |= AF;
    } else {
        cpu.eflags &= !AF;
    }
}

/// Compute AF for a binary add/sub (carry out of bit 3).
#[inline]
fn af_add(a: u32, b: u32, mask: u32, r: u32) -> bool {
    ((a ^ b ^ r) & (mask & 0x10)) != 0
}

#[inline]
pub fn cond_true(cpu: &X86, c: Cond) -> bool {
    let f = cpu.eflags;
    let (cf, pf, zf, sf, of, _af) = (
        f & CF != 0,
        f & PF != 0,
        f & ZF != 0,
        f & SF != 0,
        f & OF != 0,
        f & AF != 0,
    );
    match c {
        Cond::O => of,
        Cond::No => !of,
        Cond::B => cf,
        Cond::Ae => !cf,
        Cond::E => zf,
        Cond::Ne => !zf,
        Cond::Be => cf || zf,
        Cond::A => !cf && !zf,
        Cond::S => sf,
        Cond::Ns => !sf,
        Cond::P => pf,
        Cond::Np => !pf,
        Cond::L => sf != of,
        Cond::Ge => sf == of,
        Cond::Le => zf || (sf != of),
        Cond::G => !zf && (sf == of),
    }
}

// ---------------------------------------------------------------------------
// Operand access
// ---------------------------------------------------------------------------

/// Returns (value, is_mem) for an operand given the current CPU state.
fn read_opnd(cpu: &mut X86, o: &Opnd, kind: AccessKind) -> Result<u32, Error> {
    let v = match o {
        Opnd::Reg(r, bits) => read_gpr(cpu, *r, *bits),
        Opnd::Mem(m, bits) => read_mem(cpu, m, *bits, kind)?,
        Opnd::Imm(v) | Opnd::ImmSext(v) => *v,
        Opnd::Acc(bits) => read_gpr(cpu, 0, *bits),
        Opnd::Dx => cpu.reg16(2) as u32,
        Opnd::Port(p) => *p as u32,
        Opnd::Cl => cpu.reg8(1) as u32,
        Opnd::Sreg(s) => cpu.seg[*s as usize].sel as u32,
        Opnd::Rel { disp } => (*disp as i64 as u64) as u32,
        Opnd::None | Opnd::FarPtr { .. } => 0,
    };
    Ok(v)
}

#[inline]
fn read_gpr(cpu: &X86, r: u8, bits: Bits) -> u32 {
    match bits {
        Bits::B8 => cpu.reg8(r as i8) as u32,
        Bits::B16 => cpu.reg16(r as i8) as u32,
        Bits::B32 => cpu.reg32(r as i8),
    }
}

fn write_gpr(cpu: &mut X86, r: u8, bits: Bits, v: u32) {
    match bits {
        Bits::B8 => cpu.set_reg8(r as i8, v as u8),
        Bits::B16 => cpu.set_reg16(r as i8, v as u16),
        Bits::B32 => cpu.set_reg32(r as i8, v),
    }
}

/// Effective address of a memory operand (offset within segment).
#[inline]
pub fn eff(cpu: &X86, m: &MemRef) -> u32 {
    mmu::eff_addr(cpu, m.base, m.index, m.scale, m.disp, m.a16)
}

fn read_mem(cpu: &mut X86, m: &MemRef, bits: Bits, kind: AccessKind) -> Result<u32, Error> {
    let off = eff(cpu, m);
    Ok(match bits {
        Bits::B8 => mmu::read8(cpu, m.seg, off, kind)? as u32,
        Bits::B16 => mmu::read16(cpu, m.seg, off, kind)? as u32,
        Bits::B32 => mmu::read32(cpu, m.seg, off, kind)?,
    })
}

fn write_mem(cpu: &mut X86, m: &MemRef, bits: Bits, v: u32, kind: AccessKind) -> Result<(), Error> {
    let off = eff(cpu, m);
    match bits {
        Bits::B8 => mmu::write8(cpu, m.seg, off, v as u8, kind),
        Bits::B16 => mmu::write16(cpu, m.seg, off, v as u16, kind),
        Bits::B32 => mmu::write32(cpu, m.seg, off, v, kind),
    }
}

/// Write an operand back. Returns false if the operand is read-only (imm).
fn write_opnd(cpu: &mut X86, o: &Opnd, v: u32, kind: AccessKind) -> Result<bool, Error> {
    match o {
        Opnd::Reg(r, bits) => {
            write_gpr(cpu, *r, *bits, v);
            Ok(true)
        }
        Opnd::Acc(bits) => {
            write_gpr(cpu, 0, *bits, v);
            Ok(true)
        }
        Opnd::Mem(m, bits) => {
            write_mem(cpu, m, *bits, v, kind)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn read_far_target(cpu: &mut X86, o: &Opnd) -> Result<(u16, u32), Error> {
    match *o {
        Opnd::FarPtr { sel, off } => Ok((sel, off)),
        Opnd::Mem(m, bits) => {
            let off = eff(cpu, &m);
            let target = match bits {
                Bits::B16 => mmu::read16(cpu, m.seg, off, AccessKind::Read)? as u32,
                Bits::B32 => mmu::read32(cpu, m.seg, off, AccessKind::Read)?,
                Bits::B8 => return Err(Error::Internal("far pointer cannot be byte-sized".into())),
            };
            let sel = mmu::read16(cpu, m.seg, off.wrapping_add(bits.bytes()), AccessKind::Read)?;
            Ok((sel, target))
        }
        _ => Err(Error::Internal("far target operand".into())),
    }
}

fn read_near_target(cpu: &mut X86, o: &Opnd, next: u32) -> Result<u32, Error> {
    match *o {
        Opnd::Rel { disp } => Ok((next as i64 + disp as i64) as u32),
        Opnd::Reg(_, _) | Opnd::Mem(_, _) => read_opnd(cpu, o, AccessKind::Read),
        _ => Ok(next),
    }
}

fn far_jump(cpu: &mut X86, sel: u16, off: u32) -> Result<(), Error> {
    load_segment(cpu, Seg::Cs, sel)?;
    cpu.eip = off;
    Ok(())
}

// ---------------------------------------------------------------------------
// Stack helpers
// ---------------------------------------------------------------------------

/// ESP/EIP addressing-mode dependent help: returns whether the stack is 32-bit.
fn stack_size(d: &Decoded) -> u32 {
    // Default stack size = operand size unless 0x67 masks it on SS (rare).
    if d.o16 { 2 } else { 4 }
}

fn push(cpu: &mut X86, size: u32, val: u32) -> Result<(), Error> {
    let esp = cpu.gpr[Reg::Esp as usize];
    let new_esp = if size == 2 {
        (esp & 0xFFFF_0000) | ((esp as u16).wrapping_sub(size as u16) as u32)
    } else {
        esp.wrapping_sub(size)
    };
    mmu::stack_write(cpu, new_esp, size, val, AccessKind::Write)?;
    cpu.gpr[Reg::Esp as usize] = new_esp;
    Ok(())
}

fn pop(cpu: &mut X86, size: u32) -> Result<u32, Error> {
    let esp = cpu.gpr[Reg::Esp as usize];
    let v = mmu::stack_read(cpu, esp, size, AccessKind::Read)?;
    cpu.gpr[Reg::Esp as usize] = if size == 2 {
        (esp & 0xFFFF_0000) | ((esp as u16).wrapping_add(size as u16) as u32)
    } else {
        esp.wrapping_add(size)
    };
    Ok(v)
}

#[inline]
fn update_index(cpu: &mut X86, reg: Reg, step: u32, a16: bool) {
    let idx = reg as usize;
    if a16 {
        let v = (cpu.gpr[idx] as u16).wrapping_add(step as u16);
        cpu.gpr[idx] = (cpu.gpr[idx] & 0xFFFF_0000) | v as u32;
    } else {
        cpu.gpr[idx] = cpu.gpr[idx].wrapping_add(step);
    }
}

#[inline]
fn count_reg(cpu: &X86, a16: bool) -> u32 {
    if a16 {
        cpu.gpr[Reg::Ecx as usize] & 0xFFFF
    } else {
        cpu.gpr[Reg::Ecx as usize]
    }
}

#[inline]
fn set_count(cpu: &mut X86, v: u32, a16: bool) {
    if a16 {
        cpu.set_reg16(Reg::Ecx as i8, v as u16);
    } else {
        cpu.gpr[Reg::Ecx as usize] = v;
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

pub fn alu(cpu: &mut X86, op: AluOp, a: u32, b: u32, sz: Bits) -> u32 {
    let mask = mask_for(sz);
    let sign = sign_bit_for(sz);
    let a = a & mask;
    let b = b & mask;
    match op {
        AluOp::Add | AluOp::Adc => {
            let c = if op == AluOp::Adc && cpu.eflags & CF != 0 {
                1
            } else {
                0
            };
            let r = a.wrapping_add(b).wrapping_add(c);
            let carry = (a as u64) + (b as u64) + (c as u64) > mask as u64;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, carry);
            set_of(cpu, ((!(a ^ b) & (a ^ r)) & sign) != 0);
            set_af(cpu, af_add(a, b, mask, r));
            r & mask
        }
        AluOp::Or | AluOp::And | AluOp::Xor => {
            let r = match op {
                AluOp::Or => a | b,
                AluOp::And => a & b,
                _ => a ^ b,
            } & mask;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, false);
            set_of(cpu, false);
            set_af(cpu, false);
            r
        }
        AluOp::Sub | AluOp::Sbb | AluOp::Cmp => {
            let b_ = if op == AluOp::Sbb && cpu.eflags & CF != 0 {
                b.wrapping_add(1)
            } else {
                b
            };
            let b_ = b_ & mask;
            let r = a.wrapping_sub(b_);
            let borrow = a < b_;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, borrow);
            set_of(cpu, ((a ^ b_) & (a ^ r) & sign) != 0);
            set_af(cpu, af_add(a, b_, mask, r));
            r & mask
        }
    }
}

pub fn inc_dec(cpu: &mut X86, v: u32, sz: Bits, inc: bool) -> u32 {
    let mask = mask_for(sz);
    let sign = sign_bit_for(sz);
    let v = v & mask;
    let r = if inc {
        v.wrapping_add(1) & mask
    } else {
        v.wrapping_sub(1) & mask
    };
    set_flags_zsp(cpu, r, mask);
    set_of(
        cpu,
        if inc {
            v == sign.wrapping_sub(1)
        } else {
            v == sign
        },
    );
    set_af(cpu, (v & 0xF) == (if inc { 0xF } else { 0 }));
    r
}

pub fn shift(cpu: &mut X86, kind: ShiftKind, v: u32, count: u32, sz: Bits) -> u32 {
    let bits = match sz {
        Bits::B8 => 8,
        Bits::B16 => 16,
        _ => 32,
    };
    let mask = mask_for(sz);
    let sign = sign_bit_for(sz);
    let v = v & mask;
    let cnt = count & 0x1F;
    if cnt == 0 {
        return v;
    }
    match kind {
        ShiftKind::Rol => {
            let cnt = cnt % bits;
            if cnt == 0 {
                return v;
            }
            let r = ((v << cnt) | (v >> (bits - cnt))) & mask;
            set_cf(cpu, r & 1 != 0);
            if cnt == 1 {
                set_of(cpu, ((r ^ (r >> 1)) & sign) != 0);
            }
            r
        }
        ShiftKind::Ror => {
            let cnt = cnt % bits;
            if cnt == 0 {
                return v;
            }
            let r = ((v >> cnt) | (v << (bits - cnt))) & mask;
            set_cf(cpu, r & sign != 0);
            if cnt == 1 {
                set_of(cpu, ((r ^ (r << 1)) & sign) != 0);
            }
            r
        }
        ShiftKind::Rcl => {
            let cnt = if bits == 32 { cnt } else { cnt % (bits + 1) };
            if cnt == 0 {
                return v;
            }
            let ring_bits = bits + 1;
            let ring_mask = (1u64 << ring_bits) - 1;
            let ring = ((v as u64) | (u64::from(cpu.eflags & CF != 0) << bits)) & ring_mask;
            let rr = ((ring << cnt) | (ring >> (ring_bits - cnt))) & ring_mask;
            let r = (rr as u32) & mask;
            set_cf(cpu, ((rr >> bits) & 1) != 0);
            if cnt == 1 {
                set_of(
                    cpu,
                    ((r ^ ((cpu.eflags & CF != 0) as u32 * sign)) & sign) != 0,
                );
            }
            r
        }
        ShiftKind::Rcr => {
            let cnt = if bits == 32 { cnt } else { cnt % (bits + 1) };
            if cnt == 0 {
                return v;
            }
            let ring_bits = bits + 1;
            let ring_mask = (1u64 << ring_bits) - 1;
            let ring = ((v as u64) | (u64::from(cpu.eflags & CF != 0) << bits)) & ring_mask;
            let rr = ((ring >> cnt) | (ring << (ring_bits - cnt))) & ring_mask;
            let r = (rr as u32) & mask;
            set_cf(cpu, ((rr >> bits) & 1) != 0);
            if cnt == 1 {
                set_of(cpu, ((r ^ (r << 1)) & sign) != 0);
            }
            r
        }
        ShiftKind::Shl => {
            if cnt > bits {
                set_cf(cpu, false);
                set_of(cpu, false);
                set_flags_zsp(cpu, 0, mask);
                return 0;
            }
            let r = (v << cnt) & mask;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, ((v >> (bits - cnt)) & 1) != 0);
            if cnt == 1 {
                set_of(cpu, ((r ^ v) & sign) != 0);
            }
            r
        }
        ShiftKind::Shr => {
            if cnt > bits {
                set_cf(cpu, false);
                set_of(cpu, false);
                set_flags_zsp(cpu, 0, mask);
                return 0;
            }
            let r = v >> cnt;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, ((v >> (cnt - 1)) & 1) != 0);
            if cnt == 1 {
                set_of(cpu, v & sign != 0);
            }
            r
        }
        ShiftKind::Sar => {
            let cnt = cnt.min(bits);
            let extended = if v & sign != 0 { v | !mask } else { v };
            let r = ((extended as i32) >> cnt) as u32 & mask;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, ((v >> (cnt - 1)) & 1) != 0);
            if cnt == 1 {
                set_of(cpu, false);
            }
            r
        }
    }
}

fn mul(cpu: &mut X86, v: u32, sz: Bits, signed: bool) -> Result<(), Error> {
    let mask = match sz {
        Bits::B8 => MASK8,
        Bits::B16 => MASK16,
        _ => u32::MAX,
    };
    let a = read_gpr(cpu, 0, sz) & mask;
    match sz {
        Bits::B8 => {
            let (hi, lo) = if signed {
                let (a, b) = (a as i8 as i16, v as i8 as i16);
                let r = (a as i16).wrapping_mul(b);
                ((r >> 8) as u8 as u32, (r & 0xFF) as u32)
            } else {
                let r = (a as u16).wrapping_mul(v as u16);
                ((r >> 8) as u32, (r & 0xFF) as u32)
            };
            cpu.set_reg8(0, lo as u8);
            cpu.set_reg8(4, hi as u8); // AH
            let overflow = if signed {
                (hi as i8) != ((lo as i8) >> 7)
            } else {
                hi != 0
            };
            set_cf(cpu, overflow);
            set_of(cpu, overflow);
            Ok(())
        }
        Bits::B16 => {
            let (hi, lo) = if signed {
                let (a, b) = (a as i16 as i32, v as i16 as i32);
                let r = a.wrapping_mul(b);
                ((r >> 16) as u16 as u32, (r & 0xFFFF) as u32)
            } else {
                let r = (a as u32).wrapping_mul(v as u32);
                (r >> 16, r & 0xFFFF)
            };
            cpu.set_reg16(0, lo as u16);
            cpu.set_reg16(2, hi as u16); // DX
            let overflow = if signed {
                (hi as i16 as i32) != (((lo as i16) >> 15) as i32)
            } else {
                hi != 0
            };
            set_cf(cpu, overflow);
            set_of(cpu, overflow);
            Ok(())
        }
        Bits::B32 => {
            if signed {
                let (a, b) = (a as i32 as i64, v as i32 as i64);
                let r = a.wrapping_mul(b);
                let hi = (r >> 32) as u32;
                let lo = r as u32;
                cpu.set_reg32(0, lo);
                cpu.set_reg32(2, hi);
                let overflow = r != lo as i32 as i64;
                set_cf(cpu, overflow);
                set_of(cpu, overflow);
            } else {
                let r = (a as u64).wrapping_mul(v as u64);
                cpu.set_reg32(0, r as u32);
                cpu.set_reg32(2, (r >> 32) as u32);
                let overflow = r > u32::MAX as u64;
                set_cf(cpu, overflow);
                set_of(cpu, overflow);
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Interrupt / exception dispatch (shared with interpreter)
// ---------------------------------------------------------------------------

pub fn real_mode_int(cpu: &mut X86, vector: u8) -> Result<(), Error> {
    // IVT: idtr.base = 0 always in real mode.
    let addr = (vector as u32) * 4;
    let off = cpu.mem.phys_read16(addr);
    let seg = cpu.mem.phys_read16(addr + 2);
    let flags = cpu.eflags & 0xFFFF;
    let cs = cpu.seg[Seg::Cs as usize].sel;
    let ip = cpu.eip as u16;

    push(cpu, 2, flags)?;
    push(cpu, 2, cs as u32)?;
    push(cpu, 2, ip as u32)?;

    cpu.eflags &= !(0x100 | 0x200 | 0x400); // IF, TF, AC
    cpu.seg[Seg::Cs as usize] = crate::cpu::SegVal::real(seg);
    cpu.eip = off as u32;
    Ok(())
}

pub fn dispatch_interrupt(cpu: &mut X86, vector: u8, has_error: bool, code: u32) -> StepOut {
    if cpu.cr[0] & 1 == 0 {
        // Real mode: honor IF for maskable interrupts.
        if has_error {
            // Software ints don't carry error codes; treat as invalid.
            return real_mode_int(cpu, 13)
                .map_or(StepOut::Error(Error::Internal("int13".into())), |_| {
                    StepOut::Interrupt
                });
        }
        let _ = code;
        return match real_mode_int(cpu, vector) {
            Ok(()) => StepOut::Interrupt,
            Err(e) => StepOut::Error(e),
        };
    }

    // Protected mode: IDT gate.
    let idt = cpu.idtr;
    if vector as u32 * 8 + 7 > idt.limit as u32 + 8 {
        return StepOut::Error(Error::Internal(format!(
            "IDT limit ({}) exceeded for vector {vector}",
            idt.limit
        )));
    }
    let base = idt.base.wrapping_add(vector as u32 * 8);
    let gate_lo = mmu::read16(cpu, Seg::Es, base, AccessKind::Read).unwrap_or(0); // physical below
    let gate_hi = mmu::read16(cpu, Seg::Es, base + 4, AccessKind::Read).unwrap_or(0);
    let dpl = ((gate_hi >> 13) & 3) as u8;
    let offset = (gate_lo as u32) | (((gate_hi & 0xFFFF) as u32) << 16);
    let selector = mmu::read16(cpu, Seg::Es, base + 2, AccessKind::Read).unwrap_or(0);
    let _ = dpl;
    if cpu.cpl() != 0 && dpl < cpu.cpl() {
        return StepOut::Error(Error::Internal("gate DPL < CPL".into()));
    }
    // Push flags/CS/EIP, then jump.
    let old_flags = cpu.eflags;
    let old_cs = cpu.seg[Seg::Cs as usize].sel;
    let old_ip = cpu.eip;
    let size = if cpu.seg[Seg::Cs as usize].desc.db {
        4
    } else {
        2
    };
    if let Err(e) = push(cpu, 4, old_flags) {
        return StepOut::Error(e);
    }
    if let Err(e) = push(cpu, 4, old_cs as u32) {
        return StepOut::Error(e);
    }
    if let Err(e) = push(cpu, 4, old_ip) {
        return StepOut::Error(e);
    }
    if let Err(e) = push(cpu, 4, code) {
        return StepOut::Error(e);
    }
    cpu.eflags &= !flag::IF;
    cpu.seg[Seg::Cs as usize] = crate::cpu::SegVal::real(selector);
    cpu.eip = offset;
    let _ = size;
    let _ = has_error;
    StepOut::Interrupt
}

/// Deliver one maskable hardware interrupt before the next instruction.
///
/// External devices raise IRQs through the machine PIC, exposed as an INTA-style
/// `ack_irq` callback. Only acknowledge the PIC when IF is set; otherwise the
/// request must remain pending in device state.
pub fn deliver_maskable_interrupt(cpu: &mut X86) -> Option<StepOut> {
    if cpu.eflags & flag::IF == 0 {
        return None;
    }

    if let Some(vec) = cpu.pending_irq.take().or_else(|| cpu.mem.ack_irq()) {
        cpu.halted = false;
        return Some(dispatch_interrupt(cpu, vec, false, 0));
    }

    if cpu.halted {
        cpu.mem.tick_device(1);
        return Some(StepOut::Ok);
    }

    None
}

// ---------------------------------------------------------------------------
// The main execution driver (one instruction).
// ---------------------------------------------------------------------------

pub fn step(cpu: &mut X86) -> StepOut {
    let decoded = match crate::decode::fetch(cpu) {
        Ok(d) => d,
        Err(e) => return StepOut::Error(Error::Unsupported(e)),
    };
    exec(cpu, &decoded)
}

pub fn exec(cpu: &mut X86, d: &Decoded) -> StepOut {
    let sz = d.size();
    let size = sz.bytes();
    let next = cpu.eip.wrapping_add(d.len as u32);

    // REP prefix is handled by the string instructions.
    let out = match d.op {
        // ---- ALU ----
        Op::Alu(op) => {
            let a = match read_opnd(cpu, &d.ops[0], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let b = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            // `alu` computes flags and result together; call it exactly once so
            // carry-dependent ops (ADC/SBB) see the *input* flags.
            let r = alu(cpu, op, a, b, bits);
            if op != AluOp::Cmp {
                if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                    return StepOut::Error(e);
                }
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Inc | Op::Dec => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let r = inc_dec(cpu, v, bits, d.op == Op::Inc);
            if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Neg => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let _ = alu(cpu, AluOp::Sub, 0, v, bits);
            let r = (0u32.wrapping_sub(v))
                & match bits {
                    Bits::B8 => MASK8,
                    Bits::B16 => MASK16,
                    Bits::B32 => u32::MAX,
                };
            if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Not => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let mask = match bits {
                Bits::B8 => MASK8,
                Bits::B16 => MASK16,
                Bits::B32 => u32::MAX,
            };
            let r = !v & mask;
            if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Mul | Op::Imul => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            if let Err(e) = mul(cpu, v, bits, d.op == Op::Imul) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Div | Op::Idiv => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let mask = match bits {
                Bits::B8 => MASK8,
                Bits::B16 => MASK16,
                Bits::B32 => u32::MAX,
            };
            let (num_lo, num_hi) = match bits {
                Bits::B8 => (cpu.reg8(0) as u32, cpu.reg8(4) as u32),
                Bits::B16 => (cpu.reg16(0) as u32, cpu.reg16(2) as u32),
                Bits::B32 => (cpu.reg32(0), cpu.reg32(2)),
            };
            if v == 0 {
                return dispatch_interrupt(cpu, 0, false, 0);
            }
            if d.op == Op::Div {
                let wide = match bits {
                    Bits::B8 => ((num_hi as u64) << 8) | num_lo as u64,
                    Bits::B16 => ((num_hi as u64) << 16) | num_lo as u64,
                    Bits::B32 => ((num_hi as u64) << 32) | num_lo as u64,
                };
                let divisor = v as u64;
                let q = wide / divisor;
                let r = wide % divisor;
                if q > mask as u64 {
                    return dispatch_interrupt(cpu, 0, false, 0);
                }
                match bits {
                    Bits::B8 => {
                        cpu.set_reg8(0, q as u8);
                        cpu.set_reg8(4, r as u8);
                    }
                    Bits::B16 => {
                        cpu.set_reg16(0, q as u16);
                        cpu.set_reg16(2, r as u16);
                    }
                    Bits::B32 => {
                        cpu.set_reg32(0, q as u32);
                        cpu.set_reg32(2, r as u32);
                    }
                }
            } else {
                // IDIV
                let n = match bits {
                    Bits::B8 => (((num_hi as u16) << 8) | (num_lo as u16)) as i16 as i128,
                    Bits::B16 => (((num_hi as u32) << 16) | (num_lo as u32)) as i32 as i128,
                    Bits::B32 => (((num_hi as u64) << 32) | (num_lo as u64)) as i64 as i128,
                };
                let d = match bits {
                    Bits::B8 => v as i8 as i128,
                    Bits::B16 => v as i16 as i128,
                    Bits::B32 => v as i32 as i128,
                };
                if d == 0 {
                    return dispatch_interrupt(cpu, 0, false, 0);
                }
                let q = n / d;
                let r = n % d;
                // overflow check: quotient must fit in mask+1 signed.
                let min = -(1i128 << (bits.bytes() * 8 - 1));
                let max = (1i128 << (bits.bytes() * 8 - 1)) - 1;
                if q < min || q > max {
                    return dispatch_interrupt(cpu, 0, false, 0);
                }
                match bits {
                    Bits::B8 => {
                        cpu.set_reg8(0, q as u8);
                        cpu.set_reg8(4, r as u8);
                    }
                    Bits::B16 => {
                        cpu.set_reg16(0, q as u16);
                        cpu.set_reg16(2, r as u16);
                    }
                    Bits::B32 => {
                        cpu.set_reg32(0, q as u32);
                        cpu.set_reg32(2, r as u32);
                    }
                }
            }
            let _ = mask;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Imul1 => {
            let dst = d.ops[0];
            let src = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match dst {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) => b,
                _ => sz,
            };
            let imm = match d.ops[2] {
                Opnd::Imm(v) | Opnd::ImmSext(v) => Some(v),
                _ => None,
            };
            let b = match imm {
                Some(imm) => imm,
                None => src,
            };
            let (r, of) = match bits {
                Bits::B16 => {
                    let wide = (src as i16 as i32).wrapping_mul(b as i16 as i32);
                    let lo = wide as u16;
                    (lo as u32, wide != lo as i16 as i32)
                }
                Bits::B32 => {
                    let wide = (src as i32 as i64).wrapping_mul(b as i32 as i64);
                    let lo = wide as u32;
                    (lo, wide != lo as i32 as i64)
                }
                Bits::B8 => {
                    let wide = (src as i8 as i16).wrapping_mul(b as i8 as i16);
                    let lo = wide as u8;
                    (lo as u32, wide != lo as i8 as i16)
                }
            };
            write_opnd(cpu, &dst, r, AccessKind::Write).ok();
            set_cf(cpu, of);
            set_of(cpu, of);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Shift(kind, by) => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let count = match by {
                ShiftBy::One => 1,
                ShiftBy::Cl => cpu.reg8(1) as u32,
                ShiftBy::Imm(c) => c as u32,
            };
            let r = shift(cpu, kind, v, count, bits);
            if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Dshift(shl, by) => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let src = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let count = match by {
                ShiftBy::One => 1,
                ShiftBy::Cl => cpu.reg8(1) as u32,
                ShiftBy::Imm(c) => c as u32,
            };
            let bits_n = match bits {
                Bits::B8 => 8,
                Bits::B16 => 16,
                Bits::B32 => 32,
            };
            let mask = match bits {
                Bits::B8 => MASK8,
                Bits::B16 => MASK16,
                Bits::B32 => u32::MAX,
            };
            let (r, cf) = if shl {
                let cnt = count.min(bits_n);
                if cnt == 0 {
                    (v, false)
                } else {
                    let r = ((v << cnt) | (src >> (bits_n - cnt).max(0))) & mask;
                    (r, (v >> (bits_n - cnt)) & 1 != 0)
                }
            } else {
                let cnt = count.min(bits_n);
                if cnt == 0 {
                    (v, false)
                } else {
                    let r = ((v >> cnt) | (src << (bits_n - cnt).max(0))) & mask;
                    (r, (v >> (cnt - 1)) & 1 != 0)
                }
            };
            set_cf(cpu, cf);
            if count == 1 {
                set_of(cpu, ((v >> (bits_n - 1)) & 1) != (cf as u32));
            } else {
                set_of(cpu, false);
            }
            if let Err(e) = write_opnd(cpu, &d.ops[0], r, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Test => {
            let a = match read_opnd(cpu, &d.ops[0], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let b = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let mask = match bits {
                Bits::B8 => MASK8,
                Bits::B16 => MASK16,
                Bits::B32 => u32::MAX,
            };
            let r = a & b & mask;
            set_flags_zsp(cpu, r, mask);
            set_cf(cpu, false);
            set_of(cpu, false);
            set_af(cpu, false);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Bit(op) => {
            let src_opnd = if matches!(op, BitOp::Bsf | BitOp::Bsr) {
                d.ops[1]
            } else {
                d.ops[0]
            };
            let rm = match read_opnd(cpu, &src_opnd, AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let mem_flag = matches!(src_opnd, Opnd::Mem(..));
            let bits = match src_opnd {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let bits_n = match bits {
                Bits::B8 => 8,
                Bits::B16 => 16,
                Bits::B32 => 32,
            };
            let mask = match bits {
                Bits::B8 => MASK8,
                Bits::B16 => MASK16,
                Bits::B32 => u32::MAX,
            };
            match op {
                BitOp::Bt => {
                    let pos_in = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                        Ok(v) => v,
                        Err(e) => return StepOut::Error(e),
                    };
                    let pos = if mem_flag {
                        pos_in & 0x1F
                    } else {
                        pos_in % bits_n
                    };
                    set_cf(cpu, (rm >> pos) & 1 != 0);
                }
                BitOp::Bts | BitOp::Btr | BitOp::Btc => {
                    let pos_in = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                        Ok(v) => v,
                        Err(e) => return StepOut::Error(e),
                    };
                    let pos = if mem_flag {
                        pos_in & 0x1F
                    } else {
                        pos_in % bits_n
                    };
                    set_cf(cpu, (rm >> pos) & 1 != 0);
                    let bit = 1u32 << pos;
                    let nv = match op {
                        BitOp::Bts => rm | bit,
                        BitOp::Btr => rm & !bit,
                        _ => rm ^ bit,
                    };
                    if let Err(e) = write_opnd(cpu, &d.ops[0], nv & mask, AccessKind::Write) {
                        return StepOut::Error(e);
                    }
                }
                BitOp::Bsf | BitOp::Bsr => {
                    if rm & mask == 0 {
                        set_flags_zsp(cpu, 0, mask);
                        set_cf(cpu, false);
                        set_of(cpu, false);
                        set_af(cpu, false);
                    } else {
                        let idx = if op == BitOp::Bsf {
                            (rm & mask).trailing_zeros()
                        } else {
                            31 - (rm & mask).leading_zeros()
                        };
                        write_opnd(cpu, &d.ops[0], idx, AccessKind::Write).ok();
                        set_flags_zsp(cpu, rm & mask, mask);
                        set_cf(cpu, false);
                        set_of(cpu, false);
                        set_af(cpu, false);
                    }
                }
                _ => unreachable!(),
            }
            let _ = bits_n;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Mov => {
            let src = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            if let Err(e) = write_opnd(cpu, &d.ops[0], src, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Movzx(from8) | Op::Movsx(from8) => {
            let dst = d.ops[0];
            let src = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits_n = match d.ops[1] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) => {
                    if from8 {
                        8
                    } else {
                        b.bytes() * 8
                    }
                }
                _ => 8,
            };
            let v = if from8 {
                if matches!(d.op, Op::Movsx(_)) {
                    (src as i8 as i32) as u32
                } else {
                    src & 0xFF
                }
            } else if matches!(d.op, Op::Movsx(_)) {
                if bits_n == 16 {
                    (src as i16 as i32) as u32
                } else {
                    src
                }
            } else {
                src
            };
            let _ = bits_n;
            write_opnd(cpu, &dst, v, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lea => {
            let m = match d.ops[1] {
                Opnd::Mem(m, _) => m,
                _ => return StepOut::Error(Error::Internal("LEA mem".into())),
            };
            let ea = eff(cpu, &m);
            write_opnd(cpu, &d.ops[0], ea, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Xchg => {
            let a = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let b = match read_opnd(cpu, &d.ops[1], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            write_opnd(cpu, &d.ops[0], b, AccessKind::Write).ok();
            write_opnd(cpu, &d.ops[1], a, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cmov(c) => {
            if cond_true(cpu, c) {
                let src = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                write_opnd(cpu, &d.ops[0], src, AccessKind::Write).ok();
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Setcc(c) => {
            let v = if cond_true(cpu, c) { 1 } else { 0 };
            if let Err(e) = write_opnd(cpu, &d.ops[0], v, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Push => {
            let v = match read_opnd(cpu, &d.ops[0], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let v = match d.ops[0] {
                Opnd::Reg(_, Bits::B16) => v & 0xFFFF,
                Opnd::Imm(v) => v,
                _ => v,
            };
            if let Err(e) = push(cpu, size, v) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Pop => {
            let v = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            if let Err(e) = write_opnd(cpu, &d.ops[0], v, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::PushF => {
            let flags = if size == 2 {
                cpu.eflags & 0xFFFF
            } else {
                cpu.eflags
            };
            if let Err(e) = push(cpu, size, flags) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::PopF => {
            let v = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let saved = cpu.eflags & 0x100; // TF/IF/... preserved semantics
            let _ = saved;
            cpu.eflags = if size == 2 {
                (cpu.eflags & 0xFFFF_0000) | (v & 0xFFFF)
            } else {
                v
            };
            cpu.eflags |= 0x2;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::PushA => {
            let orig_esp = cpu.gpr[Reg::Esp as usize];
            for i in [0u8, 1, 2, 3] {
                if let Err(e) = push(cpu, size, cpu.gpr[i as usize]) {
                    return StepOut::Error(e);
                }
            }
            if let Err(e) = push(cpu, size, orig_esp) {
                return StepOut::Error(e);
            }
            for i in [5u8, 6, 7] {
                if let Err(e) = push(cpu, size, cpu.gpr[i as usize]) {
                    return StepOut::Error(e);
                }
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::PopA => {
            // Pop DI, SI, BP, skip SP, then BX, DX, CX, AX.
            if size == 2 {
                let di = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let si = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let bp = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let _ = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                }; // SP slot
                let bx = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let dx = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let cx = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let ax = match pop(cpu, 2) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                cpu.set_reg16(7, di as u16);
                cpu.set_reg16(6, si as u16);
                cpu.set_reg16(5, bp as u16);
                cpu.set_reg16(3, bx as u16);
                cpu.set_reg16(2, dx as u16);
                cpu.set_reg16(1, cx as u16);
                cpu.set_reg16(0, ax as u16);
            } else {
                let di = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let si = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let bp = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let _ = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                }; // SP slot
                let bx = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let dx = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let cx = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                let ax = match pop(cpu, 4) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                cpu.set_reg32(7, di);
                cpu.set_reg32(6, si);
                cpu.set_reg32(5, bp);
                cpu.set_reg32(3, bx);
                cpu.set_reg32(2, dx);
                cpu.set_reg32(1, cx);
                cpu.set_reg32(0, ax);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Call { far } => {
            if far {
                let (sel, off) = match read_far_target(cpu, &d.ops[0]) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                if let Err(e) = push(cpu, size, cpu.seg[Seg::Cs as usize].sel as u32) {
                    return StepOut::Error(e);
                }
                if let Err(e) = push(cpu, size, next) {
                    return StepOut::Error(e);
                }
                if let Err(e) = far_jump(cpu, sel, off) {
                    return StepOut::Error(e);
                }
            } else {
                let target = match read_near_target(cpu, &d.ops[0], next) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                if let Err(e) = push(cpu, size, next) {
                    return StepOut::Error(e);
                }
                cpu.eip = target;
            }
            StepOut::Ok
        }
        Op::Jump { far } => {
            if far {
                let (sel, off) = match read_far_target(cpu, &d.ops[0]) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                if let Err(e) = far_jump(cpu, sel, off) {
                    return StepOut::Error(e);
                }
            } else {
                cpu.eip = match read_near_target(cpu, &d.ops[0], next) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
            }
            StepOut::Ok
        }
        Op::Ret { far, imm } => {
            let ip = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            if far {
                let cs = match pop(cpu, size) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                cpu.seg[Seg::Cs as usize] = crate::cpu::SegVal::real(cs as u16);
            }
            cpu.eip = ip;
            if let Some(imm) = imm {
                let esp = cpu.gpr[Reg::Esp as usize].wrapping_add(imm as u32);
                cpu.gpr[Reg::Esp as usize] = esp;
            }
            StepOut::Ok
        }
        Op::Jcc(c) => {
            if cond_true(cpu, c) {
                cpu.eip = d.rel_target(next);
            } else {
                cpu.eip = next;
            }
            StepOut::Ok
        }
        Op::Loop(cond) => {
            let dec = cpu.gpr[Reg::Ecx as usize].wrapping_sub(1);
            if d.o16 {
                cpu.set_reg16(1, dec as u16);
            } else {
                cpu.gpr[Reg::Ecx as usize] = dec;
            }
            let taken = match cond {
                None => dec & if d.o16 { 0xFFFF } else { 0xFFFF_FFFF } != 0,
                Some(k) => {
                    let zf = cpu.eflags & ZF != 0;
                    (dec & if d.o16 { 0xFFFF } else { 0xFFFF_FFFF } != 0)
                        && if k == 0xE0 { !zf } else { zf }
                }
            };
            if taken {
                cpu.eip = d.rel_target(next);
            } else {
                cpu.eip = next;
            }
            StepOut::Ok
        }
        Op::Jcxz => {
            let cx = if d.a16 {
                cpu.gpr[Reg::Ecx as usize] & 0xFFFF
            } else {
                cpu.gpr[Reg::Ecx as usize]
            };
            if cx == 0 {
                cpu.eip = d.rel_target(next);
            } else {
                cpu.eip = next;
            }
            StepOut::Ok
        }
        Op::Movs(bits) => {
            let size_b = bits.bytes();
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                size_b.wrapping_neg()
            } else {
                size_b
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            for _ in 0..n {
                let b = match read_opnd(
                    cpu,
                    &Opnd::Mem(
                        crate::decode::MemRef {
                            seg: Seg::Ds,
                            base: Some(6),
                            index: None,
                            scale: 1,
                            disp: 0,
                            a16,
                        },
                        bits,
                    ),
                    AccessKind::Read,
                ) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                if let Err(e) = write_opnd(
                    cpu,
                    &Opnd::Mem(
                        crate::decode::MemRef {
                            seg: Seg::Es,
                            base: Some(7),
                            index: None,
                            scale: 1,
                            disp: 0,
                            a16,
                        },
                        bits,
                    ),
                    b,
                    AccessKind::Write,
                ) {
                    return StepOut::Error(e);
                }
                update_index(cpu, Reg::Esi, step, a16);
                update_index(cpu, Reg::Edi, step, a16);
            }
            if d.rep != Rep::None {
                set_count(cpu, 0, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Stos(bits) => {
            let size_b = bits.bytes();
            let a16 = d.a16;
            let acc = read_gpr(cpu, 0, bits);
            let step = if cpu.eflags & flag::DF != 0 {
                size_b.wrapping_neg()
            } else {
                size_b
            };
            let m = crate::decode::MemRef {
                seg: Seg::Es,
                base: Some(7),
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            for _ in 0..n {
                if let Err(e) = write_opnd(cpu, &Opnd::Mem(m, bits), acc, AccessKind::Write) {
                    return StepOut::Error(e);
                }
                update_index(cpu, Reg::Edi, step, a16);
            }
            if d.rep != Rep::None {
                set_count(cpu, 0, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lods(bits) => {
            let size_b = bits.bytes();
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                size_b.wrapping_neg()
            } else {
                size_b
            };
            let m = crate::decode::MemRef {
                seg: Seg::Ds,
                base: Some(6),
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            for _ in 0..n {
                let v = match read_opnd(cpu, &Opnd::Mem(m, bits), AccessKind::Read) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                if let Err(e) = write_opnd(cpu, &Opnd::Acc(bits), v, AccessKind::Write) {
                    return StepOut::Error(e);
                }
                update_index(cpu, Reg::Esi, step, a16);
            }
            if d.rep != Rep::None {
                set_count(cpu, 0, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cmps(bits) => {
            let size_b = bits.bytes();
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                size_b.wrapping_neg()
            } else {
                size_b
            };
            let m = |base: Option<u8>, seg: Seg| crate::decode::MemRef {
                seg,
                base,
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            let mut cx = n;
            for _ in 0..n {
                let a =
                    match read_opnd(cpu, &Opnd::Mem(m(Some(6), Seg::Ds), bits), AccessKind::Read) {
                        Ok(v) => v,
                        Err(e) => return StepOut::Error(e),
                    };
                let b =
                    match read_opnd(cpu, &Opnd::Mem(m(Some(7), Seg::Es), bits), AccessKind::Read) {
                        Ok(v) => v,
                        Err(e) => return StepOut::Error(e),
                    };
                alu(cpu, AluOp::Cmp, a, b, bits);
                update_index(cpu, Reg::Esi, step, a16);
                update_index(cpu, Reg::Edi, step, a16);
                if d.rep != Rep::None {
                    cx = cx.wrapping_sub(1);
                    let zf = cpu.eflags & ZF != 0;
                    if (d.rep == Rep::Z && !zf) || (d.rep == Rep::NZ && zf) {
                        break;
                    }
                }
            }
            if d.rep != Rep::None {
                set_count(cpu, cx, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Scas(bits) => {
            let size_b = bits.bytes();
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                size_b.wrapping_neg()
            } else {
                size_b
            };
            let acc = read_gpr(cpu, 0, bits);
            let m = crate::decode::MemRef {
                seg: Seg::Es,
                base: Some(7),
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            let mut cx = n;
            for _ in 0..n {
                let v = match read_opnd(cpu, &Opnd::Mem(m, bits), AccessKind::Read) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                alu(cpu, AluOp::Cmp, acc, v, bits);
                update_index(cpu, Reg::Edi, step, a16);
                if d.rep != Rep::None {
                    cx = cx.wrapping_sub(1);
                    let zf = cpu.eflags & ZF != 0;
                    if (d.rep == Rep::Z && !zf) || (d.rep == Rep::NZ && zf) {
                        break;
                    }
                }
            }
            if d.rep != Rep::None {
                set_count(cpu, cx, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Ins(bits) => {
            let port = cpu.gpr[Reg::Edx as usize] as u16;
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                bits.bytes().wrapping_neg()
            } else {
                bits.bytes()
            };
            let m = crate::decode::MemRef {
                seg: Seg::Es,
                base: Some(7),
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            for _ in 0..n {
                let v = cpu.mem.io_read(port, bits.bytes() as u8);
                if let Err(e) = write_opnd(cpu, &Opnd::Mem(m, bits), v, AccessKind::Write) {
                    return StepOut::Error(e);
                }
                update_index(cpu, Reg::Edi, step, a16);
            }
            if d.rep != Rep::None {
                set_count(cpu, 0, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Outs(bits) => {
            let port = cpu.gpr[Reg::Edx as usize] as u16;
            let a16 = d.a16;
            let step = if cpu.eflags & flag::DF != 0 {
                bits.bytes().wrapping_neg()
            } else {
                bits.bytes()
            };
            let m = crate::decode::MemRef {
                seg: Seg::Ds,
                base: Some(6),
                index: None,
                scale: 1,
                disp: 0,
                a16,
            };
            let n = if d.rep == Rep::None {
                1
            } else {
                count_reg(cpu, a16)
            };
            for _ in 0..n {
                let v = match read_opnd(cpu, &Opnd::Mem(m, bits), AccessKind::Read) {
                    Ok(v) => v,
                    Err(e) => return StepOut::Error(e),
                };
                cpu.mem.io_write(port, bits.bytes() as u8, v);
                update_index(cpu, Reg::Esi, step, a16);
            }
            if d.rep != Rep::None {
                set_count(cpu, 0, a16);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Xlat => {
            let al = cpu.reg8(0) as u32;
            let base = if d.a16 {
                cpu.gpr[Reg::Ebx as usize] as u16 as u32
            } else {
                cpu.gpr[Reg::Ebx as usize]
            };
            let off = base.wrapping_add(al);
            let m = crate::decode::MemRef {
                seg: Seg::Ds,
                base: None,
                index: None,
                scale: 1,
                disp: off as i32,
                a16: d.a16,
            };
            let v = match read_opnd(cpu, &Opnd::Mem(m, Bits::B8), AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            cpu.set_reg8(0, v as u8);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::In => {
            let port = match d.ops[1] {
                Opnd::Port(p) => p,
                Opnd::Dx => cpu.gpr[Reg::Edx as usize] as u16,
                _ => 0,
            };
            let size_b = match d.ops[0] {
                Opnd::Acc(b) => b.bytes(),
                _ => 1,
            };
            let v = cpu.mem.io_read(port, size_b as u8);
            write_opnd(cpu, &d.ops[0], v, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Out => {
            let port = match d.ops[0] {
                Opnd::Port(p) => p,
                Opnd::Dx => cpu.gpr[Reg::Edx as usize] as u16,
                _ => 0,
            };
            let v = read_opnd(cpu, &d.ops[1], AccessKind::Read).unwrap_or(0);
            let size_b = match d.ops[1] {
                Opnd::Acc(b) => b.bytes(),
                _ => 1,
            };
            cpu.mem.io_write(port, size_b as u8, v);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Int(v) => {
            cpu.eip = next;
            dispatch_interrupt(cpu, v, false, 0)
        }
        Op::Int3 => {
            cpu.eip = next;
            dispatch_interrupt(cpu, 3, false, 0)
        }
        Op::Into => {
            if cpu.eflags & OF != 0 {
                cpu.eip = next;
                dispatch_interrupt(cpu, 4, false, 0)
            } else {
                cpu.eip = next;
                StepOut::Ok
            }
        }
        Op::Iret => {
            let ip = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let cs = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let fl = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            cpu.eip = ip;
            cpu.seg[Seg::Cs as usize] = crate::cpu::SegVal::real(cs as u16);
            cpu.eflags = if size == 2 {
                (cpu.eflags & 0xFFFF_0000) | (fl & 0xFFFF)
            } else {
                fl
            };
            cpu.eflags |= 0x2;
            StepOut::Ok
        }
        Op::Hlt => {
            cpu.eip = next;
            if cpu.eflags & flag::IF != 0 {
                cpu.halted = true;
                StepOut::Ok
            } else {
                StepOut::Error(Error::Halt)
            }
        }
        Op::Cli => {
            cpu.eflags &= !flag::IF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Sti => {
            cpu.eflags |= flag::IF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cld => {
            cpu.eflags &= !flag::DF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Std => {
            cpu.eflags |= flag::DF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Clc => {
            cpu.eflags &= !CF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Stc => {
            cpu.eflags |= CF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cmc => {
            cpu.eflags ^= CF;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Salc => {
            cpu.set_reg8(0, if cpu.eflags & CF != 0 { 0xFF } else { 0 });
            cpu.eip = next;
            StepOut::Ok
        }
        Op::MovRm2Sreg => {
            let sel = match read_opnd(cpu, &d.ops[1], AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let sreg = match d.ops[0] {
                Opnd::Sreg(s) => s,
                _ => return StepOut::Error(Error::Internal("sreg".into())),
            };
            if let Err(e) = load_segment(cpu, sreg, sel as u16) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::MovSreg2Rm => {
            let sel = match d.ops[1] {
                Opnd::Sreg(s) => cpu.seg[s as usize].sel as u32,
                _ => 0,
            };
            if let Err(e) = write_opnd(cpu, &d.ops[0], sel, AccessKind::Write) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::PopSeg(sreg) => {
            let sel = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            } as u16;
            if let Err(e) = load_segment(cpu, sreg, sel) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lss | Op::Lds | Op::Les | Op::Lfs | Op::Lgs => {
            let m = match d.ops[1] {
                Opnd::Mem(m, _) => m,
                _ => return StepOut::Error(Error::Internal("lds mem".into())),
            };
            let off = eff(cpu, &m);
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) => b,
                _ => Bits::B32,
            };
            let val = match bits {
                Bits::B16 => mmu::read16(cpu, m.seg, off, AccessKind::Read).map(|v| v as u32),
                Bits::B32 => mmu::read32(cpu, m.seg, off, AccessKind::Read),
                Bits::B8 => unreachable!(),
            };
            let sel_off = off.wrapping_add(bits.bytes());
            let sel = mmu::read16(cpu, m.seg, sel_off, AccessKind::Read);
            let v = match val {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let s = match sel {
                Ok(s) => s,
                Err(e) => return StepOut::Error(e),
            };
            let dst = match d.ops[0] {
                Opnd::Reg(r, _) => r,
                _ => return StepOut::Error(Error::Internal("lds reg".into())),
            };
            write_gpr(cpu, dst, bits, v);
            let sreg = match d.op {
                Op::Lss => Seg::Ss,
                Op::Lds => Seg::Ds,
                Op::Les => Seg::Es,
                Op::Lfs => Seg::Fs,
                _ => Seg::Gs,
            };
            if sreg == Seg::Ss {
                // special: SS cannot be loaded directly; emulate via descriptor
                if let Err(e) = load_segment(cpu, sreg, s) {
                    return StepOut::Error(e);
                }
            } else {
                if let Err(e) = load_segment(cpu, sreg, s) {
                    return StepOut::Error(e);
                }
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lar | Op::Lsl => {
            let sel_opnd = d.ops[1];
            let sel = match read_opnd(cpu, &sel_opnd, AccessKind::Read) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            } as u16;
            let d_ = load_descriptor(cpu, sel);
            let ok = match d_ {
                Ok(dd) => {
                    // LAR: valid if present and not a system/gate; LSL similar.
                    let allowed = dd.typ & 0x08 != 0 || (dd.typ & 0x0C) == 0x00;
                    if allowed {
                        let val = if matches!(d.op, Op::Lar) {
                            (desc_access_byte(&dd) as u32) & 0x00FF_FF00
                                | ((dd.dpl as u32) << 8)
                                | ((if dd.db { 1 } else { 0 }) << 22)
                                | ((if dd.g { 1 } else { 0 }) << 23)
                        } else {
                            dd.eff_limit()
                        };
                        write_opnd(cpu, &d.ops[0], val, AccessKind::Write).ok();
                        set_flags_zsp(cpu, val, u32::MAX);
                        set_cf(cpu, false);
                        set_of(cpu, false);
                        set_af(cpu, false);
                        true
                    } else {
                        set_flags_zsp(cpu, 0, u32::MAX);
                        set_cf(cpu, false);
                        set_of(cpu, false);
                        set_af(cpu, false);
                        false
                    }
                }
                Err(_) => {
                    set_flags_zsp(cpu, 0, u32::MAX);
                    set_cf(cpu, false);
                    set_of(cpu, false);
                    set_af(cpu, false);
                    false
                }
            };
            if ok {
                set_cf(cpu, false);
            } else {
                set_cf(cpu, true);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::VErr(_) => {
            // VERR/VERW: set ZF = 1 if segment readable/writable; we approximate.
            let sel = read_opnd(cpu, &d.ops[0], AccessKind::Read).unwrap_or(0) as u16;
            let d_ = load_descriptor(cpu, sel);
            let readable = d_.as_ref().map(|d| d.typ & 0x08 != 0).unwrap_or(false);
            let writable = d_.as_ref().map(|d| (d.typ & 0x0C) == 0x02).unwrap_or(false);
            let v = match d.op {
                Op::VErr(true) => readable,
                _ => writable,
            };
            if v {
                cpu.eflags |= ZF;
            } else {
                cpu.eflags &= !ZF;
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Arpl => {
            // ARPL r/m16, r16: adjusts RPL field.
            let a = read_opnd(cpu, &d.ops[0], AccessKind::Rmw).unwrap_or(0) as u16;
            let b = read_opnd(cpu, &d.ops[1], AccessKind::Read).unwrap_or(0) as u16;
            let rpl = b & 3;
            let (nv, changed) = if (a & 3) < rpl {
                ((a & !3) | rpl, true)
            } else {
                (a, false)
            };
            write_opnd(cpu, &d.ops[0], nv as u32, AccessKind::Write).ok();
            if changed {
                cpu.eflags |= ZF;
            } else {
                cpu.eflags &= !ZF;
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lgdt | Op::Lidt => {
            let m = match d.ops[0] {
                Opnd::Mem(m, _) => m,
                _ => return StepOut::Error(Error::Internal("lgdt mem".into())),
            };
            let off = eff(cpu, &m);
            let limit = mmu::read16(cpu, m.seg, off, AccessKind::Read).unwrap_or(0);
            let base = mmu::read32(cpu, m.seg, off + 2, AccessKind::Read).unwrap_or(0);
            if d.op == Op::Lgdt {
                cpu.gdtr.base = base;
                cpu.gdtr.limit = limit;
            } else {
                cpu.idtr.base = base;
                cpu.idtr.limit = limit;
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Sgdt | Op::Sidt => {
            let m = match d.ops[0] {
                Opnd::Mem(m, _) => m,
                _ => return StepOut::Error(Error::Internal("sgdt mem".into())),
            };
            let off = eff(cpu, &m);
            let (base, limit) = if d.op == Op::Sgdt {
                (cpu.gdtr.base, cpu.gdtr.limit)
            } else {
                (cpu.idtr.base, cpu.idtr.limit)
            };
            mmu::write16(cpu, m.seg, off, limit, AccessKind::Write).ok();
            mmu::write32(cpu, m.seg, off + 2, base, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lldt => {
            let sel = read_opnd(cpu, &d.ops[0], AccessKind::Read).unwrap_or(0) as u16;
            if let Err(e) = load_ldtr(cpu, sel) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Sldt => {
            let sel = cpu.ldtr.sel as u32;
            write_opnd(cpu, &d.ops[0], sel, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Ltr => {
            let sel = read_opnd(cpu, &d.ops[0], AccessKind::Read).unwrap_or(0) as u16;
            if let Err(e) = load_tr(cpu, sel) {
                return StepOut::Error(e);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Str => {
            let sel = cpu.tr.sel as u32;
            write_opnd(cpu, &d.ops[0], sel, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Smsw => {
            let v = cpu.cr[0] & 0xFFFF;
            write_opnd(cpu, &d.ops[0], v, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lmsw => {
            let v = read_opnd(cpu, &d.ops[0], AccessKind::Read).unwrap_or(0) as u16;
            cpu.cr[0] = (cpu.cr[0] & !0xF) | ((v & 0xF) as u32);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Clts => {
            cpu.cr[0] &= !(1 << 3);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::MovCr => {
            let cr = match d.ops[0] {
                Opnd::Imm(v) => v as usize,
                _ => return StepOut::Error(Error::Internal("movcr".into())),
            };
            let gpr = match d.ops[1] {
                Opnd::Reg(r, _) => r as usize,
                _ => return StepOut::Error(Error::Internal("movcr".into())),
            };
            let to_cr = match d.ops[2] {
                Opnd::Imm(v) => v != 0,
                _ => false,
            };
            if to_cr {
                if cr == 0 || cr == 1 || cr == 3 || cr == 4 {
                    cpu.cr[cr] = cpu.gpr[gpr];
                }
            } else {
                cpu.gpr[gpr] = cpu.cr[cr.min(3)];
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::MovDr => {
            let dr = match d.ops[0] {
                Opnd::Imm(v) => v as usize,
                _ => 0,
            };
            let gpr = match d.ops[1] {
                Opnd::Reg(r, _) => r as usize,
                _ => return StepOut::Error(Error::Internal("movdr".into())),
            };
            let to_dr = match d.ops[2] {
                Opnd::Imm(v) => v != 0,
                _ => false,
            };
            if to_dr {
                // DR0-3; DR6/7 are status/control. We store them in a scratch.
                cpu.cr[3] = if dr == 0 { cpu.gpr[gpr] } else { cpu.cr[3] };
            } else {
                cpu.gpr[gpr] = if dr == 0 { cpu.cr[3] } else { 0 };
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Bswap => {
            let r = match d.ops[0] {
                Opnd::Reg(r, _) => r,
                _ => return StepOut::Error(Error::Internal("bswap".into())),
            };
            let v = cpu.gpr[r as usize];
            cpu.gpr[r as usize] = v.swap_bytes();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Xadd => {
            let a = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let b = match read_opnd(cpu, &d.ops[1], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let r = alu(cpu, AluOp::Add, a, b, bits);
            write_opnd(cpu, &d.ops[0], r, AccessKind::Write).ok();
            write_opnd(cpu, &d.ops[1], a, AccessKind::Write).ok();
            cpu.eip = next;
            StepOut::Ok
        }
        Op::CmpXchg => {
            let a = match read_opnd(cpu, &d.ops[0], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let b = match read_opnd(cpu, &d.ops[1], AccessKind::Rmw) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            let bits = match d.ops[0] {
                Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
                _ => sz,
            };
            let acc = read_gpr(cpu, 0, bits);
            let r = alu(cpu, AluOp::Cmp, acc, a, bits);
            if r == 0 {
                write_opnd(cpu, &d.ops[0], b, AccessKind::Write).ok();
            } else {
                write_gpr(cpu, 0, bits, a);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cpuid => {
            // CPUID is 486+; on a 386 it should #UD. Bochs BIOS doesn't use it.
            cpu.gpr[0] = 1;
            cpu.gpr[1] = 0x486;
            cpu.gpr[2] = 0;
            cpu.gpr[3] = 0;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Rdtsc => {
            cpu.gpr[0] = cpu.cycles as u32;
            cpu.gpr[2] = (cpu.cycles >> 32) as u32;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Rdmsr | Op::Wrmsr => {
            // MSRs don't exist on the 386; no-op (they'd #UD on real hardware,
            // but no BIOS code should hit them).
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Invd | Op::Wbinvd => {
            // Flush caches; no-op.
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Invlpg => {
            // Flush a TLB entry; no-op (we don't cache).
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Nop | Op::Wait => {
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Daa => {
            let al = cpu.reg8(0);
            let old = al;
            if cpu.eflags & AF != 0 || (al & 0xF) > 9 {
                let r = al.wrapping_add(0x06);
                cpu.set_reg8(0, r);
                cpu.eflags |= AF;
            } else {
                cpu.eflags &= !AF;
            }
            if cpu.eflags & CF != 0 || old > 0x99 {
                cpu.set_reg8(0, cpu.reg8(0).wrapping_add(0x60));
                cpu.eflags |= CF;
            } else {
                cpu.eflags &= !CF;
            }
            set_flags_zsp(cpu, cpu.reg8(0) as u32, MASK8);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Das => {
            let al = cpu.reg8(0);
            let old = al;
            if cpu.eflags & AF != 0 || (al & 0xF) > 9 {
                let r = al.wrapping_sub(0x06);
                cpu.set_reg8(0, r);
                cpu.eflags |= AF;
            } else {
                cpu.eflags &= !AF;
            }
            if cpu.eflags & CF != 0 || old > 0x99 {
                cpu.set_reg8(0, cpu.reg8(0).wrapping_sub(0x60));
                cpu.eflags |= CF;
            } else {
                cpu.eflags &= !CF;
            }
            set_flags_zsp(cpu, cpu.reg8(0) as u32, MASK8);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Aaa => {
            if cpu.eflags & AF != 0 || (cpu.reg8(0) & 0xF) > 9 {
                cpu.set_reg8(0, cpu.reg8(0).wrapping_add(6));
                cpu.set_reg16(0, cpu.reg16(0).wrapping_add(0x100));
                cpu.eflags |= AF;
                cpu.eflags |= CF;
            } else {
                cpu.eflags &= !(AF | CF);
            }
            cpu.set_reg8(0, cpu.reg8(0) & 0xF);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Aas => {
            if cpu.eflags & AF != 0 || (cpu.reg8(0) & 0xF) > 9 {
                cpu.set_reg8(0, cpu.reg8(0).wrapping_sub(6));
                cpu.set_reg16(0, cpu.reg16(0).wrapping_sub(0x100));
                cpu.eflags |= AF;
                cpu.eflags |= CF;
            } else {
                cpu.eflags &= !(AF | CF);
            }
            cpu.set_reg8(0, cpu.reg8(0) & 0xF);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Aam => {
            let al = cpu.reg8(0);
            let ah = al / 10;
            let al_ = al % 10;
            cpu.set_reg8(4, ah); // AH
            cpu.set_reg8(0, al_);
            set_flags_zsp(cpu, al_ as u32, MASK8);
            cpu.eflags &= !(CF | OF | AF);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Aad => {
            let al = cpu.reg8(0);
            let ah = cpu.reg8(4);
            let v = (al as u16).wrapping_add((ah as u16) * 10);
            cpu.set_reg8(0, v as u8);
            cpu.set_reg8(4, 0);
            set_flags_zsp(cpu, v as u32, MASK8);
            cpu.eflags &= !(CF | OF | AF);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cbw => {
            if d.o16 {
                let al = cpu.reg8(0);
                let ax = al as i8 as i16;
                cpu.set_reg16(0, ax as u16);
            } else {
                let ax = cpu.reg16(0);
                cpu.set_reg32(0, ax as i16 as i32 as u32);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Cwd => {
            if d.o16 {
                let ax = cpu.reg16(0);
                let dx = ((ax as i16) >> 15) as u16;
                cpu.set_reg16(2, dx);
            } else {
                let eax = cpu.reg32(0);
                let edx = ((eax as i32) >> 31) as u32;
                cpu.set_reg32(2, edx);
            }
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Sahf => {
            cpu.eflags = (cpu.eflags & !0xFF) | ((cpu.reg8(4) as u32) & 0xFF); // AH -> flags
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Lahf => {
            cpu.set_reg8(4, (cpu.eflags & 0xFF) as u8);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Enter(imm, level) => {
            // Frame creation: push EBP, frame pointer chain.
            let frame_size = imm as u32;
            let esp = cpu.gpr[Reg::Esp as usize];
            let ebp = cpu.gpr[Reg::Ebp as usize];
            let size = stack_size(d);
            if let Err(e) = push(cpu, size, ebp) {
                return StepOut::Error(e);
            }
            let new_ebp = cpu.gpr[Reg::Esp as usize];
            for _ in 1..level.saturating_sub(1) {
                // Simplified: only handle level 0 and 1 properly.
                // level 0: new EBP = ESP-4 (old EBP pushed).
                // level 1: push [old EBP].
                if level > 1 && level <= 31 {
                    let esp_now = cpu.gpr[Reg::Esp as usize];
                    let old_ebp = cpu.mem.phys_read32(esp_now + size);
                    push(cpu, size, old_ebp).ok();
                }
            }
            if level == 0 {
                cpu.gpr[Reg::Ebp as usize] = new_ebp;
            } else {
                if level >= 1 {
                    // use the previous EBP chain; keep simple: new_ebp = old ebp pushed
                    cpu.gpr[Reg::Ebp as usize] = new_ebp;
                }
            }
            let _ = esp;
            let _ = frame_size;
            cpu.gpr[Reg::Esp as usize] = cpu.gpr[Reg::Esp as usize].wrapping_sub(frame_size);
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Leave => {
            let ebp = cpu.gpr[Reg::Ebp as usize];
            let size = stack_size(d);
            cpu.gpr[Reg::Esp as usize] = ebp;
            let old_ebp = match pop(cpu, size) {
                Ok(v) => v,
                Err(e) => return StepOut::Error(e),
            };
            cpu.gpr[Reg::Ebp as usize] = old_ebp;
            cpu.eip = next;
            StepOut::Ok
        }
        Op::Unsupported(b) | Op::TwoByte(b) => StepOut::Error(Error::Unsupported(format!(
            "opcode {b:#04x} at eip {:#x}",
            cpu.eip
        ))),
        _ => {
            // Fallback for any missed opcode.
            cpu.eip = next;
            StepOut::Error(Error::Unsupported(format!(
                "unhandled op {:?} at eip {:#x}",
                d.op, cpu.eip
            )))
        }
    };
    if !matches!(cpu.mode(), Mode::Protected32) {
        cpu.eip &= 0xFFFF;
    }
    out
}

// ---------------------------------------------------------------------------
// Descriptor handling
// ---------------------------------------------------------------------------

fn load_segment(cpu: &mut X86, seg: Seg, sel: u16) -> Result<(), Error> {
    if cpu.cr[0] & 1 == 0 {
        cpu.seg[seg as usize] = crate::cpu::SegVal::real(sel);
        return Ok(());
    }
    if sel & !3 == 0 {
        if matches!(seg, Seg::Cs | Seg::Ss) {
            return Err(Error::Internal("null CS/SS selector".into()));
        }
        cpu.seg[seg as usize] = crate::cpu::SegVal {
            sel,
            desc: Desc::NULL,
        };
        return Ok(());
    }
    // Protected mode: load GDT/LDT descriptor.
    let d = load_descriptor(cpu, sel)?;
    if !d.p {
        return Err(Error::Internal("segment not present".into()));
    }
    cpu.seg[seg as usize] = crate::cpu::SegVal { sel, desc: d };
    Ok(())
}

fn load_descriptor(cpu: &X86, sel: u16) -> Result<Desc, Error> {
    let ti = sel & 4 != 0;
    let index = (sel >> 3) as u32;
    let base = if ti {
        cpu.ldtr.desc.base
    } else {
        cpu.gdtr.base
    };
    let limit = if ti {
        cpu.ldtr.desc.limit as u32
    } else {
        cpu.gdtr.limit as u32
    };
    if index * 8 + 7 > limit {
        return Err(Error::Internal("descriptor out of range".into()));
    }
    let addr = base.wrapping_add(index * 8);
    let lo = cpu.mem.phys_read32(addr);
    let hi = cpu.mem.phys_read32(addr + 4);
    let base = ((lo >> 16) & 0xFFFF) | ((hi & 0xFF) << 16) | ((hi >> 24) << 24);
    let limit = (lo & 0xFFFF) | (((hi >> 16) & 0xF) << 16);
    let g = hi & (1 << 23) != 0;
    let db = hi & (1 << 22) != 0;
    let p = hi & (1 << 15) != 0;
    let typ = ((hi >> 8) & 0x1F) as u8;
    let dpl = ((hi >> 13) & 3) as u8;
    let avl = hi & (1 << 20) != 0;
    Ok(Desc {
        base,
        limit,
        g,
        db,
        p,
        typ,
        dpl,
        avl,
    })
}

fn load_ldtr(cpu: &mut X86, sel: u16) -> Result<(), Error> {
    let d = load_descriptor(cpu, sel)?;
    if d.typ != 0x02 {
        return Err(Error::Internal("LLDT of non-LDT".into()));
    }
    cpu.ldtr = crate::cpu::SegVal { sel, desc: d };
    Ok(())
}

fn load_tr(cpu: &mut X86, sel: u16) -> Result<(), Error> {
    let d = load_descriptor(cpu, sel)?;
    if d.typ != 0x09 && d.typ != 0x0B {
        return Err(Error::Internal("LTR of non-TSS".into()));
    }
    // Load ESP0/SS0 from the TSS (helpful for ring transitions).
    let base = d.base;
    let ss0 = cpu.mem.phys_read16(base + 8);
    let esp0 = cpu.mem.phys_read32(base + 4);
    cpu.tss.ss0 = ss0;
    cpu.tss.esp0 = esp0;
    cpu.tss.cr3 = cpu.mem.phys_read32(base);
    cpu.tr = crate::cpu::SegVal { sel, desc: d };
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers for the interpreter's duplicate write (fixes ALU double-eval bug)
// ---------------------------------------------------------------------------

/// Compute the ALU result for `a op b` without writing flags (used after
/// `alu` has already been called, so we just recompute the raw value).
pub(crate) fn a_after(_r: u32, _bits: Bits) -> u32 {
    _r
}

/// Access byte of a descriptor (for LAR).
fn desc_access_byte(d: &Desc) -> u8 {
    (d.p as u8) << 7 | d.dpl << 5 | d.typ
}

// ---------------------------------------------------------------------------
// Public entry points used by the JIT (native helper calls).
// ---------------------------------------------------------------------------

/// Compute flags for TEST (a & b). Can't reuse `alu` (which writes flags for
/// AND but also needs the result); TEST only sets flags.
pub fn test_flags(cpu: &mut X86, a: u32, b: u32, sz: Bits) -> u32 {
    let mask = match sz {
        Bits::B8 => MASK8,
        Bits::B16 => MASK16,
        Bits::B32 => u32::MAX,
    };
    let r = a & b & mask;
    set_flags_zsp(cpu, r, mask);
    set_cf(cpu, false);
    set_of(cpu, false);
    set_af(cpu, false);
    r
}

/// NEG: flags from 0 - v, result = -v.
pub fn neg_op(cpu: &mut X86, v: u32, sz: Bits) -> u32 {
    let mask = match sz {
        Bits::B8 => MASK8,
        Bits::B16 => MASK16,
        Bits::B32 => u32::MAX,
    };
    let _ = alu(cpu, AluOp::Sub, 0, v, sz);
    (0u32.wrapping_sub(v)) & mask
}

/// Push a value (JIT helper).
pub fn push_public(cpu: &mut X86, size: u32, val: u32) -> Result<(), Error> {
    push(cpu, size, val)
}

/// Pop a value (JIT helper).
pub fn pop_public(cpu: &mut X86, size: u32) -> Result<u32, Error> {
    pop(cpu, size)
}

/// Execute one string instruction (shared by interpreter string arms and the
/// JIT helper). `op`: 0=movs,1=stos,2=lods,3=cmps,4=scas,5=ins,6=outs.
/// Handles REP/REPE/REPNE over the whole length, matching the interpreter.
pub fn string_op(cpu: &mut X86, op: u8, rep: Rep, bits: Bits, a16: u32) {
    let step = if cpu.eflags & flag::DF != 0 {
        bits.bytes().wrapping_neg()
    } else {
        bits.bytes()
    };
    let a16 = a16 != 0;
    let mk = |base: Option<u8>, seg: Seg| MemRef {
        seg,
        base,
        index: None,
        scale: 1,
        disp: 0,
        a16,
    };
    let acc = read_gpr(cpu, 0, bits);

    match (op, rep) {
        (1, Rep::Z) => {
            // REP STOS
            let count = count_reg(cpu, a16);
            for _ in 0..count {
                let _ = write_mem(cpu, &mk(Some(7), Seg::Es), bits, acc, AccessKind::Write);
                update_index(cpu, Reg::Edi, step, a16);
            }
            set_count(cpu, 0, a16);
        }
        (0, Rep::Z) => {
            // REP MOVS
            let count = count_reg(cpu, a16);
            for _ in 0..count {
                let v = read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                let _ = write_mem(cpu, &mk(Some(7), Seg::Es), bits, v, AccessKind::Write);
                update_index(cpu, Reg::Esi, step, a16);
                update_index(cpu, Reg::Edi, step, a16);
            }
            set_count(cpu, 0, a16);
        }
        (2, Rep::Z) => {
            let count = count_reg(cpu, a16);
            for _ in 0..count {
                let v = read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                write_gpr(cpu, 0, bits, v);
                update_index(cpu, Reg::Esi, step, a16);
            }
            set_count(cpu, 0, a16);
        }
        (3, Rep::Z) | (3, Rep::NZ) => {
            let count = count_reg(cpu, a16);
            let mut cx = count;
            for _ in 0..count {
                let a = read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                let b = read_mem(cpu, &mk(Some(7), Seg::Es), bits, AccessKind::Read).unwrap_or(0);
                alu(cpu, AluOp::Cmp, a, b, bits);
                update_index(cpu, Reg::Esi, step, a16);
                update_index(cpu, Reg::Edi, step, a16);
                cx -= 1;
                let zf = cpu.eflags & ZF != 0;
                if (rep == Rep::Z && !zf) || (rep == Rep::NZ && zf) {
                    break;
                }
            }
            set_count(cpu, cx, a16);
        }
        (4, Rep::Z) | (4, Rep::NZ) => {
            let count = count_reg(cpu, a16);
            let mut cx = count;
            for _ in 0..count {
                let v = read_mem(cpu, &mk(Some(7), Seg::Es), bits, AccessKind::Read).unwrap_or(0);
                alu(cpu, AluOp::Cmp, acc, v, bits);
                update_index(cpu, Reg::Edi, step, a16);
                cx -= 1;
                let zf = cpu.eflags & ZF != 0;
                if (rep == Rep::Z && !zf) || (rep == Rep::NZ && zf) {
                    break;
                }
            }
            set_count(cpu, cx, a16);
        }
        _ => {
            // Single step (possibly with a useless REP in front — the value
            // of ECX doesn't matter for the helper path).
            match op {
                0 => {
                    let v =
                        read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                    let _ = write_mem(cpu, &mk(Some(7), Seg::Es), bits, v, AccessKind::Write);
                    update_index(cpu, Reg::Esi, step, a16);
                    update_index(cpu, Reg::Edi, step, a16);
                }
                1 => {
                    let _ = write_mem(cpu, &mk(Some(7), Seg::Es), bits, acc, AccessKind::Write);
                    update_index(cpu, Reg::Edi, step, a16);
                }
                2 => {
                    let v =
                        read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                    write_gpr(cpu, 0, bits, v);
                    update_index(cpu, Reg::Esi, step, a16);
                }
                3 => {
                    let a =
                        read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                    let b =
                        read_mem(cpu, &mk(Some(7), Seg::Es), bits, AccessKind::Read).unwrap_or(0);
                    alu(cpu, AluOp::Cmp, a, b, bits);
                    update_index(cpu, Reg::Esi, step, a16);
                    update_index(cpu, Reg::Edi, step, a16);
                }
                4 => {
                    let v =
                        read_mem(cpu, &mk(Some(7), Seg::Es), bits, AccessKind::Read).unwrap_or(0);
                    alu(cpu, AluOp::Cmp, acc, v, bits);
                    update_index(cpu, Reg::Edi, step, a16);
                }
                5 => {
                    let port = cpu.gpr[Reg::Edx as usize] as u16;
                    let v = cpu.mem.io_read(port, bits.bytes() as u8);
                    let _ = write_mem(cpu, &mk(Some(7), Seg::Es), bits, v, AccessKind::Write);
                    update_index(cpu, Reg::Edi, step, a16);
                }
                _ => {
                    let port = cpu.gpr[Reg::Edx as usize] as u16;
                    let v =
                        read_mem(cpu, &mk(Some(6), Seg::Ds), bits, AccessKind::Read).unwrap_or(0);
                    cpu.mem.io_write(port, bits.bytes() as u8, v);
                    update_index(cpu, Reg::Esi, step, a16);
                }
            }
        }
    }
}
