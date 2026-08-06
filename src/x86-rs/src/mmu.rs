//! Memory management: linear-address computation (16/32-bit effective
//! addresses), segment translation (real + protected), paging (386 CR0.PG),
//! and the shared low-level accessors used by the decoder, interpreter and JIT.

use crate::cpu::{AccessKind, Error, Seg, X86};

/// Result of a translation.
enum Translate {
    /// Real/PM direct: physical address.
    Phys(u32),
    /// Page fault: page-present / page-write flags.
    PageFault {
        present: bool,
        write: bool,
        addr: u32,
    },
    /// Segment fault: general protection or SS.
    SegFault { ss: bool, addr: u32 },
}

/// Compute an effective address from a `MemRef` produced by the decoder.
/// With `a16` the 16-bit effective address wraps at 0xFFFF.
#[inline]
pub fn eff_addr(
    cpu: &X86,
    base: Option<u8>,
    index: Option<u8>,
    scale: u8,
    disp: i32,
    a16: bool,
) -> u32 {
    let mut ea: u64 = 0;
    if let Some(b) = base {
        ea += cpu.gpr[b as usize] as u64;
    }
    if let Some(i) = index {
        ea += (cpu.gpr[i as usize] as u64) * scale as u64;
    }
    ea = ea.wrapping_add(disp as i64 as u64);
    if a16 { ea as u16 as u32 } else { ea as u32 }
}

/// Translate a segment:offset access into a physical address.
/// Real mode: 20-bit wrap. Protected mode: limit check + base; paging walk.
fn translate(cpu: &X86, seg: Seg, off: u32, size: u32, kind: AccessKind) -> Translate {
    let d = &cpu.seg[seg as usize].desc;

    if cpu.cr[0] & 1 == 0 {
        // Real mode: base = selector << 4, 20-bit physical.
        let lin = (d.base as u64).wrapping_add(off as u64);
        return Translate::Phys((lin & 0xFFFFF) as u32);
    }

    // Protected mode: enforce limit on the offset.
    let limit = d.eff_limit();
    if off > limit {
        return Translate::SegFault {
            ss: seg == Seg::Ss,
            addr: off,
        };
    }
    if size > 1 && off > limit.wrapping_sub(size - 1) {
        return Translate::SegFault {
            ss: seg == Seg::Ss,
            addr: off,
        };
    }
    let lin = (d.base as u64).wrapping_add(off as u64) as u32;

    if cpu.cr[0] & (1 << 31) == 0 {
        return Translate::Phys(lin);
    }

    // ---- paging ----
    let pd_base = cpu.cr[3] & !0xFFF;
    let pde_addr = pd_base + (((lin >> 22) & 0x3FF) * 4);
    let pde = read_phys32(cpu, pde_addr);
    if pde & 1 == 0 {
        return Translate::PageFault {
            present: false,
            write: false,
            addr: lin,
        };
    }
    let pt_base = pde & !0xFFF;
    let pte_addr = pt_base + (((lin >> 12) & 0x3FF) * 4);
    let pte = read_phys32(cpu, pte_addr);
    if pte & 1 == 0 {
        return Translate::PageFault {
            present: false,
            write: false,
            addr: lin,
        };
    }
    let write = matches!(kind, AccessKind::Write | AccessKind::Rmw);
    if write && (pde & 2 == 0 || pte & 2 == 0) {
        return Translate::PageFault {
            present: true,
            write: true,
            addr: lin,
        };
    }
    if cpu.cpl() != 0 && (pde & 4 == 0 || pte & 4 == 0) {
        return Translate::PageFault {
            present: true,
            write,
            addr: lin,
        };
    }
    let phys = (pte & !0xFFF) | (lin & 0xFFF);
    Translate::Phys(phys)
}

/// Read a physical dword from the memory map (used during paging walk).
#[inline]
fn read_phys32(cpu: &X86, addr: u32) -> u32 {
    let m = &cpu.mem;
    (m.phys_read8(addr) as u32)
        | ((m.phys_read8(addr + 1) as u32) << 8)
        | ((m.phys_read8(addr + 2) as u32) << 16)
        | ((m.phys_read8(addr + 3) as u32) << 24)
}

// ---------------------------------------------------------------------------
// Instruction fetch — caller passes the *offset* from CS base (EIP).
// ---------------------------------------------------------------------------

/// Fetch the instruction byte at CS base + `off`. Real mode wraps at 1 MB.
#[inline]
pub fn fetch8(cpu: &X86, off: u32) -> Result<u8, String> {
    let phys = if cpu.cr[0] & 1 == 0 {
        let cs_lin = (cpu.seg_base(Seg::Cs) as u64).wrapping_add((off as u16) as u64);
        (cs_lin & 0xFFFFF) as u32
    } else {
        let cs_lin = (cpu.seg_base(Seg::Cs) as u64).wrapping_add(off as u64);
        // Protected mode fetch: limit-checked by CS descriptor.
        let limit = cpu.seg_limit(Seg::Cs);
        if off > limit {
            return Err(format!(
                "CS limit exceeded (EIP {off:#x} > limit {limit:#x})"
            ));
        }
        cs_lin as u32
    };
    Ok(cpu.mem.phys_read8(phys))
}

/// Fetch the instruction word at CS base + `off`.
#[inline]
pub fn fetch16(cpu: &X86, off: u32) -> Result<u16, String> {
    let lo = fetch8(cpu, off)? as u16;
    let hi = fetch8(cpu, off + 1)? as u16;
    Ok(lo | (hi << 8))
}

/// Fetch the instruction dword at CS base + `off`.
#[inline]
pub fn fetch32(cpu: &X86, off: u32) -> Result<u32, String> {
    let lo = fetch16(cpu, off)? as u32;
    let hi = fetch16(cpu, off + 2)? as u32;
    Ok(lo | (hi << 16))
}

/// Convert a 16-bit segment:offset (real mode) to a physical address.
#[inline]
pub fn real_addr(seg: u16, off: u32) -> u32 {
    ((seg as u32) << 4).wrapping_add(off) & 0xFFFFF
}

// ---------------------------------------------------------------------------
// Data accessors used by the interpreter and generated JIT code.
// ---------------------------------------------------------------------------

#[inline]
pub fn read8(cpu: &mut X86, seg: Seg, off: u32, kind: AccessKind) -> Result<u8, Error> {
    match translate(cpu, seg, off, 1, kind) {
        Translate::Phys(a) => Ok(cpu.mem.phys_read8(a)),
        t => Err(translate_err(t)),
    }
}

#[inline]
pub fn write8(cpu: &mut X86, seg: Seg, off: u32, val: u8, kind: AccessKind) -> Result<(), Error> {
    match translate(cpu, seg, off, 1, kind) {
        Translate::Phys(a) => {
            cpu.mem.phys_write8(a, val);
            Ok(())
        }
        t => Err(translate_err(t)),
    }
}

#[inline]
pub fn read16(cpu: &mut X86, seg: Seg, off: u32, kind: AccessKind) -> Result<u16, Error> {
    let lo = read8(cpu, seg, off, kind)?;
    let hi = read8(cpu, seg, off + 1, kind)?;
    Ok(lo as u16 | ((hi as u16) << 8))
}

#[inline]
pub fn write16(cpu: &mut X86, seg: Seg, off: u32, val: u16, kind: AccessKind) -> Result<(), Error> {
    write8(cpu, seg, off, val as u8, kind)?;
    write8(cpu, seg, off + 1, (val >> 8) as u8, kind)
}

#[inline]
pub fn read32(cpu: &mut X86, seg: Seg, off: u32, kind: AccessKind) -> Result<u32, Error> {
    let lo = read16(cpu, seg, off, kind)?;
    let hi = read16(cpu, seg, off + 2, kind)?;
    Ok(lo as u32 | ((hi as u32) << 16))
}

#[inline]
pub fn write32(cpu: &mut X86, seg: Seg, off: u32, val: u32, kind: AccessKind) -> Result<(), Error> {
    write16(cpu, seg, off, val as u16, kind)?;
    write16(cpu, seg, off + 2, (val >> 16) as u16, kind)
}

/// Stack read/write helpers (SS segment). `size` is 1/2/4.
#[inline]
pub fn stack_read(cpu: &mut X86, off: u32, size: u32, kind: AccessKind) -> Result<u32, Error> {
    match size {
        1 => read8(cpu, Seg::Ss, off, kind).map(|v| v as u32),
        2 => read16(cpu, Seg::Ss, off, kind).map(|v| v as u32),
        _ => read32(cpu, Seg::Ss, off, kind),
    }
}

#[inline]
pub fn stack_write(
    cpu: &mut X86,
    off: u32,
    size: u32,
    val: u32,
    kind: AccessKind,
) -> Result<(), Error> {
    match size {
        1 => write8(cpu, Seg::Ss, off, val as u8, kind),
        2 => write16(cpu, Seg::Ss, off, val as u16, kind),
        _ => write32(cpu, Seg::Ss, off, val, kind),
    }
}

/// Translate to physical for a caller that already knows the access (JIT fast
/// path). Returns the physical address on success, Err(Error) on fault.
#[inline]
pub fn translate_for_access(
    cpu: &X86,
    seg: Seg,
    off: u32,
    size: u32,
    kind: AccessKind,
) -> Result<u32, Error> {
    match translate(cpu, seg, off, size, kind) {
        Translate::Phys(p) => Ok(p),
        t => Err(translate_err(t)),
    }
}

fn translate_err(t: Translate) -> Error {
    match t {
        Translate::Phys(_) => unreachable!(),
        Translate::PageFault { addr, .. } => Error::Internal(format!("page fault at {addr:#x}")),
        Translate::SegFault { addr, .. } => Error::Internal(format!("segment fault at {addr:#x}")),
    }
}
