//! CPU state: registers, descriptors, segments, TSS, and the shared `X86`
//! structure that both the interpreter and the JIT operate on.
//!
//! This crate emulates a 386-class CPU (real mode + 32-bit protected mode,
//! paging). It is *instruction-set* software only: firmware (Bochs BIOS /
//! VGABIOS) and devices live in the `vibex86` crate.

use crate::mem::Mem;

/// General-purpose register identifiers (i386 order: EAX, ECX, EDX, EBX, ESP,
/// EBP, ESI, EDI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reg {
    Eax = 0,
    Ecx = 1,
    Edx = 2,
    Ebx = 3,
    Esp = 4,
    Ebp = 5,
    Esi = 6,
    Edi = 7,
}

impl Reg {
    #[inline]
    pub fn from_idx(i: u8) -> Reg {
        match i & 7 {
            0 => Reg::Eax,
            1 => Reg::Ecx,
            2 => Reg::Edx,
            3 => Reg::Ebx,
            4 => Reg::Esp,
            5 => Reg::Ebp,
            6 => Reg::Esi,
            _ => Reg::Edi,
        }
    }
}

/// Segment register names, canonical order ES CS SS DS FS GS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Seg {
    Es = 0,
    Cs = 1,
    Ss = 2,
    Ds = 3,
    Fs = 4,
    Gs = 5,
}

impl Seg {
    #[inline]
    pub fn from_idx(i: u8) -> Seg {
        match i & 7 {
            0 => Seg::Es,
            1 => Seg::Cs,
            2 => Seg::Ss,
            3 => Seg::Ds,
            4 => Seg::Fs,
            _ => Seg::Gs,
        }
    }
}

/// CPU execution mode (for the public API / diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Real,
    Protected16,
    Protected32,
}

/// Descriptor privilege summary (diagnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescProt {
    No,
    Yes,
}

/// Descriptor kind (diagnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescType {
    Code,
    Data,
    System,
}

/// Outcome of a reset/triple-fault trip (diagnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trip {
    None,
    TripleFault,
    Reset,
}

/// Flags bits in EFLAGS.
pub mod flag {
    pub const CF: u32 = 1 << 0;
    pub const PF: u32 = 1 << 2;
    pub const AF: u32 = 1 << 4;
    pub const ZF: u32 = 1 << 6;
    pub const SF: u32 = 1 << 7;
    pub const TF: u32 = 1 << 8;
    pub const IF: u32 = 1 << 9;
    pub const DF: u32 = 1 << 10;
    pub const OF: u32 = 1 << 11;
    pub const IOPL: u32 = 3 << 12;
    pub const NT: u32 = 1 << 14;
    pub const RF: u32 = 1 << 16;
    pub const VM: u32 = 1 << 17;
    pub const AC: u32 = 1 << 18;
}

/// Cached segment / descriptor-register value: selector + full descriptor.
#[derive(Debug, Clone, Copy)]
pub struct Desc {
    pub base: u32,
    pub limit: u32,
    pub g: bool,
    pub db: bool,
    pub p: bool,
    /// Descriptor type field (bits 3..=7 of the access byte).
    pub typ: u8,
    pub dpl: u8,
    #[allow(dead_code)]
    pub avl: bool,
}

impl Desc {
    pub const NULL: Desc = Desc {
        base: 0,
        limit: 0,
        g: false,
        db: false,
        p: false,
        typ: 0,
        dpl: 0,
        avl: false,
    };

    #[inline]
    pub fn real(sel: u16) -> Desc {
        Desc {
            base: (sel as u32) << 4,
            limit: 0xFFFF,
            g: false,
            db: false,
            p: true,
            typ: 0x93,
            dpl: 0,
            avl: false,
        }
    }

    /// Effective limit remembering the 386 quirk: G-bit descriptors of limit
    /// 0xFFFFF have effective limit 0xFFFF_FFFF.
    #[inline]
    pub fn eff_limit(&self) -> u32 {
        if self.g {
            (self.limit << 12) | 0xFFF
        } else {
            self.limit
        }
    }

    #[inline]
    pub fn is_code(&self) -> bool {
        self.typ & 0x8 != 0
    }

    #[inline]
    pub fn conforming(&self) -> bool {
        self.typ & 0x4 != 0
    }
}

/// A segment register's cached selector + descriptor.
#[derive(Debug, Clone, Copy)]
pub struct SegVal {
    pub sel: u16,
    pub desc: Desc,
}

impl SegVal {
    pub const REAL_CS: SegVal = SegVal {
        sel: 0xF000,
        desc: Desc {
            base: 0xF0000,
            limit: 0xFFFF,
            g: false,
            db: false,
            p: true,
            typ: 0x9B,
            dpl: 0,
            avl: false,
        },
    };
    pub const REAL_DS: SegVal = SegVal {
        sel: 0,
        desc: Desc {
            base: 0,
            limit: 0xFFFF,
            g: false,
            db: false,
            p: true,
            typ: 0x93,
            dpl: 0,
            avl: false,
        },
    };

    #[inline]
    pub fn real(sel: u16) -> SegVal {
        SegVal {
            sel,
            desc: Desc::real(sel),
        }
    }

    #[inline]
    pub fn base(&self) -> u32 {
        self.desc.base
    }

    #[inline]
    pub fn limit(&self) -> u32 {
        self.desc.eff_limit()
    }
}

/// GDTR/IDTR value.
#[derive(Debug, Clone, Copy)]
pub struct DescReg {
    pub base: u32,
    pub limit: u16,
}

impl DescReg {
    pub const ZERO: DescReg = DescReg { base: 0, limit: 0 };
}

/// Subset of the 386 TSS we maintain.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tss {
    pub ss0: u16,
    pub esp0: u32,
    #[allow(dead_code)]
    pub ss1: u16,
    #[allow(dead_code)]
    pub esp1: u32,
    #[allow(dead_code)]
    pub ss2: u16,
    #[allow(dead_code)]
    pub esp2: u32,
    pub cr3: u32,
    #[allow(dead_code)]
    pub iopb: u16,
}

/// Why execution stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    TripleFault,
    Reset,
    Halt,
    Unsupported(String),
    BusFault(u32),
    Internal(String),
}

/// Result of `X86::step`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOut {
    Ok,
    Interrupt,
    Error(Error),
}

/// Kind of memory access, used for fault reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
    Fetch,
    Rmw,
}

/// CPU core: registers + memory + host devices, driven either by the
/// interpreter or the Cranelift JIT.
pub struct X86 {
    pub gpr: [u32; 8],
    pub eip: u32,
    pub eflags: u32,
    pub seg: [SegVal; 6],
    pub gdtr: DescReg,
    pub idtr: DescReg,
    pub ldtr: SegVal,
    pub tr: SegVal,
    /// CR0 (PE/PG bits matter), CR1…CR3.
    pub cr: [u32; 4],
    pub tss: Tss,
    pub pending_irq: Option<u8>,
    pub mem: Mem,
    pub cycles: u64,
    pub halted: bool,
    /// Force interpreter mode (used by tests/diagnostics).
    pub force_interp: bool,
}

impl X86 {
    pub fn new() -> Self {
        let mut mem = Mem::new();
        mem.install_default_ram();
        X86 {
            gpr: [0; 8],
            eip: 0xFFF0,
            eflags: 0x2,
            seg: [
                SegVal::REAL_DS,
                SegVal::REAL_CS,
                SegVal::REAL_DS,
                SegVal::REAL_DS,
                SegVal::REAL_DS,
                SegVal::REAL_DS,
            ],
            gdtr: DescReg::ZERO,
            idtr: DescReg::ZERO,
            ldtr: SegVal::real(0),
            tr: SegVal::real(0),
            cr: [0; 4],
            tss: Tss::default(),
            pending_irq: None,
            mem,
            cycles: 0,
            halted: false,
            force_interp: false,
        }
    }

    /// Reset to the i386 reset state (CS=0xF000, EIP=0xFFF0, real mode).
    pub fn reset(&mut self) {
        self.gpr = [0; 8];
        self.eip = 0xFFF0;
        self.eflags = 0x2;
        self.seg = [
            SegVal::REAL_DS,
            SegVal::REAL_CS,
            SegVal::REAL_DS,
            SegVal::REAL_DS,
            SegVal::REAL_DS,
            SegVal::REAL_DS,
        ];
        self.gdtr = DescReg::ZERO;
        self.idtr = DescReg::ZERO;
        self.ldtr = SegVal::real(0);
        self.tr = SegVal::real(0);
        self.cr = [0; 4];
        self.tss = Tss::default();
        self.pending_irq = None;
        self.cycles = 0;
        self.halted = false;
    }

    // ---- register accessors ----

    #[inline]
    pub fn reg32(&self, r: i8) -> u32 {
        self.gpr[(r & 7) as usize]
    }
    #[inline]
    pub fn set_reg32(&mut self, r: i8, v: u32) {
        self.gpr[(r & 7) as usize] = v;
    }
    #[inline]
    pub fn reg16(&self, r: i8) -> u16 {
        self.gpr[(r & 7) as usize] as u16
    }
    #[inline]
    pub fn set_reg16(&mut self, r: i8, v: u16) {
        let i = (r & 7) as usize;
        self.gpr[i] = (self.gpr[i] & 0xFFFF_0000) | v as u32;
    }
    #[inline]
    pub fn reg8(&self, r: i8) -> u8 {
        if r & 4 != 0 {
            let i = (r & 3) as usize;
            (self.gpr[i] >> 8) as u8
        } else {
            let i = (r & 3) as usize;
            self.gpr[i] as u8
        }
    }
    #[inline]
    pub fn set_reg8(&mut self, r: i8, v: u8) {
        let v = v as u32;
        if r & 4 != 0 {
            let i = (r & 3) as usize;
            self.gpr[i] = (self.gpr[i] & 0xFFFF_00FF) | (v << 8);
        } else {
            let i = (r & 3) as usize;
            self.gpr[i] = (self.gpr[i] & 0xFFFF_FF00) | v;
        }
    }

    #[inline]
    pub fn seg_base(&self, s: Seg) -> u32 {
        self.seg[s as usize].base()
    }
    #[inline]
    pub fn seg_limit(&self, s: Seg) -> u32 {
        self.seg[s as usize].limit()
    }

    /// Current CPU mode.
    #[inline]
    pub fn mode(&self) -> Mode {
        if self.cr[0] & 1 == 0 {
            Mode::Real
        } else if self.seg[Seg::Cs as usize].desc.db {
            Mode::Protected32
        } else {
            Mode::Protected16
        }
    }

    /// Current privilege level (0 in real mode, else CS.DPL).
    #[inline]
    pub fn cpl(&self) -> u8 {
        if self.cr[0] & 1 == 0 {
            0
        } else {
            self.seg[Seg::Cs as usize].desc.dpl
        }
    }

    /// Execute a single instruction (decode + interpreter semantics).
    pub fn step(&mut self) -> StepOut {
        crate::interp::step(self)
    }

    /// Run until `Error` (halt/triple fault).
    pub fn run(&mut self) -> Result<(), Error> {
        loop {
            match self.step() {
                StepOut::Ok | StepOut::Interrupt => {}
                StepOut::Error(e) => return Err(e),
            }
        }
    }

    /// Raise (possibly nested) exception `vector` immediately.
    pub fn interrupt(&mut self, vector: u8, has_error: bool, error: u32) -> StepOut {
        crate::interp::dispatch_interrupt(self, vector, has_error, error)
    }
}
