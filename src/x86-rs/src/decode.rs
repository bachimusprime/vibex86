//! The x86 decoder: shared by the interpreter and the JIT.
//!
//! Produces a `Decoded` instruction from the byte stream at CS:EIP without
//! modifying CPU state (other than nothing — fetch is a pure read). Both
//! engines consume the same representation, so their instruction sets match by
//! construction.

use crate::{
    X86,
    cpu::{Mode, Seg},
    mmu,
};

/// Operand / address size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bits {
    B8,
    B16,
    B32,
}

impl Bits {
    #[inline]
    pub fn is_8(&self) -> bool {
        matches!(self, Bits::B8)
    }
    #[inline]
    pub fn bytes(self) -> u32 {
        match self {
            Bits::B8 => 1,
            Bits::B16 => 2,
            Bits::B32 => 4,
        }
    }
}

/// Memory effective address (ModRM memory operand).
#[derive(Debug, Clone, Copy)]
pub struct MemRef {
    pub seg: Seg,
    pub base: Option<u8>,  // reg index 0..7
    pub index: Option<u8>, // reg index 0..7
    pub scale: u8,         // 1/2/4/8
    pub disp: i32,         // sign-extended displacement
    pub a16: bool,         // 16-bit effective address
}

/// An operand slot.
#[derive(Debug, Clone, Copy)]
pub enum Opnd {
    None,
    /// Register. Values 4..7 for 8-bit ops decode as AH/BH/CH/DH (high byte).
    Reg(u8, Bits),
    /// Memory operand.
    Mem(MemRef, Bits),
    /// Zero-extended immediate.
    Imm(u32),
    /// Sign-extended immediate (imm8 -> 16/32).
    ImmSext(u32),
    /// Far pointer literal (call/jmp segment:offset).
    FarPtr {
        sel: u16,
        off: u32,
    },
    /// Relative displacement — target = EIP after instruction + disp.
    Rel {
        disp: i32,
    },
    /// Segment register (ES/CS/SS/DS/FS/GS) as data operand.
    Sreg(Seg),
    /// Implicit accumulator (AL/AX/EAX).
    Acc(Bits),
    /// Implicit port register (DX).
    Dx,
    /// Immediate port number.
    Port(u16),
    /// CL (shift count register).
    Cl,
}

/// REP prefix kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rep {
    None,
    Z,  // REP / REPE
    NZ, // REPNE
}

/// Condition codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cond {
    O,
    No,
    B,
    Ae,
    E,
    Ne,
    Be,
    A,
    S,
    Ns,
    P,
    Np,
    L,
    Ge,
    Le,
    G,
}

impl Cond {
    pub fn from_opcode(b: u8) -> Cond {
        match b & 0xF {
            0 => Cond::O,
            1 => Cond::No,
            2 => Cond::B,
            3 => Cond::Ae,
            4 => Cond::E,
            5 => Cond::Ne,
            6 => Cond::Be,
            7 => Cond::A,
            8 => Cond::S,
            9 => Cond::Ns,
            10 => Cond::P,
            11 => Cond::Np,
            12 => Cond::L,
            13 => Cond::Ge,
            14 => Cond::Le,
            _ => Cond::G,
        }
    }
}

/// ALU operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AluOp {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
}

/// Shift/rotate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftKind {
    Rol,
    Ror,
    Rcl,
    Rcr,
    Shl,
    Shr,
    Sar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShiftBy {
    One,
    Cl,
    Imm(u8),
}

/// Bit test/scan op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitOp {
    Bt,
    Bts,
    Btr,
    Btc,
    Bsf,
    Bsr,
}

/// Instruction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    // ---- ALU ----
    Alu(AluOp), // ops[0]=dst=src operand, ops[1]=other
    Inc,
    Dec,
    Neg,
    Not, // ops[0]=rm
    Mul,
    Imul,
    Div,
    Idiv,                      // ops[0]=rm; implicit AL/AX/EAX
    Imul1,                     // ops[0]=dst reg, ops[1]=src rm, ops[2]=imm or None
    Shift(ShiftKind, ShiftBy), // ops[0]=rm
    Dshift(bool, ShiftBy),     // SHLD/SHRD; ops[0]=dst rm, ops[1]=src reg
    // ---- bit ----
    Test,       // ops[0], ops[1]
    Bit(BitOp), // ops[0]=rm, ops[1]=imm/reg
    // ---- data movement ----
    Mov,
    Movzx(bool), // true if source is 8-bit
    Movsx(bool), // true if source is 8-bit
    Lea,
    Xchg,
    Cmov(Cond),
    Setcc(Cond),
    Push,
    Pop, // ops[0]=rm/reg/imm/sreg
    PushF,
    PopF,
    PushA,
    PopA,
    Call { far: bool },
    Jump { far: bool },
    Ret { far: bool, imm: Option<u16> },
    Jcc(Cond),        // ops[0]=Rel
    Loop(Option<u8>), // None=LOOP, Some(0xE0)=LOOPNE, Some(0xE1)=LOOPE
    Jcxz,
    // ---- strings ----
    Movs(Bits),
    Stos(Bits),
    Lods(Bits),
    Cmps(Bits),
    Scas(Bits),
    Ins(Bits),
    Outs(Bits),
    Xlat,
    // ---- io ----
    In,
    Out,
    // ---- interrupts / system ----
    Int(u8),
    Int3,
    Into,
    Iret,
    Hlt,
    Cli,
    Sti,
    Cld,
    Std,
    Clc,
    Stc,
    Cmc,
    Salc,
    // ---- segments & descriptors ----
    MovSreg2Rm, // MOV r/m16, Sreg
    MovRm2Sreg, // MOV Sreg, r/m16
    PopSeg(Seg),
    Lss,
    Lds,
    Les,
    Lfs,
    Lgs,
    Lar,
    Lsl,
    VErr(bool), // VERR / VERW
    Arpl,
    Lgdt,
    Lidt,
    Sgdt,
    Sidt,
    Lldt,
    Sldt,
    Ltr,
    Str,
    Smsw,
    Lmsw,
    Clts,
    MovCr,
    MovDr,
    // ---- 386 misc ----
    Bswap,
    CmpXchg,
    Xadd,
    Cpuid,
    Rdtsc,
    Rdmsr,
    Wrmsr,
    Wbinvd,
    Invd,
    Invlpg,
    // ---- misc ----
    Nop,
    Wait,
    Daa,
    Das,
    Aaa,
    Aas,
    Aam,
    Aad,
    Cbw,
    Cwd,
    Sahf,
    Lahf,
    Enter(u16, u8),
    Leave,
    // ---- anything else ----
    Unsupported(u8),
    TwoByte(u8),
}

/// A fully decoded instruction.
#[derive(Debug, Clone, Copy)]
pub struct Decoded {
    pub op: Op,
    pub len: u8,
    pub ops: [Opnd; 4],
    pub rep: Rep,
    pub lock: bool,
    /// Operand-size is 16 for this instruction (else 32).
    pub o16: bool,
    /// Address-size is 16 for this instruction (else 32).
    pub a16: bool,
    /// Segment override applied to the memory operand.
    pub seg_override: Option<Seg>,
}

impl Decoded {
    /// Default operand size (B16 with 0x66, B32 with 32-bit opsize).
    #[inline]
    pub fn size(&self) -> Bits {
        if self.o16 { Bits::B16 } else { Bits::B32 }
    }

    /// Unconditional (or conditional) relative target EIP computed from the
    /// Rel operand: EIP after the instruction + disp.
    #[inline]
    pub fn rel_target(&self, next_eip: u32) -> u32 {
        match self.ops[0] {
            Opnd::Rel { disp } => (next_eip as i64 + disp as i64) as u32,
            _ => next_eip,
        }
    }
}

/// Fetch and decode one instruction at CS:EIP. Returns the decoded instruction
/// and its length. Does *not* advance EIP.
pub fn fetch(cpu: &X86) -> Result<Decoded, String> {
    let eip = cpu.eip;
    let mut dec = Decoder { cpu, pos: 0, eip };
    dec.run()
}

struct Decoder<'a> {
    cpu: &'a X86,
    pos: u32,
    eip: u32,
}

impl<'a> Decoder<'a> {
    #[inline]
    fn raw(&mut self) -> Result<u8, String> {
        let b = mmu::fetch8(self.cpu, self.eip + self.pos)?;
        self.pos += 1;
        Ok(b)
    }

    #[inline]
    fn raw16(&mut self) -> Result<u16, String> {
        let lo = self.raw()? as u16;
        let hi = self.raw()? as u16;
        Ok(lo | (hi << 8))
    }

    #[inline]
    fn raw32(&mut self) -> Result<u32, String> {
        let lo = self.raw16()? as u32;
        let hi = self.raw16()? as u32;
        Ok(lo | (hi << 16))
    }

    fn reg_opnd(&self, reg: u8, bits: Bits) -> Opnd {
        Opnd::Reg(reg, bits)
    }

    /// Decode modrm for a memory operand; returns (mod, memref|reg-id, reg field).
    fn rm(
        &mut self,
        modrm: u8,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<(u8, Opnd, u8), String> {
        let m = modrm >> 6;
        let reg = (modrm >> 3) & 7;
        let rm = modrm & 7;
        if m == 3 {
            return Ok((
                m,
                Opnd::Reg(rm, if o16 { Bits::B16 } else { Bits::B32 }),
                reg,
            ));
        }
        let (base, index, scale, disp) = if a16 {
            let direct = m == 0 && rm == 6;
            let (b, idx) = if direct {
                (None, None)
            } else {
                match rm {
                    0 => (Some(3), Some(6)), // BX+SI
                    1 => (Some(3), Some(7)), // BX+DI
                    2 => (Some(5), Some(6)), // BP+SI
                    3 => (Some(5), Some(7)), // BP+DI
                    4 => (Some(6), None),    // SI
                    5 => (Some(7), None),    // DI
                    6 => (Some(5), None),    // BP
                    _ => (Some(3), None),    // BX
                }
            };
            let disp = match m {
                0 => {
                    if rm == 6 {
                        self.raw16()? as i32
                    } else {
                        0
                    }
                }
                1 => self.raw()? as i8 as i32,
                _ => self.raw16()? as i16 as i32,
            };
            (b, idx, 1u8, disp)
        } else {
            let mut base = None;
            let mut index = None;
            let mut scale = 1u8;
            let mut disp = 0i32;
            if rm == 4 {
                let sib = self.raw()?;
                scale = match sib >> 6 {
                    0 => 1,
                    1 => 2,
                    2 => 4,
                    _ => 8,
                };
                let idx = (sib >> 3) & 7;
                let base_idx = sib & 7;
                if idx != 4 {
                    index = Some(idx);
                }
                match m {
                    0 => {
                        if base_idx == 5 {
                            disp = self.raw32()? as i32;
                        } else {
                            base = Some(base_idx);
                        }
                    }
                    1 => {
                        base = Some(base_idx);
                        disp = self.raw()? as i8 as i32;
                    }
                    _ => {
                        base = Some(base_idx);
                        disp = self.raw32()? as i32;
                    }
                }
            } else {
                match m {
                    0 => {
                        if rm == 5 {
                            disp = self.raw32()? as i32;
                        } else {
                            base = Some(rm);
                        }
                    }
                    1 => {
                        base = Some(rm);
                        disp = self.raw()? as i8 as i32;
                    }
                    _ => {
                        base = Some(rm);
                        disp = self.raw32()? as i32;
                    }
                }
            }
            (base, index, scale, disp)
        };
        let seg = match seg_ov {
            Some(s) => s,
            None => {
                // SS default with BP base in 16-bit, or EBP base in 32-bit.
                let ss_base = if a16 {
                    base == Some(5)
                } else {
                    base == Some(5)
                };
                if ss_base { Seg::Ss } else { Seg::Ds }
            }
        };
        Ok((
            m,
            Opnd::Mem(
                MemRef {
                    seg,
                    base,
                    index,
                    scale,
                    disp,
                    a16,
                },
                if o16 { Bits::B16 } else { Bits::B32 },
            ),
            reg,
        ))
    }

    /// Like `rm` but with an explicit operand size (8-bit variants etc).
    fn rm8(
        &mut self,
        modrm: u8,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<(u8, Opnd, u8), String> {
        let (m, opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
        let opnd = match opnd {
            Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B8),
            Opnd::Mem(r, _) => Opnd::Mem(r, Bits::B8),
            other => other,
        };
        Ok((m, opnd, reg))
    }

    fn run(&mut self) -> Result<Decoded, String> {
        let default_16 = !matches!(self.cpu.mode(), Mode::Protected32);
        let mut o16 = default_16;
        let mut a16 = default_16;
        let mut lock = false;
        let mut rep = Rep::None;
        let mut seg_ov: Option<Seg> = None;

        loop {
            let b = self.raw()?;
            match b {
                0x66 => o16 = !default_16,
                0x67 => a16 = !default_16,
                0xF0 => lock = true,
                0xF2 => rep = Rep::NZ,
                0xF3 => rep = Rep::Z,
                0x26 => seg_ov = Some(Seg::Es),
                0x2E => seg_ov = Some(Seg::Cs),
                0x36 => seg_ov = Some(Seg::Ss),
                0x3E => seg_ov = Some(Seg::Ds),
                0x64 => seg_ov = Some(Seg::Fs),
                0x65 => seg_ov = Some(Seg::Gs),
                _ => {
                    self.pos -= 1;
                    break;
                }
            }
            if self.pos >= 15 {
                return Err("instruction exceeds 15-byte limit".into());
            }
        }

        let b0 = self.raw()?;
        let mut d = self.decode_opcode(b0, o16, a16, seg_ov)?;
        d.rep = rep;
        d.lock = lock;
        Ok(d)
    }

    fn fin(&self, op: Op, ops: [Opnd; 4], o16: bool, a16: bool, seg_ov: Option<Seg>) -> Decoded {
        Decoded {
            op,
            len: self.pos as u8,
            ops,
            rep: Rep::None,
            lock: false,
            o16,
            a16,
            seg_override: seg_ov,
        }
    }

    fn decode_opcode(
        &mut self,
        b: u8,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<Decoded, String> {
        let size = if o16 { Bits::B16 } else { Bits::B32 };
        let acc = |bits: Bits| Opnd::Acc(bits);

        match b {
            // ---- ALU group ----
            0x00..=0x05
            | 0x08..=0x0D
            | 0x10..=0x15
            | 0x18..=0x1D
            | 0x20..=0x25
            | 0x28..=0x2D
            | 0x30..=0x35
            | 0x38..=0x3D => {
                let op = match b / 8 {
                    0 => crate::decode::AluOp::Add,
                    1 => crate::decode::AluOp::Or,
                    2 => crate::decode::AluOp::Adc,
                    3 => crate::decode::AluOp::Sbb,
                    4 => crate::decode::AluOp::And,
                    5 => crate::decode::AluOp::Sub,
                    6 => crate::decode::AluOp::Xor,
                    _ => crate::decode::AluOp::Cmp,
                };
                let sz8 = b % 2 == 0;
                let sz = if sz8 { Bits::B8 } else { size };
                match b % 8 {
                    0 | 1 => {
                        // ALU r/m, r
                        let modrm = self.raw()?;
                        let (_, rm_opnd, reg) = if sz8 {
                            self.rm8(modrm, o16, a16, seg_ov)?
                        } else {
                            self.rm(modrm, o16, a16, seg_ov)?
                        };
                        Ok(self.fin(
                            Op::Alu(op),
                            [rm_opnd, Opnd::Reg(reg, sz), Opnd::None, Opnd::None],
                            o16,
                            a16,
                            seg_ov,
                        ))
                    }
                    2 | 3 => {
                        // ALU r, r/m
                        let modrm = self.raw()?;
                        let (_, rm_opnd, reg) = if sz8 {
                            self.rm8(modrm, o16, a16, seg_ov)?
                        } else {
                            self.rm(modrm, o16, a16, seg_ov)?
                        };
                        Ok(self.fin(
                            Op::Alu(op),
                            [Opnd::Reg(reg, sz), rm_opnd, Opnd::None, Opnd::None],
                            o16,
                            a16,
                            seg_ov,
                        ))
                    }
                    _ => {
                        // ALU acc, imm
                        let imm = if sz8 {
                            self.raw()? as u32
                        } else if o16 {
                            self.raw16()? as u32
                        } else {
                            self.raw32()?
                        };
                        Ok(self.fin(
                            Op::Alu(op),
                            [acc(sz), Opnd::Imm(imm), Opnd::None, Opnd::None],
                            o16,
                            a16,
                            seg_ov,
                        ))
                    }
                }
            }

            // ---- segment pushes/pops and misc one-byte ----
            0x06 => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Es), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0x07 => Ok(self.fin(Op::PopSeg(Seg::Es), [Opnd::None; 4], o16, a16, seg_ov)),
            0x0E => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Cs), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0x16 => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Ss), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0x17 => Ok(self.fin(Op::PopSeg(Seg::Ss), [Opnd::None; 4], o16, a16, seg_ov)),
            0x1E => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Ds), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0x1F => Ok(self.fin(Op::PopSeg(Seg::Ds), [Opnd::None; 4], o16, a16, seg_ov)),
            0x0F => return self.decode_two_byte(o16, a16, seg_ov),
            0x27 => Ok(self.fin(Op::Daa, [Opnd::None; 4], o16, a16, seg_ov)),
            0x2F => Ok(self.fin(Op::Das, [Opnd::None; 4], o16, a16, seg_ov)),
            0x37 => Ok(self.fin(Op::Aaa, [Opnd::None; 4], o16, a16, seg_ov)),
            0x3F => Ok(self.fin(Op::Aas, [Opnd::None; 4], o16, a16, seg_ov)),

            // ---- INC/DEC reg ----
            0x40..=0x47 => {
                let r = b - 0x40;
                Ok(self.fin(
                    Op::Inc,
                    [Opnd::Reg(r, size), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x48..=0x4F => {
                let r = b - 0x48;
                Ok(self.fin(
                    Op::Dec,
                    [Opnd::Reg(r, size), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x50..=0x57 => {
                let r = b - 0x50;
                Ok(self.fin(
                    Op::Push,
                    [Opnd::Reg(r, size), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x58..=0x5F => {
                let r = b - 0x58;
                Ok(self.fin(
                    Op::Pop,
                    [Opnd::Reg(r, size), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x60 => Ok(self.fin(Op::PushA, [Opnd::None; 4], o16, a16, seg_ov)),
            0x61 => Ok(self.fin(Op::PopA, [Opnd::None; 4], o16, a16, seg_ov)),
            0x62 => Err("BOUND unsupported".into()),
            0x63 => {
                let modrm = self.raw()?;
                let (m, opnd, reg) = self.rm(modrm, false, a16, seg_ov)?;
                if m == 3 {
                    // ARPL reg, reg
                    Ok(self.fin(
                        Op::Arpl,
                        [opnd, Opnd::Reg(reg, Bits::B16), Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                } else {
                    Ok(self.fin(
                        Op::Arpl,
                        [opnd, Opnd::Reg(reg, Bits::B16), Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                }
            }
            0x68 => {
                let imm = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                Ok(self.fin(
                    Op::Push,
                    [Opnd::Imm(imm), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x69 => {
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                let imm = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                Ok(self.fin(
                    Op::Imul1,
                    [Opnd::Reg(reg, size), src, Opnd::Imm(imm), Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x6A => {
                let imm = self.raw()? as i8 as i32 as u32;
                Ok(self.fin(
                    Op::Push,
                    [Opnd::ImmSext(imm), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x6B => {
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                let imm = self.raw()? as i8 as i32 as u32;
                Ok(self.fin(
                    Op::Imul1,
                    [Opnd::Reg(reg, size), src, Opnd::ImmSext(imm), Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x6C => Ok(self.fin(Op::Ins(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0x6D => Ok(self.fin(Op::Ins(size), [Opnd::None; 4], o16, a16, seg_ov)),
            0x6E => Ok(self.fin(Op::Outs(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0x6F => Ok(self.fin(Op::Outs(size), [Opnd::None; 4], o16, a16, seg_ov)),

            // ---- short Jcc ----
            0x70..=0x7F => {
                let c = Cond::from_opcode(b);
                let disp = self.raw()? as i8 as i32;
                Ok(self.fin(
                    Op::Jcc(c),
                    [Opnd::Rel { disp }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }

            // ---- group 1 (ALU r/m, imm) ----
            0x80 | 0x82 => self.group1(b, o16, a16, seg_ov, 8),
            0x81 => self.group1(b, o16, a16, seg_ov, 0),
            0x83 => self.group1(b, o16, a16, seg_ov, 0x80),

            0x84 => {
                let modrm = self.raw()?;
                let (_, a, reg) = self.rm8(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Test,
                    [a, Opnd::Reg(reg, Bits::B8), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x85 => {
                let modrm = self.raw()?;
                let (_, a, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Test,
                    [a, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x86 | 0x87 => {
                let modrm = self.raw()?;
                let sz8 = b == 0x86;
                let (_, a, reg) = if sz8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let sz = if sz8 { Bits::B8 } else { size };
                Ok(self.fin(
                    Op::Xchg,
                    [a, Opnd::Reg(reg, sz), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x88..=0x8B => {
                let modrm = self.raw()?;
                let sz8 = b == 0x88 || b == 0x8A;
                let to_mem = b == 0x88 || b == 0x89;
                let (_, a, reg) = if sz8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let (rm_opnd, reg_opnd) = (a, Opnd::Reg(reg, if sz8 { Bits::B8 } else { size }));
                if to_mem {
                    Ok(self.fin(
                        Op::Mov,
                        [rm_opnd, reg_opnd, Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                } else {
                    Ok(self.fin(
                        Op::Mov,
                        [reg_opnd, rm_opnd, Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                }
            }
            0x8C => {
                // MOV r/m16, Sreg
                let modrm = self.raw()?;
                let (_, opnd, reg) = self.rm(modrm, false, a16, seg_ov)?;
                let opnd = match opnd {
                    Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B16),
                    Opnd::Mem(m, _) => Opnd::Mem(m, Bits::B16),
                    other => other,
                };
                Ok(self.fin(
                    Op::MovSreg2Rm,
                    [opnd, Opnd::Sreg(Seg::from_idx(reg)), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x8D => {
                let modrm = self.raw()?;
                let (m, opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                match opnd {
                    Opnd::Mem(..) => Ok(self.fin(
                        Op::Lea,
                        [Opnd::Reg(reg, size), opnd, Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    )),
                    _ => {
                        let _ = m;
                        Err("LEA with register operand".into())
                    }
                }
            }
            0x8E => {
                // MOV Sreg, r/m16
                let modrm = self.raw()?;
                let (_, opnd, reg) = self.rm(modrm, false, a16, seg_ov)?;
                let opnd = match opnd {
                    Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B16),
                    Opnd::Mem(m, _) => Opnd::Mem(m, Bits::B16),
                    other => other,
                };
                let sreg = Seg::from_idx(reg);
                if sreg == Seg::Cs {
                    return Err("MOV CS,r/m unsupported".into());
                }
                Ok(self.fin(
                    Op::MovRm2Sreg,
                    [Opnd::Sreg(sreg), opnd, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x8F => {
                let modrm = self.raw()?;
                let (_, opnd, _) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Pop,
                    [opnd, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x90 => Ok(self.fin(Op::Nop, [Opnd::None; 4], o16, a16, seg_ov)),
            0x91..=0x97 => {
                let r = b - 0x91;
                Ok(self.fin(
                    Op::Xchg,
                    [acc(size), Opnd::Reg(r, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x98 => Ok(self.fin(Op::Cbw, [Opnd::None; 4], o16, a16, seg_ov)),
            0x99 => Ok(self.fin(Op::Cwd, [Opnd::None; 4], o16, a16, seg_ov)),
            0x9A => {
                let off = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                let sel = self.raw16()?;
                Ok(self.fin(
                    Op::Call { far: true },
                    [
                        Opnd::FarPtr { sel, off },
                        Opnd::None,
                        Opnd::None,
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x9B => Ok(self.fin(Op::Wait, [Opnd::None; 4], o16, a16, seg_ov)),
            0x9C => Ok(self.fin(Op::PushF, [Opnd::None; 4], o16, a16, seg_ov)),
            0x9D => Ok(self.fin(Op::PopF, [Opnd::None; 4], o16, a16, seg_ov)),
            0x9E => Ok(self.fin(Op::Sahf, [Opnd::None; 4], o16, a16, seg_ov)),
            0x9F => Ok(self.fin(Op::Lahf, [Opnd::None; 4], o16, a16, seg_ov)),

            // ---- moffs ----
            0xA0..=0xA3 => {
                let sz8 = b == 0xA0 || b == 0xA2;
                let sz = if sz8 { Bits::B8 } else { size };
                let off = if a16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                let mem = Opnd::Mem(
                    MemRef {
                        seg: seg_ov.unwrap_or(Seg::Ds),
                        base: None,
                        index: None,
                        scale: 1,
                        disp: off as i32,
                        a16,
                    },
                    sz,
                );
                let to_mem = b == 0xA2 || b == 0xA3;
                if to_mem {
                    Ok(self.fin(
                        Op::Mov,
                        [mem, acc(sz), Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                } else {
                    Ok(self.fin(
                        Op::Mov,
                        [acc(sz), mem, Opnd::None, Opnd::None],
                        o16,
                        a16,
                        seg_ov,
                    ))
                }
            }

            // ---- string ops ----
            0xA4 => Ok(self.fin(Op::Movs(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0xA5 => Ok(self.fin(Op::Movs(size), [Opnd::None; 4], o16, a16, seg_ov)),
            0xA6 => Ok(self.fin(Op::Cmps(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0xA7 => Ok(self.fin(Op::Cmps(size), [Opnd::None; 4], o16, a16, seg_ov)),
            0xA8 => {
                let imm = self.raw()? as u32;
                Ok(self.fin(
                    Op::Test,
                    [acc(Bits::B8), Opnd::Imm(imm), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xA9 => {
                let imm = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                Ok(self.fin(
                    Op::Test,
                    [acc(size), Opnd::Imm(imm), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xAA => Ok(self.fin(Op::Stos(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAB => Ok(self.fin(Op::Stos(size), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAC => Ok(self.fin(Op::Lods(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAD => Ok(self.fin(Op::Lods(size), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAE => Ok(self.fin(Op::Scas(Bits::B8), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAF => Ok(self.fin(Op::Scas(size), [Opnd::None; 4], o16, a16, seg_ov)),

            // ---- MOV r, imm8 / MOV r, imm16/32 ----
            0xB0..=0xB7 => {
                let r = b - 0xB0;
                let imm = self.raw()? as u32;
                Ok(self.fin(
                    Op::Mov,
                    [
                        Opnd::Reg(r, Bits::B8),
                        Opnd::Imm(imm),
                        Opnd::None,
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xB8..=0xBF => {
                let r = b - 0xB8;
                let imm = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                Ok(self.fin(
                    Op::Mov,
                    [Opnd::Reg(r, size), Opnd::Imm(imm), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }

            // ---- shifts ----
            0xC0 | 0xC1 => {
                let modrm = self.raw()?;
                let count = self.raw()?;
                self.group2(b, modrm, ShiftBy::Imm(count), o16, a16, seg_ov)
            }
            0xD0..=0xD3 => {
                let modrm = self.raw()?;
                let by = match b {
                    0xD0 | 0xD1 => ShiftBy::One,
                    _ => ShiftBy::Cl,
                };
                self.group2(b, modrm, by, o16, a16, seg_ov)
            }

            0xC2 => {
                let imm = self.raw16()?;
                Ok(self.fin(
                    Op::Ret {
                        far: false,
                        imm: Some(imm),
                    },
                    [Opnd::None; 4],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xC3 => Ok(self.fin(
                Op::Ret {
                    far: false,
                    imm: None,
                },
                [Opnd::None; 4],
                o16,
                a16,
                seg_ov,
            )),
            0xC4 => self.lds_like(Op::Les, o16, a16, seg_ov),
            0xC5 => self.lds_like(Op::Lds, o16, a16, seg_ov),
            0xC6 | 0xC7 => {
                let modrm = self.raw()?;
                let sz8 = b == 0xC6;
                let (_, opnd, _) = if sz8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let imm = if sz8 {
                    self.raw()? as u32
                } else if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                Ok(self.fin(
                    Op::Mov,
                    [opnd, Opnd::Imm(imm), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xC8 => {
                let imm16 = self.raw16()?;
                let level = self.raw()?;
                Ok(self.fin(Op::Enter(imm16, level), [Opnd::None; 4], o16, a16, seg_ov))
            }
            0xC9 => Ok(self.fin(Op::Leave, [Opnd::None; 4], o16, a16, seg_ov)),
            0xCA => {
                let imm = self.raw16()?;
                Ok(self.fin(
                    Op::Ret {
                        far: true,
                        imm: Some(imm),
                    },
                    [Opnd::None; 4],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xCB => Ok(self.fin(
                Op::Ret {
                    far: true,
                    imm: None,
                },
                [Opnd::None; 4],
                o16,
                a16,
                seg_ov,
            )),
            0xCC => Ok(self.fin(Op::Int3, [Opnd::None; 4], o16, a16, seg_ov)),
            0xCD => {
                let v = self.raw()?;
                Ok(self.fin(Op::Int(v), [Opnd::None; 4], o16, a16, seg_ov))
            }
            0xCE => Ok(self.fin(Op::Into, [Opnd::None; 4], o16, a16, seg_ov)),
            0xCF => Ok(self.fin(Op::Iret, [Opnd::None; 4], o16, a16, seg_ov)),
            0xD4 => {
                let _ = self.raw()?;
                Ok(self.fin(Op::Aam, [Opnd::None; 4], o16, a16, seg_ov))
            }
            0xD5 => {
                let _ = self.raw()?;
                Ok(self.fin(Op::Aad, [Opnd::None; 4], o16, a16, seg_ov))
            }
            0xD6 => Ok(self.fin(Op::Salc, [Opnd::None; 4], o16, a16, seg_ov)),
            0xD7 => Ok(self.fin(Op::Xlat, [Opnd::None; 4], o16, a16, seg_ov)),
            0xD8..=0xDF => {
                // ESC (FPU): consume a ModRM + displacement so the byte stream
                // stays aligned; no coprocessor is emulated.
                let modrm = self.raw()?;
                let _ = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(Op::Nop, [Opnd::None; 4], o16, a16, seg_ov))
            }
            0xE0..=0xE3 => {
                let disp = self.raw()? as i8 as i32;
                let op = match b {
                    0xE0 => Op::Loop(Some(0xE0)),
                    0xE1 => Op::Loop(Some(0xE1)),
                    0xE2 => Op::Loop(None),
                    _ => Op::Jcxz,
                };
                Ok(self.fin(
                    op,
                    [Opnd::Rel { disp }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE4 => {
                let port = self.raw()? as u16;
                Ok(self.fin(
                    Op::In,
                    [acc(Bits::B8), Opnd::Port(port), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE5 => {
                let port = self.raw()? as u16;
                Ok(self.fin(
                    Op::In,
                    [acc(size), Opnd::Port(port), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE6 => {
                let port = self.raw()? as u16;
                Ok(self.fin(
                    Op::Out,
                    [Opnd::Port(port), acc(Bits::B8), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE7 => {
                let port = self.raw()? as u16;
                Ok(self.fin(
                    Op::Out,
                    [Opnd::Port(port), acc(size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE8 => {
                let rel = if o16 {
                    self.raw16()? as i16 as i32
                } else {
                    self.raw32()? as i32
                };
                Ok(self.fin(
                    Op::Call { far: false },
                    [Opnd::Rel { disp: rel }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xE9 => {
                let rel = if o16 {
                    self.raw16()? as i16 as i32
                } else {
                    self.raw32()? as i32
                };
                Ok(self.fin(
                    Op::Jump { far: false },
                    [Opnd::Rel { disp: rel }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xEA => {
                let off = if o16 {
                    self.raw16()? as u32
                } else {
                    self.raw32()?
                };
                let sel = self.raw16()?;
                Ok(self.fin(
                    Op::Jump { far: true },
                    [
                        Opnd::FarPtr { sel, off },
                        Opnd::None,
                        Opnd::None,
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xEB => {
                let disp = self.raw()? as i8 as i32;
                Ok(self.fin(
                    Op::Jump { far: false },
                    [Opnd::Rel { disp }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xEC => Ok(self.fin(
                Op::In,
                [acc(Bits::B8), Opnd::Dx, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xED => Ok(self.fin(
                Op::In,
                [acc(size), Opnd::Dx, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xEE => Ok(self.fin(
                Op::Out,
                [Opnd::Dx, acc(Bits::B8), Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xEF => Ok(self.fin(
                Op::Out,
                [Opnd::Dx, acc(size), Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xF1 => Ok(self.fin(Op::Int(1), [Opnd::None; 4], o16, a16, seg_ov)),
            0xF4 => Ok(self.fin(Op::Hlt, [Opnd::None; 4], o16, a16, seg_ov)),
            0xF5 => Ok(self.fin(Op::Cmc, [Opnd::None; 4], o16, a16, seg_ov)),
            0xF6 | 0xF7 => {
                let modrm = self.raw()?;
                self.group3(b, modrm, o16, a16, seg_ov)
            }
            0xF8 => Ok(self.fin(Op::Clc, [Opnd::None; 4], o16, a16, seg_ov)),
            0xF9 => Ok(self.fin(Op::Stc, [Opnd::None; 4], o16, a16, seg_ov)),
            0xFA => Ok(self.fin(Op::Cli, [Opnd::None; 4], o16, a16, seg_ov)),
            0xFB => Ok(self.fin(Op::Sti, [Opnd::None; 4], o16, a16, seg_ov)),
            0xFC => Ok(self.fin(Op::Cld, [Opnd::None; 4], o16, a16, seg_ov)),
            0xFD => Ok(self.fin(Op::Std, [Opnd::None; 4], o16, a16, seg_ov)),
            0xFE => {
                let modrm = self.raw()?;
                let grp = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm8(modrm, false, a16, seg_ov)?;
                let op = match grp {
                    0 => Op::Inc,
                    1 => Op::Dec,
                    _ => return Err("group FE undefined".into()),
                };
                Ok(self.fin(
                    op,
                    [opnd, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xFF => {
                let modrm = self.raw()?;
                let grp = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm(modrm, o16, a16, seg_ov)?;
                let op = match grp {
                    0 => Op::Inc,
                    1 => Op::Dec,
                    2 => Op::Call { far: false },
                    3 => Op::Call { far: true },
                    4 => Op::Jump { far: false },
                    5 => Op::Jump { far: true },
                    6 => Op::Push,
                    _ => return Err("group FF undefined".into()),
                };
                Ok(self.fin(
                    op,
                    [opnd, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            _ => Ok(self.fin(Op::Unsupported(b), [Opnd::None; 4], o16, a16, seg_ov)),
        }
    }

    fn group1(
        &mut self,
        b: u8,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
        kind: u8,
    ) -> Result<Decoded, String> {
        let modrm = self.raw()?;
        let grp = (modrm >> 3) & 7;
        let op = match grp {
            0 => AluOp::Add,
            1 => AluOp::Or,
            2 => AluOp::Adc,
            3 => AluOp::Sbb,
            4 => AluOp::And,
            5 => AluOp::Sub,
            6 => AluOp::Xor,
            _ => AluOp::Cmp,
        };
        let size = if kind == 8 {
            Bits::B8
        } else if o16 {
            Bits::B16
        } else {
            Bits::B32
        };
        let (_, opnd, _) = if size == Bits::B8 {
            self.rm8(modrm, o16, a16, seg_ov)?
        } else {
            self.rm(modrm, o16, a16, seg_ov)?
        };
        let imm = if kind == 0x80 {
            self.raw()? as i8 as i32 as u32
        } else if size == Bits::B8 {
            self.raw()? as u32
        } else if o16 {
            self.raw16()? as u32
        } else {
            self.raw32()?
        };
        let imm_opnd = if kind == 0x80 {
            Opnd::ImmSext(imm)
        } else {
            Opnd::Imm(imm)
        };
        let _ = b;
        Ok(self.fin(
            Op::Alu(op),
            [opnd, imm_opnd, Opnd::None, Opnd::None],
            o16,
            a16,
            seg_ov,
        ))
    }

    fn group2(
        &mut self,
        b: u8,
        modrm: u8,
        by: ShiftBy,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<Decoded, String> {
        let grp = (modrm >> 3) & 7;
        let kind = match grp {
            0 => ShiftKind::Rol,
            1 => ShiftKind::Ror,
            2 => ShiftKind::Rcl,
            3 => ShiftKind::Rcr,
            4 => ShiftKind::Shl,
            5 => ShiftKind::Shr,
            6 => ShiftKind::Shl,
            _ => ShiftKind::Sar,
        };
        // 8-bit forms: C0, D0, D2. 16/32-bit forms: C1, D1, D3.
        let is8 = b == 0xC0 || b == 0xD0 || b == 0xD2;
        let _size = if is8 {
            Bits::B8
        } else if o16 {
            Bits::B16
        } else {
            Bits::B32
        };
        let (_, opnd, _) = if is8 {
            self.rm8(modrm, o16, a16, seg_ov)?
        } else {
            self.rm(modrm, o16, a16, seg_ov)?
        };
        Ok(self.fin(
            Op::Shift(kind, by),
            [opnd, Opnd::None, Opnd::None, Opnd::None],
            o16,
            a16,
            seg_ov,
        ))
    }

    fn group3(
        &mut self,
        b: u8,
        modrm: u8,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<Decoded, String> {
        let grp = (modrm >> 3) & 7;
        let is8 = b == 0xF6;
        let size = if is8 {
            Bits::B8
        } else if o16 {
            Bits::B16
        } else {
            Bits::B32
        };
        let (_, opnd, _) = if is8 {
            self.rm8(modrm, o16, a16, seg_ov)?
        } else {
            self.rm(modrm, o16, a16, seg_ov)?
        };
        let op = match grp {
            0 | 1 => Op::Test,
            2 => Op::Not,
            3 => Op::Neg,
            4 => Op::Mul,
            5 => Op::Imul,
            6 => Op::Div,
            _ => Op::Idiv,
        };
        if grp <= 1 {
            let imm = if size == Bits::B8 {
                self.raw()? as u32
            } else if o16 {
                self.raw16()? as u32
            } else {
                self.raw32()?
            };
            Ok(self.fin(
                op,
                [opnd, Opnd::Imm(imm), Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            ))
        } else {
            Ok(self.fin(
                op,
                [opnd, Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            ))
        }
    }

    fn lds_like(
        &mut self,
        op: Op,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<Decoded, String> {
        let modrm = self.raw()?;
        let size = if o16 { Bits::B16 } else { Bits::B32 };
        let (m, opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
        match opnd {
            Opnd::Mem(..) => Ok(self.fin(
                op,
                [Opnd::Reg(reg, size), opnd, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            _ => {
                let _ = m;
                Err("LDS/ES/SS/FS/GS with register operand".into())
            }
        }
    }

    fn decode_two_byte(
        &mut self,
        o16: bool,
        a16: bool,
        seg_ov: Option<Seg>,
    ) -> Result<Decoded, String> {
        let b = self.raw()?;
        let size = if o16 { Bits::B16 } else { Bits::B32 };

        match b {
            0x00 => {
                // group 6: SLDT/STR/LLDT/LTR/VERR/VERW
                let modrm = self.raw()?;
                let grp = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm(modrm, false, a16, seg_ov)?;
                let opnd = match opnd {
                    Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B16),
                    Opnd::Mem(m, _) => Opnd::Mem(m, Bits::B16),
                    other => other,
                };
                let op = match grp {
                    0 => Op::Sldt,
                    1 => Op::Str,
                    2 => Op::Lldt,
                    3 => Op::Ltr,
                    4 => Op::VErr(true),
                    5 => Op::VErr(false),
                    _ => return Err("group 6 undefined".into()),
                };
                Ok(self.fin(
                    op,
                    [opnd, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x01 => {
                // group 7: SGDT/SIDT/LGDT/LIDT/SMSW/LMSW/INVLPG
                let modrm = self.raw()?;
                let grp = (modrm >> 3) & 7;
                let (m, opnd, _) = self.rm(modrm, false, a16, seg_ov)?;
                let mem = match opnd {
                    Opnd::Mem(m, _) => m,
                    _ => return Err("group 7 needs memory operand".into()),
                };
                let op = match grp {
                    0 => Op::Sgdt,
                    1 => Op::Sidt,
                    2 => Op::Lgdt,
                    3 => Op::Lidt,
                    4 => Op::Smsw,
                    5 => Op::Lmsw,
                    6 => Op::Invlpg,
                    _ => return Err("group 7 undefined".into()),
                };
                let _ = m;
                Ok(self.fin(
                    op,
                    [
                        Opnd::Mem(mem, Bits::B16),
                        Opnd::None,
                        Opnd::None,
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x02 => {
                let modrm = self.raw()?;
                let (_, opnd, reg) = self.rm(modrm, false, a16, seg_ov)?;
                let opnd = match opnd {
                    Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B16),
                    Opnd::Mem(m, _) => Opnd::Mem(m, Bits::B16),
                    other => other,
                };
                Ok(self.fin(
                    Op::Lar,
                    [Opnd::Reg(reg, size), opnd, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x03 => {
                let modrm = self.raw()?;
                let (_, opnd, reg) = self.rm(modrm, false, a16, seg_ov)?;
                let opnd = match opnd {
                    Opnd::Reg(r, _) => Opnd::Reg(r, Bits::B16),
                    Opnd::Mem(m, _) => Opnd::Mem(m, Bits::B16),
                    other => other,
                };
                Ok(self.fin(
                    Op::Lsl,
                    [Opnd::Reg(reg, size), opnd, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x06 => Ok(self.fin(Op::Clts, [Opnd::None; 4], o16, a16, seg_ov)),
            0x07 => Ok(self.fin(Op::Invd, [Opnd::None; 4], o16, a16, seg_ov)),
            0x08 => Ok(self.fin(Op::Invd, [Opnd::None; 4], o16, a16, seg_ov)),
            0x09 => Ok(self.fin(Op::Wbinvd, [Opnd::None; 4], o16, a16, seg_ov)),
            0x20 | 0x22 => {
                // MOV r32, CRn / MOV CRn, r32
                let modrm = self.raw()?;
                let cr = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm(modrm, false, a16, seg_ov)?;
                let gpr = match opnd {
                    Opnd::Reg(r, _) => r,
                    _ => return Err("MOV CR needs register".into()),
                };
                let to_cr = b == 0x22;
                Ok(self.fin(
                    Op::MovCr,
                    [
                        Opnd::Imm(cr as u32),
                        Opnd::Reg(gpr, Bits::B32),
                        Opnd::Imm(to_cr as u32),
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x21 | 0x23 => {
                // MOV r32, DRn / MOV DRn, r32 (debug regs — treat as ordinary)
                let modrm = self.raw()?;
                let dr = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm(modrm, false, a16, seg_ov)?;
                let gpr = match opnd {
                    Opnd::Reg(r, _) => r,
                    _ => return Err("MOV DR needs register".into()),
                };
                let to_dr = b == 0x23;
                Ok(self.fin(
                    Op::MovDr,
                    [
                        Opnd::Imm(dr as u32),
                        Opnd::Reg(gpr, Bits::B32),
                        Opnd::Imm(to_dr as u32),
                        Opnd::None,
                    ],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x30 => Ok(self.fin(Op::Wrmsr, [Opnd::None; 4], o16, a16, seg_ov)),
            0x31 => Ok(self.fin(Op::Rdtsc, [Opnd::None; 4], o16, a16, seg_ov)),
            0x32 => Ok(self.fin(Op::Rdmsr, [Opnd::None; 4], o16, a16, seg_ov)),

            0x40..=0x4F => {
                let c = Cond::from_opcode(b);
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Cmov(c),
                    [Opnd::Reg(reg, size), src, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }

            0x80..=0x8F => {
                let c = Cond::from_opcode(b);
                let rel = if o16 {
                    self.raw16()? as i16 as i32
                } else {
                    self.raw32()? as i32
                };
                Ok(self.fin(
                    Op::Jcc(c),
                    [Opnd::Rel { disp: rel }, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0x90..=0x9F => {
                let c = Cond::from_opcode(b);
                let modrm = self.raw()?;
                let (_, opnd, _) = self.rm8(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Setcc(c),
                    [opnd, Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xA0 => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Fs), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xA1 => Ok(self.fin(Op::PopSeg(Seg::Fs), [Opnd::None; 4], o16, a16, seg_ov)),
            0xA2 => Ok(self.fin(Op::Cpuid, [Opnd::None; 4], o16, a16, seg_ov)),
            0xA3 => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Bt),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xA4 => {
                let modrm = self.raw()?;
                let count = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Dshift(true, ShiftBy::Imm(count)),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xA5 => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Dshift(true, ShiftBy::Cl),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xA8 => Ok(self.fin(
                Op::Push,
                [Opnd::Sreg(Seg::Gs), Opnd::None, Opnd::None, Opnd::None],
                o16,
                a16,
                seg_ov,
            )),
            0xA9 => Ok(self.fin(Op::PopSeg(Seg::Gs), [Opnd::None; 4], o16, a16, seg_ov)),
            0xAB => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Bts),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xAC => {
                let modrm = self.raw()?;
                let count = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Dshift(false, ShiftBy::Imm(count)),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xAD => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Dshift(false, ShiftBy::Cl),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xAF => {
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Imul1,
                    [Opnd::Reg(reg, size), src, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xB0 | 0xB1 => {
                let modrm = self.raw()?;
                let sz8 = b == 0xB0;
                let (_, a, reg) = if sz8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let sz = if sz8 { Bits::B8 } else { size };
                Ok(self.fin(
                    Op::CmpXchg,
                    [a, Opnd::Reg(reg, sz), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xB2 => self.lds_like(Op::Lss, o16, a16, seg_ov),
            0xB3 => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Btr),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xB4 => self.lds_like(Op::Lfs, o16, a16, seg_ov),
            0xB5 => self.lds_like(Op::Lgs, o16, a16, seg_ov),
            0xB6 | 0xB7 | 0xBE | 0xBF => {
                let from8 = b == 0xB6 || b == 0xBE;
                let sign = b == 0xBE || b == 0xBF;
                let modrm = self.raw()?;
                let src_sz = if from8 {
                    Bits::B8
                } else if o16 {
                    Bits::B16
                } else {
                    Bits::B32
                };
                let (_, src, reg) = if from8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let op = if sign {
                    Op::Movsx(from8)
                } else {
                    Op::Movzx(from8)
                };
                let _ = src_sz;
                Ok(self.fin(
                    op,
                    [Opnd::Reg(reg, size), src, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xBA => {
                // group 8: BT imm
                let modrm = self.raw()?;
                let imm = self.raw()?;
                let grp = (modrm >> 3) & 7;
                let (_, opnd, _) = self.rm(modrm, o16, a16, seg_ov)?;
                let bit_op = match grp {
                    4 => BitOp::Bt,
                    5 => BitOp::Bts,
                    6 => BitOp::Btr,
                    7 => BitOp::Btc,
                    _ => return Err("group 8 undefined".into()),
                };
                Ok(self.fin(
                    Op::Bit(bit_op),
                    [opnd, Opnd::Imm(imm as u32), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xBB => {
                let modrm = self.raw()?;
                let (_, rm_opnd, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Btc),
                    [rm_opnd, Opnd::Reg(reg, size), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xBC => {
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Bsf),
                    [Opnd::Reg(reg, size), src, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xBD => {
                let modrm = self.raw()?;
                let (_, src, reg) = self.rm(modrm, o16, a16, seg_ov)?;
                Ok(self.fin(
                    Op::Bit(BitOp::Bsr),
                    [Opnd::Reg(reg, size), src, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xC0 | 0xC1 => {
                let modrm = self.raw()?;
                let sz8 = b == 0xC0;
                let (_, a, reg) = if sz8 {
                    self.rm8(modrm, o16, a16, seg_ov)?
                } else {
                    self.rm(modrm, o16, a16, seg_ov)?
                };
                let sz = if sz8 { Bits::B8 } else { size };
                Ok(self.fin(
                    Op::Xadd,
                    [a, Opnd::Reg(reg, sz), Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            0xC8..=0xCF => {
                let r = b - 0xC8;
                Ok(self.fin(
                    Op::Bswap,
                    [Opnd::Reg(r, Bits::B32), Opnd::None, Opnd::None, Opnd::None],
                    o16,
                    a16,
                    seg_ov,
                ))
            }
            _ => Ok(self.fin(Op::TwoByte(b), [Opnd::None; 4], o16, a16, seg_ov)),
        }
    }
}
