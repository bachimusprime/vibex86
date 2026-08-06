//! Cranelift-based JIT.
//!
//! Compiles straight-line basic blocks to native code. Architectural subtleties
//! (flag math, memory translation, port I/O, stack ops, string ops) call back
//! into the shared `sem` layer, so JIT semantics are the interpreter's by
//! construction. Instructions the JIT does not compile end the trace and are
//! executed by one interpreter step; because those are exactly the
//! instructions that can change segmentation/paging/descriptor state, the
//! block cache is keyed on a state signature and stale blocks are recompiled.

use std::collections::HashMap;

use cranelift_codegen::ir::{
    self, AbiParam, BlockArg, InstBuilder, Signature, Value, condcodes::IntCC, types,
};
use cranelift_codegen::settings;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_native;

use crate::cpu::{Error, Seg, X86, flag};
use crate::decode::{Bits, Cond, Decoded, MemRef, Op, Opnd, Rep, ShiftBy, ShiftKind};
use crate::sem;

// ---------------------------------------------------------------------------
// Raw native helpers (extern symbols resolved by the JIT).
// ---------------------------------------------------------------------------

macro_rules! cpu {
    ($c:ident) => {
        unsafe { &mut *$c }
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_reg(cpu: *mut X86, idx: u64) -> u64 {
    cpu!(cpu).gpr[idx as usize] as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_reg(cpu: *mut X86, idx: u64, val: u64) -> u64 {
    cpu!(cpu).gpr[idx as usize] = val as u32;
    0
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_eip(cpu: *mut X86, eip: u64) -> u64 {
    cpu!(cpu).eip = eip as u32;
    0
}

fn bits_of(b: u64) -> Bits {
    match b {
        0 => Bits::B8,
        1 => Bits::B16,
        _ => Bits::B32,
    }
}
fn alu_of(op: u64) -> crate::decode::AluOp {
    use crate::decode::AluOp::*;
    match op {
        0 => Add,
        1 => Or,
        2 => Adc,
        3 => Sbb,
        4 => And,
        5 => Sub,
        6 => Xor,
        _ => Cmp,
    }
}
fn cond_of(c: u64) -> Cond {
    match c {
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
fn shift_of(k: u64) -> ShiftKind {
    match k {
        0 => ShiftKind::Rol,
        1 => ShiftKind::Ror,
        2 => ShiftKind::Rcl,
        3 => ShiftKind::Rcr,
        4 => ShiftKind::Shl,
        5 => ShiftKind::Shr,
        _ => ShiftKind::Sar,
    }
}

#[inline]
fn kind_of(k: u64) -> crate::cpu::AccessKind {
    match k {
        1 => crate::cpu::AccessKind::Write,
        2 => crate::cpu::AccessKind::Rmw,
        _ => crate::cpu::AccessKind::Read,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_alu(cpu: *mut X86, op: u64, a: u64, b: u64, bits: u64) -> u64 {
    let c = cpu!(cpu);
    sem::alu(c, alu_of(op), a as u32, b as u32, bits_of(bits)) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_test(cpu: *mut X86, a: u64, b: u64, bits: u64) -> u64 {
    let c = cpu!(cpu);
    sem::test_flags(c, a as u32, b as u32, bits_of(bits)) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_incdec(cpu: *mut X86, inc: u64, v: u64, bits: u64) -> u64 {
    let c = cpu!(cpu);
    sem::inc_dec(c, v as u32, bits_of(bits), inc != 0) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_neg(cpu: *mut X86, v: u64, bits: u64) -> u64 {
    let c = cpu!(cpu);
    sem::neg_op(c, v as u32, bits_of(bits)) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_shift(cpu: *mut X86, kind: u64, v: u64, count: u64, bits: u64) -> u64 {
    let c = cpu!(cpu);
    sem::shift(c, shift_of(kind), v as u32, count as u32, bits_of(bits)) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_cond(cpu: *mut X86, cond: u64) -> u64 {
    let c = cpu!(cpu);
    sem::cond_true(c, cond_of(cond)) as u64
}

fn mmu_rd(c: &mut X86, seg: u64, off: u64, kind: u64, size: u32) -> Option<u64> {
    let seg = Seg::from_idx(seg as u8);
    let kind = kind_of(kind);
    match size {
        1 => crate::mmu::read8(c, seg, off as u32, kind)
            .ok()
            .map(|v| v as u64),
        2 => crate::mmu::read16(c, seg, off as u32, kind)
            .ok()
            .map(|v| v as u64),
        _ => crate::mmu::read32(c, seg, off as u32, kind)
            .ok()
            .map(|v| v as u64),
    }
}
fn mmu_wr(c: &mut X86, seg: u64, off: u64, kind: u64, size: u32, val: u32) -> bool {
    let seg = Seg::from_idx(seg as u8);
    let kind = kind_of(kind);
    match size {
        1 => crate::mmu::write8(c, seg, off as u32, val as u8, kind).is_ok(),
        2 => crate::mmu::write16(c, seg, off as u32, val as u16, kind).is_ok(),
        _ => crate::mmu::write32(c, seg, off as u32, val, kind).is_ok(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_r8(cpu: *mut X86, seg: u64, off: u64, kind: u64) -> u64 {
    let c = cpu!(cpu);
    match mmu_rd(c, seg, off, kind, 1) {
        Some(v) => v,
        None => 1u64 << 32,
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_r16(cpu: *mut X86, seg: u64, off: u64, kind: u64) -> u64 {
    let c = cpu!(cpu);
    match mmu_rd(c, seg, off, kind, 2) {
        Some(v) => v,
        None => 1u64 << 32,
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_r32(cpu: *mut X86, seg: u64, off: u64, kind: u64) -> u64 {
    let c = cpu!(cpu);
    match mmu_rd(c, seg, off, kind, 4) {
        Some(v) => v,
        None => 1u64 << 32,
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_w8(cpu: *mut X86, seg: u64, off: u64, val: u64, kind: u64) -> u64 {
    let c = cpu!(cpu);
    mmu_wr(c, seg, off, kind, 1, val as u32) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_w16(
    cpu: *mut X86,
    seg: u64,
    off: u64,
    val: u64,
    kind: u64,
) -> u64 {
    let c = cpu!(cpu);
    mmu_wr(c, seg, off, kind, 2, val as u32) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_mem_w32(
    cpu: *mut X86,
    seg: u64,
    off: u64,
    val: u64,
    kind: u64,
) -> u64 {
    let c = cpu!(cpu);
    mmu_wr(c, seg, off, kind, 4, val as u32) as u64
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_io_r(cpu: *mut X86, port: u64, size: u64) -> u64 {
    cpu!(cpu).mem.io_read(port as u16, size as u8) as u64
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_io_w(cpu: *mut X86, port: u64, size: u64, val: u64) -> u64 {
    cpu!(cpu).mem.io_write(port as u16, size as u8, val as u32);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_push(cpu: *mut X86, size: u64, val: u64) -> u64 {
    let c = cpu!(cpu);
    match sem::push_public(c, size as u32, val as u32) {
        Ok(()) => 0,
        Err(_) => 1,
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_pop(cpu: *mut X86, size: u64) -> u64 {
    let c = cpu!(cpu);
    match sem::pop_public(c, size as u32) {
        Ok(v) => v as u64,
        Err(_) => 1u64 << 32,
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_stack_add(cpu: *mut X86, delta: u64) -> u64 {
    let c = cpu!(cpu);
    c.gpr[4] = c.gpr[4].wrapping_add(delta as u32);
    0
}

/// op: 0=movs, 1=stos, 2=lods, 3=cmps, 4=scas, 5=ins, 6=outs; rep: 0/1/2; bits.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_string(cpu: *mut X86, op: u64, rep: u64, bits: u64, _eip: u64) -> u64 {
    let c = cpu!(cpu);
    let rep = match rep {
        1 => Rep::Z,
        2 => Rep::NZ,
        _ => Rep::None,
    };
    sem::string_op(c, op as u8, rep, bits_of(bits), 0);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_flagop(cpu: *mut X86, op: u64) -> u64 {
    let c = cpu!(cpu);
    match op {
        0 => c.eflags &= !flag::IF,
        1 => c.eflags |= flag::IF,
        2 => c.eflags &= !flag::DF,
        3 => c.eflags |= flag::DF,
        4 => c.eflags &= !flag::CF,
        5 => c.eflags |= flag::CF,
        6 => c.eflags ^= flag::CF,
        7 => c.cr[0] &= !(1 << 3),
        _ => {}
    }
    0
}

// ---------------------------------------------------------------------------
// Imported function references
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Imports {
    get_reg: FuncId,
    set_reg: FuncId,
    set_eip: FuncId,
    alu: FuncId,
    test: FuncId,
    incdec: FuncId,
    neg: FuncId,
    shift: FuncId,
    cond: FuncId,
    mem_r8: FuncId,
    mem_r16: FuncId,
    mem_r32: FuncId,
    mem_w8: FuncId,
    mem_w16: FuncId,
    mem_w32: FuncId,
    io_r: FuncId,
    io_w: FuncId,
    push: FuncId,
    pop: FuncId,
    stack_add: FuncId,
    string: FuncId,
    flagop: FuncId,
}

#[derive(Clone, Copy)]
struct Frefs {
    get_reg: ir::FuncRef,
    set_reg: ir::FuncRef,
    set_eip: ir::FuncRef,
    alu: ir::FuncRef,
    test: ir::FuncRef,
    incdec: ir::FuncRef,
    neg: ir::FuncRef,
    shift: ir::FuncRef,
    cond: ir::FuncRef,
    mem_r8: ir::FuncRef,
    mem_r16: ir::FuncRef,
    mem_r32: ir::FuncRef,
    mem_w8: ir::FuncRef,
    mem_w16: ir::FuncRef,
    mem_w32: ir::FuncRef,
    io_r: ir::FuncRef,
    io_w: ir::FuncRef,
    push: ir::FuncRef,
    pop: ir::FuncRef,
    stack_add: ir::FuncRef,
    string: ir::FuncRef,
    flagop: ir::FuncRef,
}

/// A compiled basic block.
pub struct BlockEntry {
    pub func: unsafe extern "C" fn(*mut X86) -> u64,
    pub len: u32,
    pub state: u64,
}

pub type BlockFunc = unsafe extern "C" fn(*mut X86) -> u64;

fn block_key(cpu: &X86) -> u64 {
    ((cpu.seg[Seg::Cs as usize].sel as u64) << 32) | cpu.eip as u64
}

fn state_sig(cpu: &X86) -> u64 {
    let g = cpu.gdtr.base ^ (cpu.gdtr.limit as u32).rotate_left(16);
    let i = cpu.idtr.base ^ (cpu.idtr.limit as u32).rotate_left(24);
    (cpu.cr[0] as u64) ^ ((g as u64) << 16) ^ ((i as u64) << 40)
}

/// JIT engine.
pub struct Jit {
    #[allow(dead_code)]
    module: JITModule,
    imports: Imports,
    compiled: HashMap<u64, BlockEntry>,
    pub blocks_compiled: u64,
    pub blocks_recompiled: u64,
    pub instructions_run: u64,
}

impl Jit {
    pub fn new() -> Result<Jit, String> {
        let isa_builder = cranelift_native::builder()
            .map_err(|_| "native Cranelift target unsupported".to_string())?;
        let flags = settings::Flags::new(settings::builder());
        let isa = isa_builder
            .finish(flags)
            .map_err(|e| format!("Cranelift ISA: {e}"))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let syms: &[(&str, *const u8)] = &[
            ("jit_get_reg", jit_get_reg as *const u8),
            ("jit_set_reg", jit_set_reg as *const u8),
            ("jit_set_eip", jit_set_eip as *const u8),
            ("jit_alu", jit_alu as *const u8),
            ("jit_test", jit_test as *const u8),
            ("jit_incdec", jit_incdec as *const u8),
            ("jit_neg", jit_neg as *const u8),
            ("jit_shift", jit_shift as *const u8),
            ("jit_cond", jit_cond as *const u8),
            ("jit_mem_r8", jit_mem_r8 as *const u8),
            ("jit_mem_r16", jit_mem_r16 as *const u8),
            ("jit_mem_r32", jit_mem_r32 as *const u8),
            ("jit_mem_w8", jit_mem_w8 as *const u8),
            ("jit_mem_w16", jit_mem_w16 as *const u8),
            ("jit_mem_w32", jit_mem_w32 as *const u8),
            ("jit_io_r", jit_io_r as *const u8),
            ("jit_io_w", jit_io_w as *const u8),
            ("jit_push", jit_push as *const u8),
            ("jit_pop", jit_pop as *const u8),
            ("jit_stack_add", jit_stack_add as *const u8),
            ("jit_string", jit_string as *const u8),
            ("jit_flagop", jit_flagop as *const u8),
        ];
        builder.symbols(syms.iter().map(|(n, p)| (*n, *p)));

        let mut module = JITModule::new(builder);

        fn helper_sig(module: &mut JITModule, params: usize) -> Signature {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(types::I64));
            for _ in 0..params {
                s.params.push(AbiParam::new(types::I64));
            }
            s.returns.push(AbiParam::new(types::I64));
            s
        }

        fn import(module: &mut JITModule, name: &str, sig: &Signature) -> Result<FuncId, String> {
            module
                .declare_function(name, Linkage::Import, sig)
                .map_err(|e| format!("declare {name}: {e}"))
        }

        // Precompute every signature up front so the module borrow is not held
        // across multiple `import` calls.
        let sigs: Vec<Signature> = (1..=4).map(|n| helper_sig(&mut module, n)).collect();
        let imports = Imports {
            get_reg: import(&mut module, "jit_get_reg", &sigs[0])?,
            set_reg: import(&mut module, "jit_set_reg", &sigs[1])?,
            set_eip: import(&mut module, "jit_set_eip", &sigs[0])?,
            alu: import(&mut module, "jit_alu", &sigs[3])?,
            test: import(&mut module, "jit_test", &sigs[2])?,
            incdec: import(&mut module, "jit_incdec", &sigs[2])?,
            neg: import(&mut module, "jit_neg", &sigs[1])?,
            shift: import(&mut module, "jit_shift", &sigs[3])?,
            cond: import(&mut module, "jit_cond", &sigs[0])?,
            mem_r8: import(&mut module, "jit_mem_r8", &sigs[2])?,
            mem_r16: import(&mut module, "jit_mem_r16", &sigs[2])?,
            mem_r32: import(&mut module, "jit_mem_r32", &sigs[2])?,
            mem_w8: import(&mut module, "jit_mem_w8", &sigs[3])?,
            mem_w16: import(&mut module, "jit_mem_w16", &sigs[3])?,
            mem_w32: import(&mut module, "jit_mem_w32", &sigs[3])?,
            io_r: import(&mut module, "jit_io_r", &sigs[1])?,
            io_w: import(&mut module, "jit_io_w", &sigs[2])?,
            push: import(&mut module, "jit_push", &sigs[1])?,
            pop: import(&mut module, "jit_pop", &sigs[0])?,
            stack_add: import(&mut module, "jit_stack_add", &sigs[0])?,
            string: import(&mut module, "jit_string", &sigs[3])?,
            flagop: import(&mut module, "jit_flagop", &sigs[0])?,
        };

        module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;

        Ok(Jit {
            module,
            imports,
            compiled: HashMap::new(),
            blocks_compiled: 0,
            blocks_recompiled: 0,
            instructions_run: 0,
        })
    }

    fn block_for(&mut self, cpu: &mut X86) -> Option<&BlockEntry> {
        let key = block_key(cpu);
        let sig = state_sig(cpu);
        if self.compiled.get(&key).is_some_and(|e| e.state == sig) {
            return self.compiled.get(&key);
        }
        if self.compiled.contains_key(&key) {
            self.blocks_recompiled += 1;
        }
        let entry = self.compile(cpu)?;
        self.blocks_compiled += 1;
        if self.compiled.len() > 32 * 1024 {
            self.compiled.clear();
        }
        self.compiled.insert(key, entry);
        self.compiled.get(&key)
    }

    /// Run up to `limit` instructions. Returns the number executed or an error.
    pub fn run(&mut self, cpu: &mut X86, limit: u64) -> Result<u64, Error> {
        let mut count = 0u64;
        while count < limit {
            if sem::deliver_maskable_interrupt(cpu).is_some() {
                count += 1;
                continue;
            }

            let (status, n) = {
                let entry = match self.block_for(cpu) {
                    Some(e) => e,
                    None => {
                        let out = cpu.step();
                        count += 1;
                        if let crate::cpu::StepOut::Error(e) = out {
                            return Err(e);
                        }
                        continue;
                    }
                };
                if entry.len as u64 > limit - count {
                    let out = cpu.step();
                    count += 1;
                    if let crate::cpu::StepOut::Error(e) = out {
                        return Err(e);
                    }
                    cpu.mem.tick_device(1);
                    continue;
                }
                let func = entry.func;
                let n = entry.len as u64;
                // Safety: `func` only reads/writes `cpu` through the pointer
                // we pass; no other mutable borrow exists across the call.
                let status = unsafe { func(cpu as *mut X86) };
                (status, n)
            };

            count += n;
            cpu.cycles = cpu.cycles.wrapping_add(n);
            self.instructions_run += n;

            if status == 1 {
                let out = cpu.step();
                if let crate::cpu::StepOut::Error(e) = out {
                    return Err(e);
                }
                count += 1;
                cpu.mem.tick_device(1);
            }
            if n > 0 {
                cpu.mem.tick_device(n);
            }
        }
        Ok(count)
    }

    fn compile(&mut self, cpu: &mut X86) -> Option<BlockEntry> {
        let eip0 = cpu.eip;
        let key = block_key(cpu);
        let sig0 = state_sig(cpu);

        // Trace-decode a straight-line run of native instructions.
        let mut plan: Vec<Decoded> = Vec::new();
        let mut cur = eip0;
        let tail_status: u32 = loop {
            let saved = cpu.eip;
            cpu.eip = cur;
            let d = crate::decode::fetch(cpu).ok()?;
            cpu.eip = saved;
            let kind = nat_kind(&d);
            // Always push the last decoded instruction when it terminates the
            // trace so the block keeps the terminating opcode (status 0:
            // continue at tail with the branch already emitted inside).
            match kind {
                NatKind::Native if plan.len() < 255 => {
                    plan.push(d);
                    cur = cur.wrapping_add(d.len as u32);
                }
                NatKind::Terminal => {
                    plan.push(d);
                    break 0;
                }
                _ => {
                    break 1;
                }
            }
        };
        if plan.is_empty() {
            return None;
        }
        let tail_eip = cur;

        let name = format!("blk_{key:016x}");
        let mut ctx = self.module.make_context();
        let mut sig = self.module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        let func_id = self
            .module
            .declare_function(&name, Linkage::Local, &sig)
            .ok()?;
        ctx.func.signature = sig.clone();
        let frefs = Frefs {
            get_reg: self
                .module
                .declare_func_in_func(self.imports.get_reg, &mut ctx.func),
            set_reg: self
                .module
                .declare_func_in_func(self.imports.set_reg, &mut ctx.func),
            set_eip: self
                .module
                .declare_func_in_func(self.imports.set_eip, &mut ctx.func),
            alu: self
                .module
                .declare_func_in_func(self.imports.alu, &mut ctx.func),
            test: self
                .module
                .declare_func_in_func(self.imports.test, &mut ctx.func),
            incdec: self
                .module
                .declare_func_in_func(self.imports.incdec, &mut ctx.func),
            neg: self
                .module
                .declare_func_in_func(self.imports.neg, &mut ctx.func),
            shift: self
                .module
                .declare_func_in_func(self.imports.shift, &mut ctx.func),
            cond: self
                .module
                .declare_func_in_func(self.imports.cond, &mut ctx.func),
            mem_r8: self
                .module
                .declare_func_in_func(self.imports.mem_r8, &mut ctx.func),
            mem_r16: self
                .module
                .declare_func_in_func(self.imports.mem_r16, &mut ctx.func),
            mem_r32: self
                .module
                .declare_func_in_func(self.imports.mem_r32, &mut ctx.func),
            mem_w8: self
                .module
                .declare_func_in_func(self.imports.mem_w8, &mut ctx.func),
            mem_w16: self
                .module
                .declare_func_in_func(self.imports.mem_w16, &mut ctx.func),
            mem_w32: self
                .module
                .declare_func_in_func(self.imports.mem_w32, &mut ctx.func),
            io_r: self
                .module
                .declare_func_in_func(self.imports.io_r, &mut ctx.func),
            io_w: self
                .module
                .declare_func_in_func(self.imports.io_w, &mut ctx.func),
            push: self
                .module
                .declare_func_in_func(self.imports.push, &mut ctx.func),
            pop: self
                .module
                .declare_func_in_func(self.imports.pop, &mut ctx.func),
            stack_add: self
                .module
                .declare_func_in_func(self.imports.stack_add, &mut ctx.func),
            string: self
                .module
                .declare_func_in_func(self.imports.string, &mut ctx.func),
            flagop: self
                .module
                .declare_func_in_func(self.imports.flagop, &mut ctx.func),
        };

        let frontend_config = self.module.target_config();
        let mut fctx = FunctionBuilderContext::new();
        let mut fb = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        emit_block(&mut fb, frefs, &plan, eip0, tail_eip, tail_status);
        fb.finalize(frontend_config);

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("define {name}: {e}"))
            .ok()?;
        self.module
            .finalize_definitions()
            .map_err(|e| format!("finalize {name}: {e}"))
            .ok()?;
        let ptr = self.module.get_finalized_function(func_id);
        // Safety: `ptr` is the machine code of the function we just defined
        // with signature `(*mut X86) -> u64`.
        let func: BlockFunc = unsafe { std::mem::transmute(ptr) };
        Some(BlockEntry {
            func,
            len: plan.len() as u32,
            state: sig0,
        })
    }
}

enum NatKind {
    Native,
    Terminal,
    Fallback,
}

fn nat_kind(d: &Decoded) -> NatKind {
    use Op::*;
    match d.op {
        Alu(_) | Test | Inc | Dec | Neg | Not | Shift(..) | Mov | Movzx(_) | Movsx(_) | Lea
        | Xchg | Setcc(_) | Cmov(_) | In | Out | Xlat | Nop | Wait | Salc | Cli | Sti | Cld
        | Std | Clc | Stc | Cmc | Clts => {
            if d.rep != Rep::None {
                NatKind::Fallback
            } else {
                NatKind::Native
            }
        }
        Push | Pop
            if matches!(
                d.ops[0],
                Opnd::Reg(..) | Opnd::Mem(..) | Opnd::Imm(_) | Opnd::ImmSext(_)
            ) && d.rep == Rep::None =>
        {
            NatKind::Native
        }
        Movs(_) | Stos(_) | Lods(_) | Cmps(_) | Scas(_) | Ins(_) | Outs(_) => NatKind::Terminal,
        Jcc(_)
        | Jump { far: false }
        | Call { far: false }
        | Ret { far: false, .. }
        | Loop(_)
        | Jcxz => NatKind::Terminal,
        Hlt => NatKind::Fallback,
        _ => NatKind::Fallback,
    }
}

// ---------------------------------------------------------------------------
// Block emission
// ---------------------------------------------------------------------------

/// Emit one block. `tail_status` 0 means "continue at tail EIP with the normal
/// interpreter loop", 1 means "run the interpreter at the tail EIP once".
fn emit_block(
    fb: &mut FunctionBuilder,
    frs: Frefs,
    plan: &[Decoded],
    eip0: u32,
    tail_eip: u32,
    tail_status: u32,
) {
    let entry = fb.create_block();
    fb.switch_to_block(entry);
    fb.append_block_params_for_function_params(entry);
    let cpu = fb.block_params(entry)[0];

    // Cranelift 0.134: Variables are opaque entities created via `from_u32`.
    let mut vregs = [Variable::from_u32(0); 8];
    for (i, v) in vregs.iter_mut().enumerate() {
        *v = Variable::from_u32(i as u32);
        fb.declare_var(types::I32);
    }
    for i in 0..8 {
        let idx = fb.ins().iconst(types::I64, i as i64);
        let inst = fb.ins().call(frs.get_reg, &[cpu, idx]);
        let r = fb.inst_results(inst)[0];
        let r32 = fb.ins().ireduce(types::I32, r);
        fb.def_var(vregs[i], r32);
    }

    let epi = fb.create_block();
    fb.append_block_param(epi, types::I32);
    fb.append_block_param(epi, types::I32);
    let fault = fb.create_block();
    fb.append_block_param(fault, types::I32);

    let mut em = Emitter {
        fb,
        frs,
        cpu,
        vregs,
        epi,
        fault,
        cur: entry,
        terminated: false,
    };

    let mut cur_eip = eip0;
    for d in plan.iter() {
        em.emit_ins(d, cur_eip);
        cur_eip = cur_eip.wrapping_add(d.len as u32);
    }

    if !em.terminated {
        // Tail jump: continue at `tail_eip` with the given status.
        let te = em.fb.ins().iconst(types::I32, tail_eip as i64);
        let st = em.fb.ins().iconst(types::I32, tail_status as i64);
        em.fb
            .ins()
            .jump(em.epi, &[BlockArg::from(te), BlockArg::from(st)]);
    }

    // Fault block: run the interpreter at the given EIP (status = 1).
    em.fb.switch_to_block(em.fault);
    {
        let eip_param = em.fb.block_params(em.fault)[0];
        let one = em.fb.ins().iconst(types::I32, 1);
        em.fb
            .ins()
            .jump(em.epi, &[BlockArg::from(eip_param), BlockArg::from(one)]);
    }

    // Epilogue: store regs + eip, return status.
    em.fb.switch_to_block(em.epi);
    {
        let eip_param = em.fb.block_params(em.epi)[0];
        let status = em.fb.block_params(em.epi)[1];
        for i in 0..8 {
            let idx = em.fb.ins().iconst(types::I64, i as i64);
            let v = em.fb.use_var(em.vregs[i]);
            let v64 = em.fb.ins().uextend(types::I64, v);
            em.fb.ins().call(em.frs.set_reg, &[em.cpu, idx, v64]);
        }
        let eip64 = em.fb.ins().uextend(types::I64, eip_param);
        em.fb.ins().call(em.frs.set_eip, &[em.cpu, eip64]);
        let s64 = em.fb.ins().uextend(types::I64, status);
        em.fb.ins().return_(&[s64]);
    }

    em.fb.seal_block(entry);
    em.fb.seal_block(epi);
    em.fb.seal_block(fault);
    em.fb.seal_all_blocks();
}

struct Emitter<'a, 'b> {
    fb: &'a mut FunctionBuilder<'b>,
    frs: Frefs,
    cpu: Value,
    vregs: [Variable; 8],
    epi: cranelift_codegen::ir::Block,
    fault: cranelift_codegen::ir::Block,
    cur: cranelift_codegen::ir::Block,
    terminated: bool,
}

fn bits_id(b: Bits) -> u64 {
    match b {
        Bits::B8 => 0,
        Bits::B16 => 1,
        Bits::B32 => 2,
    }
}
fn alu_id(op: crate::decode::AluOp) -> u64 {
    use crate::decode::AluOp::*;
    match op {
        Add => 0,
        Or => 1,
        Adc => 2,
        Sbb => 3,
        And => 4,
        Sub => 5,
        Xor => 6,
        Cmp => 7,
    }
}
fn cond_id(c: Cond) -> u64 {
    c as u64
}
fn shift_id(k: ShiftKind) -> u64 {
    match k {
        ShiftKind::Rol => 0,
        ShiftKind::Ror => 1,
        ShiftKind::Rcl => 2,
        ShiftKind::Rcr => 3,
        ShiftKind::Shl => 4,
        ShiftKind::Shr => 5,
        ShiftKind::Sar => 6,
    }
}

#[inline]
fn i32(e: &mut Emitter, v: i64) -> Value {
    e.fb.ins().iconst(types::I32, v)
}
#[inline]
fn i64(e: &mut Emitter, v: i64) -> Value {
    e.fb.ins().iconst(types::I64, v)
}

/// A B1 comparison value widened into an I32 0/1 integer (replaces `bint`).
#[inline]
fn widen_bool(e: &mut Emitter, b: Value) -> Value {
    e.fb.ins().uextend(types::I32, b)
}

/// Read a register as an I32 `Value` honoring the operand size.
fn reg_val(e: &mut Emitter, reg: u8, bits: Bits) -> Value {
    let idx = if matches!(bits, Bits::B8) {
        reg & 3
    } else {
        reg & 7
    };
    let v = e.fb.use_var(e.vregs[idx as usize]);
    match bits {
        Bits::B8 => {
            if reg & 4 != 0 {
                let eight = i32(e, 8);
                let hi = e.fb.ins().ushr(v, eight);
                let mask = i32(e, 0xFF);
                e.fb.ins().band(hi, mask)
            } else {
                let mask = i32(e, 0xFF);
                e.fb.ins().band(v, mask)
            }
        }
        Bits::B16 => {
            let w = e.fb.ins().ireduce(types::I16, v);
            e.fb.ins().uextend(types::I32, w)
        }
        Bits::B32 => v,
    }
}

/// Write a register (honoring the operand size).
fn set_reg(e: &mut Emitter, reg: u8, bits: Bits, val: Value) {
    let idx = if matches!(bits, Bits::B8) {
        reg & 3
    } else {
        reg & 7
    };
    let old = e.fb.use_var(e.vregs[idx as usize]);
    match bits {
        Bits::B8 => {
            let mask = i32(e, 0xFF);
            let lo = e.fb.ins().band(val, mask);
            let new = if reg & 4 != 0 {
                let eight = i32(e, 8);
                let shifted = e.fb.ins().ishl(lo, eight);
                let keep_mask = i32(e, 0xFFFF_00FF);
                let kept = e.fb.ins().band(old, keep_mask);
                e.fb.ins().bor(kept, shifted)
            } else {
                let keep_mask = i32(e, 0xFFFF_FF00);
                let kept = e.fb.ins().band(old, keep_mask);
                e.fb.ins().bor(kept, lo)
            };
            e.fb.def_var(e.vregs[idx as usize], new);
        }
        Bits::B16 => {
            let mask = i32(e, 0xFFFF);
            let lo16 = e.fb.ins().band(val, mask);
            let keep_mask = i32(e, 0xFFFF_0000);
            let kept = e.fb.ins().band(old, keep_mask);
            let new = e.fb.ins().bor(kept, lo16);
            e.fb.def_var(e.vregs[idx as usize], new);
        }
        Bits::B32 => e.fb.def_var(e.vregs[idx as usize], val),
    }
}

/// Compute an effective address (offset within segment) as an I32 value.
fn eff_addr(e: &mut Emitter, m: &MemRef) -> Value {
    let reg_bits = if m.a16 { Bits::B16 } else { Bits::B32 };
    let mut acc = i32(e, 0);
    if let Some(b) = m.base {
        let base = reg_val(e, b, reg_bits);
        acc = e.fb.ins().iadd(acc, base);
    }
    if let Some(i) = m.index {
        let v = reg_val(e, i, reg_bits);
        acc = if m.scale != 1 {
            let scale = i32(e, m.scale as i64);
            let s = e.fb.ins().imul(v, scale);
            e.fb.ins().iadd(acc, s)
        } else {
            e.fb.ins().iadd(acc, v)
        };
    }
    if m.disp != 0 {
        let d = i32(e, m.disp as i64);
        acc = e.fb.ins().iadd(acc, d);
    }
    if m.a16 {
        let w = e.fb.ins().ireduce(types::I16, acc);
        e.fb.ins().uextend(types::I32, w)
    } else {
        acc
    }
}

impl<'a, 'b> Emitter<'a, 'b> {
    /// Call an imported helper. Args must be built *before* this call.
    fn call(&mut self, fr: ir::FuncRef, args: &[Value]) -> Value {
        let mut all_args = Vec::with_capacity(args.len() + 1);
        all_args.push(self.cpu);
        all_args.extend_from_slice(args);
        let inst = self.fb.ins().call(fr, &all_args);
        self.fb.inst_results(inst)[0]
    }

    fn mem_load_off(&mut self, seg: Seg, off: Value, bits: Bits, kind: u64, ins_eip: u32) -> Value {
        let seg = i64(self, seg as u8 as i64);
        let off64 = self.fb.ins().uextend(types::I64, off);
        let kindv = i64(self, kind as i64);
        let fr = match bits {
            Bits::B8 => self.frs.mem_r8,
            Bits::B16 => self.frs.mem_r16,
            Bits::B32 => self.frs.mem_r32,
        };
        let r = self.call(fr, &[seg, off64, kindv]);
        self.fault_branch(r, true, ins_eip);
        let lo = self.fb.ins().ireduce(types::I32, r);
        match bits {
            Bits::B8 => {
                let mask = i32(self, 0xFF);
                self.fb.ins().band(lo, mask)
            }
            Bits::B16 => {
                let mask = i32(self, 0xFFFF);
                self.fb.ins().band(lo, mask)
            }
            Bits::B32 => lo,
        }
    }

    /// `sem::cond_true` -> B1 condition value.
    fn cond_call(&mut self, c: Cond) -> Value {
        let arg = i64(self, cond_id(c) as i64);
        let r = self.call(self.frs.cond, &[arg]);
        let r32 = self.fb.ins().ireduce(types::I32, r);
        self.fb.ins().icmp_imm_s(IntCC::NotEqual, r32, 0)
    }

    fn mem_load(&mut self, m: &MemRef, bits: Bits, kind: u64, ins_eip: u32) -> Value {
        let off = eff_addr(self, m);
        self.mem_load_off(m.seg, off, bits, kind, ins_eip)
    }

    fn mem_store(&mut self, m: &MemRef, bits: Bits, val: Value, ins_eip: u32) {
        let seg = i64(self, m.seg as u8 as i64);
        let off = eff_addr(self, m);
        let off64 = self.fb.ins().uextend(types::I64, off);
        let mask = match bits {
            Bits::B8 => 0xFF,
            Bits::B16 => 0xFFFF,
            _ => u32::MAX,
        };
        let val32 = if mask == u32::MAX {
            val
        } else {
            let maskv = i32(self, mask as i64);
            self.fb.ins().band(val, maskv)
        };
        let val64 = self.fb.ins().uextend(types::I64, val32);
        let kindv = i64(self, 1);
        let fr = match bits {
            Bits::B8 => self.frs.mem_w8,
            Bits::B16 => self.frs.mem_w16,
            Bits::B32 => self.frs.mem_w32,
        };
        let r = self.call(fr, &[seg, off64, val64, kindv]);
        self.fault_branch(r, false, ins_eip);
    }

    /// If `r` indicates a fault, jump to `fault` (with the ins EIP) instead of
    /// continuing; set `self.cur` to the continuation block.
    fn fault_branch(&mut self, r: Value, high_bit: bool, ins_eip: u32) {
        let cond = if high_bit {
            let hi = self.fb.ins().ushr_imm_s(r, 32);
            let hi32 = self.fb.ins().ireduce(types::I32, hi);
            self.fb.ins().icmp_imm_s(IntCC::NotEqual, hi32, 0)
        } else {
            let r32 = self.fb.ins().ireduce(types::I32, r);
            self.fb.ins().icmp_imm_s(IntCC::NotEqual, r32, 0)
        };
        let cont = self.fb.create_block();
        let fblock = self.fb.create_block();
        self.fb.ins().brif(cond, fblock, &[], cont, &[]);
        self.fb.switch_to_block(fblock);
        let eip = i32(self, ins_eip as i64);
        self.fb.ins().jump(self.fault, &[BlockArg::from(eip)]);
        self.fb.seal_block(fblock);
        self.fb.switch_to_block(cont);
        self.fb.seal_block(cont);
        self.cur = cont;
    }

    fn opnd_read(&mut self, o: &Opnd, ins_eip: u32) -> Value {
        match o {
            Opnd::Reg(r, bits) => reg_val(self, *r, *bits),
            Opnd::Acc(bits) => reg_val(self, 0, *bits),
            Opnd::Mem(m, bits) => self.mem_load(m, *bits, 0, ins_eip),
            Opnd::Imm(v) | Opnd::ImmSext(v) => i32(self, *v as i64),
            Opnd::Cl => reg_val(self, 1, Bits::B8),
            Opnd::Dx => reg_val(self, 2, Bits::B16),
            _ => i32(self, 0),
        }
    }

    fn opnd_write(&mut self, o: &Opnd, val: Value, ins_eip: u32) {
        match o {
            Opnd::Reg(r, bits) => set_reg(self, *r, *bits, val),
            Opnd::Acc(bits) => set_reg(self, 0, *bits, val),
            Opnd::Mem(m, bits) => self.mem_store(m, *bits, val, ins_eip),
            _ => {}
        }
    }

    /// Jump to the block epilogue with `(eip=target, status=0)`.
    fn jump_target(&mut self, target: u32) {
        let t = i32(self, target as i64);
        let z = i32(self, 0);
        self.fb
            .ins()
            .jump(self.epi, &[BlockArg::from(t), BlockArg::from(z)]);
        self.terminated = true;
    }

    fn branch_cc(&mut self, cond: Value, taken_target: u32, next_eip: u32) {
        let tb = self.fb.create_block();
        let nb = self.fb.create_block();
        self.fb.ins().brif(cond, tb, &[], nb, &[]);
        self.fb.switch_to_block(tb);
        self.jump_target(taken_target);
        self.fb.seal_block(tb);
        self.fb.switch_to_block(nb);
        self.jump_target(next_eip);
        self.fb.seal_block(nb);
    }

    fn emit_ins(&mut self, d: &Decoded, ins_eip: u32) {
        let bits = match d.ops[0] {
            Opnd::Reg(_, b) | Opnd::Mem(_, b) | Opnd::Acc(b) => b,
            _ => d.size(),
        };
        let size = d.size();
        let next = ins_eip.wrapping_add(d.len as u32);

        match d.op {
            Op::Alu(op) => {
                let a = self.opnd_read(&d.ops[0], ins_eip);
                let b = self.opnd_read(&d.ops[1], ins_eip);
                let opv = i64(self, alu_id(op) as i64);
                let a64 = self.fb.ins().uextend(types::I64, a);
                let b64 = self.fb.ins().uextend(types::I64, b);
                let bv = i64(self, bits_id(bits) as i64);
                let r = self.call(self.frs.alu, &[opv, a64, b64, bv]);
                if op != crate::decode::AluOp::Cmp {
                    let r32 = self.fb.ins().ireduce(types::I32, r);
                    self.opnd_write(&d.ops[0], r32, ins_eip);
                }
            }
            Op::Inc | Op::Dec => {
                let v = self.opnd_read(&d.ops[0], ins_eip);
                let inc = i64(self, if d.op == Op::Inc { 1 } else { 0 });
                let v64 = self.fb.ins().uextend(types::I64, v);
                let bv = i64(self, bits_id(bits) as i64);
                let r = self.call(self.frs.incdec, &[inc, v64, bv]);
                let r32 = self.fb.ins().ireduce(types::I32, r);
                self.opnd_write(&d.ops[0], r32, ins_eip);
            }
            Op::Neg => {
                let v = self.opnd_read(&d.ops[0], ins_eip);
                let v64 = self.fb.ins().uextend(types::I64, v);
                let bv = i64(self, bits_id(bits) as i64);
                let r = self.call(self.frs.neg, &[v64, bv]);
                let r32 = self.fb.ins().ireduce(types::I32, r);
                self.opnd_write(&d.ops[0], r32, ins_eip);
            }
            Op::Not => {
                let v = self.opnd_read(&d.ops[0], ins_eip);
                let mask = match bits {
                    Bits::B8 => 0xFF,
                    Bits::B16 => 0xFFFF,
                    Bits::B32 => u32::MAX,
                };
                let not = self.fb.ins().bnot(v);
                let maskv = i32(self, mask as i64);
                let r = self.fb.ins().band(not, maskv);
                self.opnd_write(&d.ops[0], r, ins_eip);
            }
            Op::Test => {
                let a = self.opnd_read(&d.ops[0], ins_eip);
                let b = self.opnd_read(&d.ops[1], ins_eip);
                let a64 = self.fb.ins().uextend(types::I64, a);
                let b64 = self.fb.ins().uextend(types::I64, b);
                let bv = i64(self, bits_id(bits) as i64);
                self.call(self.frs.test, &[a64, b64, bv]);
            }
            Op::Shift(kind, by) => {
                let v = self.opnd_read(&d.ops[0], ins_eip);
                let count = match by {
                    ShiftBy::One => i32(self, 1),
                    ShiftBy::Cl => reg_val(self, 1, Bits::B8),
                    ShiftBy::Imm(c) => i32(self, c as i64),
                };
                let kv = i64(self, shift_id(kind) as i64);
                let v64 = self.fb.ins().uextend(types::I64, v);
                let c64 = self.fb.ins().uextend(types::I64, count);
                let bv = i64(self, bits_id(bits) as i64);
                let r = self.call(self.frs.shift, &[kv, v64, c64, bv]);
                let r32 = self.fb.ins().ireduce(types::I32, r);
                self.opnd_write(&d.ops[0], r32, ins_eip);
            }
            Op::Mov => {
                let src = self.opnd_read(&d.ops[1], ins_eip);
                self.opnd_write(&d.ops[0], src, ins_eip);
            }
            Op::Movzx(from8) | Op::Movsx(from8) => {
                let src = self.opnd_read(&d.ops[1], ins_eip);
                let ext = if matches!(d.op, Op::Movsx(_)) {
                    let small = if from8 {
                        self.fb.ins().ireduce(types::I8, src)
                    } else {
                        self.fb.ins().ireduce(types::I16, src)
                    };
                    self.fb.ins().sextend(types::I32, small)
                } else {
                    src
                };
                self.opnd_write(&d.ops[0], ext, ins_eip);
            }
            Op::Lea => {
                if let Opnd::Mem(m, _) = d.ops[1] {
                    let ea = eff_addr(self, &m);
                    self.opnd_write(&d.ops[0], ea, ins_eip);
                }
            }
            Op::Xchg => {
                let a = self.opnd_read(&d.ops[0], ins_eip);
                let b = self.opnd_read(&d.ops[1], ins_eip);
                self.opnd_write(&d.ops[0], b, ins_eip);
                self.opnd_write(&d.ops[1], a, ins_eip);
            }
            Op::Setcc(c) => {
                let cond = self.cond_call(c);
                let one = widen_bool(self, cond);
                self.opnd_write(&d.ops[0], one, ins_eip);
            }
            Op::Cmov(c) => {
                let cond = self.cond_call(c);
                let src = self.opnd_read(&d.ops[1], ins_eip);
                let dst = self.opnd_read(&d.ops[0], ins_eip);
                let sel = self.fb.ins().select(cond, src, dst);
                self.opnd_write(&d.ops[0], sel, ins_eip);
            }
            Op::Push => {
                let v = self.opnd_read(&d.ops[0], ins_eip);
                let sz = i64(self, size.bytes() as i64);
                let v64 = self.fb.ins().uextend(types::I64, v);
                let r = self.call(self.frs.push, &[sz, v64]);
                self.fault_branch(r, false, ins_eip);
            }
            Op::Pop => {
                let sz = i64(self, size.bytes() as i64);
                let r = self.call(self.frs.pop, &[sz]);
                self.fault_branch(r, true, ins_eip);
                let r32 = self.fb.ins().ireduce(types::I32, r);
                self.opnd_write(&d.ops[0], r32, ins_eip);
            }
            Op::Jcc(c) => {
                let cond = self.cond_call(c);
                self.branch_cc(cond, d.rel_target(next), next);
            }
            Op::Jump { far: false } => {
                let target = match d.ops[0] {
                    Opnd::Rel { disp } => i32(self, (next as i64 + disp as i64) as i64),
                    Opnd::Reg(r, b) => reg_val(self, r, b),
                    Opnd::Mem(m, b) => self.mem_load(&m, b, 0, ins_eip),
                    _ => i32(self, next as i64),
                };
                let z = i32(self, 0);
                self.fb
                    .ins()
                    .jump(self.epi, &[BlockArg::from(target), BlockArg::from(z)]);
                self.terminated = true;
            }
            Op::Call { far: false } => {
                let target = match d.ops[0] {
                    Opnd::Rel { disp } => i32(self, (next as i64 + disp as i64) as i64),
                    Opnd::Reg(r, b) => reg_val(self, r, b),
                    Opnd::Mem(m, b) => self.mem_load(&m, b, 0, ins_eip),
                    _ => i32(self, next as i64),
                };
                let sz = i64(self, size.bytes() as i64);
                let nx = i64(self, next as i64);
                let r = self.call(self.frs.push, &[sz, nx]);
                self.fault_branch(r, false, ins_eip);
                let z = i32(self, 0);
                self.fb
                    .ins()
                    .jump(self.epi, &[BlockArg::from(target), BlockArg::from(z)]);
                self.terminated = true;
            }
            Op::Ret { far: false, imm } => {
                let sz = i64(self, size.bytes() as i64);
                let r = self.call(self.frs.pop, &[sz]);
                self.fault_branch(r, true, ins_eip);
                let ip = self.fb.ins().ireduce(types::I32, r);
                if let Some(imm) = imm {
                    let iv = i64(self, imm as i64);
                    self.call(self.frs.stack_add, &[iv]);
                }
                let z = i32(self, 0);
                self.fb
                    .ins()
                    .jump(self.epi, &[BlockArg::from(ip), BlockArg::from(z)]);
                self.terminated = true;
            }
            Op::Loop(cond) => {
                let ecx = self.fb.use_var(self.vregs[1]);
                let one = i32(self, -1);
                let dec = self.fb.ins().iadd(ecx, one);
                let nz = if d.o16 {
                    let low = self.fb.ins().ireduce(types::I16, dec);
                    let ext = self.fb.ins().uextend(types::I32, low);
                    self.fb.ins().icmp_imm_s(IntCC::NotEqual, ext, 0)
                } else {
                    self.fb.ins().icmp_imm_s(IntCC::NotEqual, dec, 0)
                };
                set_reg(self, 1, if d.o16 { Bits::B16 } else { Bits::B32 }, dec);
                let take = match cond {
                    None => nz,
                    Some(k) => {
                        let zf = self.cond_call(Cond::E);
                        let nz_i = widen_bool(self, nz);
                        let zf_i = widen_bool(self, zf);
                        let both = if k == 0xE1 {
                            self.fb.ins().band(nz_i, zf_i)
                        } else {
                            let not_zf = self.fb.ins().bnot(zf_i);
                            self.fb.ins().band(nz_i, not_zf)
                        };
                        self.fb.ins().icmp_imm_s(IntCC::NotEqual, both, 0)
                    }
                };
                self.branch_cc(take, d.rel_target(next), next);
            }
            Op::Jcxz => {
                let cx = if d.a16 {
                    reg_val(self, 1, Bits::B16)
                } else {
                    self.fb.use_var(self.vregs[1])
                };
                let zero = self.fb.ins().icmp_imm_s(IntCC::Equal, cx, 0);
                self.branch_cc(zero, d.rel_target(next), next);
            }
            Op::In => {
                let port = match d.ops[1] {
                    Opnd::Port(p) => i64(self, p as i64),
                    Opnd::Dx => {
                        let dx = reg_val(self, 2, Bits::B16);
                        self.fb.ins().uextend(types::I64, dx)
                    }
                    _ => i64(self, 0),
                };
                let sz = i64(
                    self,
                    match d.ops[0] {
                        Opnd::Acc(b) => b.bytes() as i64,
                        _ => 1,
                    },
                );
                let r = self.call(self.frs.io_r, &[port, sz]);
                let v = self.fb.ins().ireduce(types::I32, r);
                let acc_bits = match d.ops[0] {
                    Opnd::Acc(b) => b,
                    _ => Bits::B8,
                };
                set_reg(self, 0, acc_bits, v);
            }
            Op::Out => {
                let port = match d.ops[0] {
                    Opnd::Port(p) => i64(self, p as i64),
                    Opnd::Dx => {
                        let dx = reg_val(self, 2, Bits::B16);
                        self.fb.ins().uextend(types::I64, dx)
                    }
                    _ => i64(self, 0),
                };
                let v = self.opnd_read(&d.ops[1], ins_eip);
                let sz = i64(
                    self,
                    match d.ops[1] {
                        Opnd::Acc(b) => b.bytes() as i64,
                        _ => 1,
                    },
                );
                let v64 = self.fb.ins().uextend(types::I64, v);
                self.call(self.frs.io_w, &[port, sz, v64]);
            }
            Op::Xlat => {
                let al = reg_val(self, 0, Bits::B8);
                let bx = reg_val(self, 3, Bits::B16);
                let off = self.fb.ins().iadd(bx, al);
                let off16 = self.fb.ins().ireduce(types::I16, off);
                let off = self.fb.ins().uextend(types::I32, off16);
                let v = self.mem_load_off(Seg::Ds, off, Bits::B8, 0, ins_eip);
                set_reg(self, 0, Bits::B8, v);
            }
            Op::Movs(_)
            | Op::Stos(_)
            | Op::Lods(_)
            | Op::Cmps(_)
            | Op::Scas(_)
            | Op::Ins(_)
            | Op::Outs(_) => {
                let op_id = i64(
                    self,
                    match d.op {
                        Op::Movs(_) => 0,
                        Op::Stos(_) => 1,
                        Op::Lods(_) => 2,
                        Op::Cmps(_) => 3,
                        Op::Scas(_) => 4,
                        Op::Ins(_) => 5,
                        _ => 6,
                    },
                );
                let rep = i64(
                    self,
                    match d.rep {
                        Rep::Z => 1,
                        Rep::NZ => 2,
                        Rep::None => 0,
                    },
                );
                let sb = match d.op {
                    Op::Movs(b)
                    | Op::Stos(b)
                    | Op::Lods(b)
                    | Op::Cmps(b)
                    | Op::Scas(b)
                    | Op::Ins(b)
                    | Op::Outs(b) => b,
                    _ => Bits::B8,
                };
                let bv = i64(self, bits_id(sb) as i64);
                let a16 = i64(self, d.a16 as i64);
                self.call(self.frs.string, &[op_id, rep, bv, a16]);
            }
            Op::Cli | Op::Sti | Op::Cld | Op::Std | Op::Clc | Op::Stc | Op::Cmc | Op::Clts => {
                let flag_id = i64(
                    self,
                    match d.op {
                        Op::Cli => 0,
                        Op::Sti => 1,
                        Op::Cld => 2,
                        Op::Std => 3,
                        Op::Clc => 4,
                        Op::Stc => 5,
                        Op::Cmc => 6,
                        _ => 7,
                    },
                );
                self.call(self.frs.flagop, &[flag_id]);
            }
            Op::Nop | Op::Wait => {}
            Op::Salc => {
                let cf = self.cond_call(Cond::B);
                let one = widen_bool(self, cf);
                let mask = i32(self, 0xFF);
                let m = self.fb.ins().imul(one, mask);
                set_reg(self, 0, Bits::B8, m);
            }
            _ => {
                let eip = i32(self, ins_eip as i64);
                self.fb.ins().jump(self.fault, &[BlockArg::from(eip)]);
                self.terminated = true;
            }
        }
    }
}
