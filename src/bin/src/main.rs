//! vibex86 — a low-level i386 PC emulator with interpreter and Cranelift JIT.
//!
//! Firmware: the *real* Bochs BIOS (`BIOS-bochs-latest`) and the *real* Bochs
//! VGA BIOS (`VGABIOS-lgpl-latest`), distributed with Bochs under the LGPL.
//! Nothing here is hand-written firmware.

mod device;
mod vga;

use std::io::{self, Read, Write};

use device::Machine;
use vga::{BIOS_IMAGE, VGA_BIOS_IMAGE, Vga, ansi_frame};
use x86_rs::{StepOut, X86};

use x86_rs::mem::Device;

// ---------------------------------------------------------------------------
// Combined device: chipset + VGA (single Device box for the bus).
// ---------------------------------------------------------------------------

struct Pc {
    machine: Machine,
    vga: Vga,
}

impl Default for Pc {
    fn default() -> Self {
        Self::new()
    }
}

impl Pc {
    fn new() -> Self {
        Self {
            machine: Machine::new(),
            vga: Vga::new(),
        }
    }

    fn tick(&mut self, cycles: u64) {
        self.machine.tick(cycles);
    }
}

impl Device for Pc {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn io_read(&mut self, port: u16, size: u8) -> u32 {
        match port {
            0x3B4..=0x3DF | 0x3C0..=0x3CF => self.vga.io_read_raw(port, size),
            _ => self.machine.io_read_raw(port, size),
        }
    }
    fn io_write(&mut self, port: u16, size: u8, data: u32) {
        match port {
            0x3B4..=0x3DF | 0x3C0..=0x3CF => self.vga.io_write_raw(port, size, data),
            _ => self.machine.io_write_raw(port, size, data),
        }
    }
    fn ack_irq(&mut self) -> Option<u8> {
        self.machine.ack_irq()
    }
    fn tick(&mut self, cycles: u64) {
        self.machine.tick(cycles);
    }
}

impl Machine {
    fn io_read_raw(&mut self, port: u16, _size: u8) -> u32 {
        // Delegates to the private io_read in device.rs
        self.io_read(port, _size)
    }
    fn io_write_raw(&mut self, port: u16, _size: u8, data: u32) {
        self.io_write(port, _size, data);
    }
}

impl Vga {
    fn io_read_raw(&mut self, port: u16, _size: u8) -> u32 {
        self.io_read(port) as u32
    }
    fn io_write_raw(&mut self, port: u16, _size: u8, data: u32) {
        self.io_write(port, data as u8);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_jit = !args.iter().any(|a| a == "--interp");
    let max_steps: u64 = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000_000);
    let quiet = args.iter().any(|a| a == "--quiet");

    println!("vibex86 — i386 PC (interpreter + Cranelift JIT)");
    println!(
        "firmware: {}",
        if use_jit {
            "Bochs BIOS + VGABIOS"
        } else {
            "Bochs BIOS + VGABIOS (interpreter)"
        }
    );

    let mut cpu = setup_machine();

    if use_jit {
        run_jit(&mut cpu, max_steps, quiet);
    } else {
        run_interp(&mut cpu, max_steps, quiet);
    }
}

/// Build the machine: memory map, firmware, devices.
fn setup_machine() -> X86 {
    let mut cpu = X86::new();
    let mem = &mut cpu.mem;

    // The 16 MB RAM is installed by `X86::new()`.
    // Add VGA framebuffer/window at A0000-C0000.
    mem.add_vga(0xA0000);

    // Real Bochs BIOS at E0000 (size 128 KB). The reset vector is at
    // FFFF0 = E0000 + 0x1FFF0; the BIOS also mirrors itself at the top of
    // the 1MB window via its own shadowing code. Bochs maps it at 0xE0000
    // (per `romimage: address=0xfffe0000` on a 4GB bus; on 1MB we install at
    // E0000 which is what the classic Bochs bxromc uses for a 128KB BIOS).
    mem.add_rom(0xE0000, BIOS_IMAGE.to_vec());

    // Real VGA BIOS at C0000 (size 42 KB).
    mem.add_rom(0xC0000, VGA_BIOS_IMAGE.to_vec());

    // Combined device.
    let pc = Pc::new();
    mem.install_device(Box::new(pc));

    cpu.reset();
    cpu.eflags = 0x2;
    cpu
}

fn run_interp(cpu: &mut X86, max_steps: u64, quiet: bool) {
    let mut steps = 0u64;
    let mut last_render = 0u64;
    loop {
        match cpu.step() {
            StepOut::Ok | StepOut::Interrupt => {}
            StepOut::Error(e) => {
                if !quiet {
                    poll_debug_output(cpu);
                }
                eprintln!(
                    "\n[core stopped] {e:?} at eip={:#x} cs={:#x}",
                    cpu.eip, cpu.seg[1].sel
                );
                print_status(cpu, steps);
                break;
            }
        }
        steps += 1;
        if steps > max_steps {
            eprintln!("\n[stopped after {max_steps} instructions]");
            print_status(cpu, steps);
            break;
        }
        if steps - last_render >= 100_000 {
            last_render = steps;
            if !quiet {
                render_frame(cpu);
            }
        }
        if !quiet {
            poll_debug_output(cpu);
        }
    }
    if !quiet {
        render_frame(cpu);
    }
}

fn run_jit(cpu: &mut X86, max_steps: u64, quiet: bool) {
    let mut jit = match x86_rs::Jit::new() {
        Ok(j) => j,
        Err(e) => {
            eprintln!("JIT unavailable ({e}); falling back to interpreter");
            run_interp(cpu, max_steps, quiet);
            return;
        }
    };
    let mut steps = 0u64;
    let mut last_render = 0u64;
    loop {
        match jit.run(cpu, 4096) {
            Ok(n) => steps += n,
            Err(e) => {
                if !quiet {
                    poll_debug_output(cpu);
                }
                eprintln!(
                    "\n[core stopped] {e:?} at eip={:#x} cs={:#x}",
                    cpu.eip, cpu.seg[1].sel
                );
                eprintln!(
                    "JIT stats: {} blocks compiled, {} recompiled, {} native insns",
                    jit.blocks_compiled, jit.blocks_recompiled, jit.instructions_run
                );
                break;
            }
        }
        if steps > max_steps {
            eprintln!("\n[stopped after {max_steps} instructions]");
            eprintln!(
                "JIT stats: {} blocks compiled, {} recompiled, {} native insns",
                jit.blocks_compiled, jit.blocks_recompiled, jit.instructions_run
            );
            break;
        }
        if steps - last_render >= 100_000 {
            last_render = steps;
            if !quiet {
                render_frame(cpu);
            }
        }
        if !quiet {
            poll_debug_output(cpu);
        }
    }
    if !quiet {
        render_frame(cpu);
    }
}

fn render_frame(cpu: &mut X86) {
    // Draw the current VGA text screen to the terminal.
    let mut buf = [0u8; 0x20000];
    cpu.mem.copy_from_phys(0xA0000, &mut buf);
    // Rebuild a Vga view from the raw bytes.
    let mut vga = Vga::new();
    vga.render(&buf);
    let s = ansi_frame(&vga);
    print!("\x1b[2J\x1b[H{s}");
    io::stdout().flush().ok();
}

fn poll_debug_output(cpu: &mut X86) {
    let Some(device) = cpu.mem.device_mut() else {
        return;
    };
    let Some(pc) = device.as_any_mut().downcast_mut::<Pc>() else {
        return;
    };
    if pc.machine.debug_bytes.is_empty() {
        return;
    }
    let out = String::from_utf8_lossy(&pc.machine.debug_bytes);
    print!("{out}");
    pc.machine.debug_bytes.clear();
    flush_stdout();
}

fn print_status(cpu: &X86, steps: u64) {
    if steps == 0 {
        return;
    }
    println!(
        "registers: eax={:#010x} ecx={:#010x} edx={:#010x} ebx={:#010x} esi={:#010x} edi={:#010x} ebp={:#010x} esp={:#010x}",
        cpu.gpr[0],
        cpu.gpr[1],
        cpu.gpr[2],
        cpu.gpr[3],
        cpu.gpr[6],
        cpu.gpr[7],
        cpu.gpr[5],
        cpu.gpr[4]
    );
    println!(
        "cs={:#06x} eip={:#x} eflags={:#x} cr0={:#010x} cr3={:#010x}",
        cpu.seg[1].sel, cpu.eip, cpu.eflags, cpu.cr[0], cpu.cr[3]
    );
    println!(
        "bda: shutdown={:#04x} halt_count={:#06x} b800[0..4]={:02x?}",
        cpu.mem.phys_read8(0x4B0),
        cpu.mem.phys_read16(0x73C),
        [
            cpu.mem.phys_read8(0xB8000),
            cpu.mem.phys_read8(0xB8001),
            cpu.mem.phys_read8(0xB8002),
            cpu.mem.phys_read8(0xB8003),
        ]
    );
    let ss_base = cpu.seg[2].desc.base;
    let bp = cpu.gpr[5] as u16 as u32;
    let sp = cpu.gpr[4] as u16 as u32;
    println!(
        "stack: ss={:#06x} bp={:#06x} sp={:#06x} [bp..]={:04x?} [sp..]={:04x?}",
        cpu.seg[2].sel,
        bp,
        sp,
        [
            cpu.mem.phys_read16(ss_base + bp),
            cpu.mem.phys_read16(ss_base + bp + 2),
            cpu.mem.phys_read16(ss_base + bp + 4),
            cpu.mem.phys_read16(ss_base + bp + 6),
        ],
        [
            cpu.mem.phys_read16(ss_base + sp),
            cpu.mem.phys_read16(ss_base + sp + 2),
            cpu.mem.phys_read16(ss_base + sp + 4),
            cpu.mem.phys_read16(ss_base + sp + 6),
        ],
    );
    for vec in 0..=0xffu32 {
        let off = cpu.mem.phys_read16(vec * 4);
        let seg = cpu.mem.phys_read16(vec * 4 + 2);
        if seg == cpu.seg[1].sel && off as u32 == cpu.eip {
            println!("ivt: vector {vec:#04x} -> {seg:04x}:{off:04x}");
        }
    }
}

fn flush_stdout() {
    io::stdout().flush().ok();
}
