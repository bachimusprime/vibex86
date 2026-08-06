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
    let mut pc = Pc::new();
    if let Ok(image) = std::fs::read("Windows 98 Second Edition Boot.img") {
        pc.machine.load_floppy(image);
    }
    mem.install_device(Box::new(pc));

    cpu.reset();
    cpu.eflags = 0x2;
    cpu
}

fn run_interp(cpu: &mut X86, max_steps: u64, quiet: bool) {
    let mut steps = 0u64;
    let mut last_render = 0u64;
    loop {
        trace_bios_diskette(cpu, steps);
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
        service_floppy_dma(cpu);
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

fn trace_bios_diskette(cpu: &X86, steps: u64) {
    if std::env::var_os("VIBEX86_TRACE_DISKETTE").is_none() {
        return;
    }
    match cpu.eip {
        0x088C | 0x888C => {
            eprintln!(
                "trace {steps}: int13 ah={:#04x} al={:#04x} ch={:#04x} cl={:#04x} dh={:#04x} dl={:#04x} es={:#06x} bx={:#06x} bp={:#06x} sp={:#06x}",
                (cpu.gpr[0] >> 8) as u8,
                cpu.gpr[0] as u8,
                (cpu.gpr[1] >> 8) as u8,
                cpu.gpr[1] as u8,
                (cpu.gpr[2] >> 8) as u8,
                cpu.gpr[2] as u8,
                cpu.seg[0].sel,
                cpu.gpr[3] as u16,
                cpu.gpr[5] as u16,
                cpu.gpr[4] as u16,
            );
        }
        0x8614 | 0x861A | 0x8626 => {
            let ss_base = cpu.seg[2].desc.base;
            let bp = cpu.gpr[5] as u16 as u32;
            eprintln!(
                "trace {steps}: set_cyl cs={:#06x} eip={:#06x} bp={:#06x} sp={:#06x} arg0={:#06x} arg1={:#06x} stack={:04x?}",
                cpu.seg[1].sel,
                cpu.eip,
                bp,
                cpu.gpr[4] as u16,
                cpu.mem.phys_read16(ss_base + bp + 4),
                cpu.mem.phys_read16(ss_base + bp + 6),
                [
                    cpu.mem.phys_read16(ss_base + bp),
                    cpu.mem.phys_read16(ss_base + bp + 2),
                    cpu.mem.phys_read16(ss_base + bp + 4),
                    cpu.mem.phys_read16(ss_base + bp + 6),
                    cpu.mem.phys_read16(ss_base + bp + 8),
                    cpu.mem.phys_read16(ss_base + bp + 10),
                ],
            );
        }
        0x8646 | 0x864D | 0x8654 | 0x866E | 0x8763 => {
            let ss_base = cpu.seg[2].desc.base;
            let bp = cpu.gpr[5] as u16 as u32;
            eprintln!(
                "trace {steps}: int13-handler cs={:#06x} eip={:#06x} bp={:#06x} drive={:#06x} ah={:#04x} fdpt={:04x?} flags_word={:#06x}",
                cpu.seg[1].sel,
                cpu.eip,
                bp,
                cpu.mem.phys_read16(ss_base + bp + 0x0e),
                cpu.mem.phys_read8(ss_base + bp + 0x13),
                [
                    cpu.mem.phys_read16(0x408),
                    cpu.mem.phys_read16(0x40A),
                    cpu.mem.phys_read16(0x40C),
                    cpu.mem.phys_read16(0x40E),
                ],
                cpu.mem.phys_read16(ss_base + bp + 0x1a),
            );
        }
        _ => {}
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
        service_floppy_dma(cpu);
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

fn service_floppy_dma(cpu: &mut X86) {
    let transfer = cpu
        .mem
        .device_mut()
        .and_then(|device| device.as_any_mut().downcast_mut::<Pc>())
        .and_then(|pc| pc.machine.take_floppy_dma_transfer());
    if let Some((addr, data)) = transfer {
        cpu.mem.phys_write_block(addr, &data);
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

fn print_status(cpu: &mut X86, steps: u64) {
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
        "cs={:#06x} eip={:#x} eflags={:#x} halted={} cr0={:#010x} cr3={:#010x}",
        cpu.seg[1].sel, cpu.eip, cpu.eflags, cpu.halted, cpu.cr[0], cpu.cr[3]
    );
    println!(
        "segs: es={:#06x}/{:05x} cs={:#06x}/{:05x} ss={:#06x}/{:05x} ds={:#06x}/{:05x} fs={:#06x}/{:05x} gs={:#06x}/{:05x}",
        cpu.seg[0].sel,
        cpu.seg[0].desc.base,
        cpu.seg[1].sel,
        cpu.seg[1].desc.base,
        cpu.seg[2].sel,
        cpu.seg[2].desc.base,
        cpu.seg[3].sel,
        cpu.seg[3].desc.base,
        cpu.seg[4].sel,
        cpu.seg[4].desc.base,
        cpu.seg[5].sel,
        cpu.seg[5].desc.base,
    );
    let ip_phys = cpu.seg[1].desc.base.wrapping_add(cpu.eip);
    println!(
        "code @cs:eip phys={:#07x} bytes={:02x?}",
        ip_phys,
        [
            cpu.mem.phys_read8(ip_phys),
            cpu.mem.phys_read8(ip_phys + 1),
            cpu.mem.phys_read8(ip_phys + 2),
            cpu.mem.phys_read8(ip_phys + 3),
            cpu.mem.phys_read8(ip_phys + 4),
            cpu.mem.phys_read8(ip_phys + 5),
            cpu.mem.phys_read8(ip_phys + 6),
            cpu.mem.phys_read8(ip_phys + 7),
        ],
    );
    println!(
        "bda: timer_tick={:#010x} int08={:04x}:{:04x} shutdown={:#04x} halt_count={:#06x} b800[0..4]={:02x?}",
        cpu.mem.phys_read32(0x46C),
        cpu.mem.phys_read16(0x08 * 4 + 2),
        cpu.mem.phys_read16(0x08 * 4),
        cpu.mem.phys_read8(0x4B0),
        cpu.mem.phys_read16(0x73C),
        [
            cpu.mem.phys_read8(0xB8000),
            cpu.mem.phys_read8(0xB8001),
            cpu.mem.phys_read8(0xB8002),
            cpu.mem.phys_read8(0xB8003),
        ]
    );
    println!(
        "bda floppy: equip={:#06x} fdpt={:04x?} motor={:#04x} status={:#04x} recal={:#04x} media={:02x?} cyl={:02x?}",
        cpu.mem.phys_read16(0x410),
        [
            cpu.mem.phys_read16(0x408),
            cpu.mem.phys_read16(0x40A),
            cpu.mem.phys_read16(0x40C),
            cpu.mem.phys_read16(0x40E),
        ],
        cpu.mem.phys_read8(0x43F),
        cpu.mem.phys_read8(0x441),
        cpu.mem.phys_read8(0x43E),
        [
            cpu.mem.phys_read8(0x478),
            cpu.mem.phys_read8(0x479),
            cpu.mem.phys_read8(0x47A),
            cpu.mem.phys_read8(0x47B),
        ],
        [
            cpu.mem.phys_read8(0x494),
            cpu.mem.phys_read8(0x495),
            cpu.mem.phys_read8(0x496),
            cpu.mem.phys_read8(0x497),
        ],
    );
    println!(
        "boot buf 0700={:02x?} 0720={:02x?}",
        [
            cpu.mem.phys_read8(0x700),
            cpu.mem.phys_read8(0x701),
            cpu.mem.phys_read8(0x702),
            cpu.mem.phys_read8(0x703),
            cpu.mem.phys_read8(0x704),
            cpu.mem.phys_read8(0x705),
            cpu.mem.phys_read8(0x706),
            cpu.mem.phys_read8(0x707),
            cpu.mem.phys_read8(0x708),
            cpu.mem.phys_read8(0x709),
            cpu.mem.phys_read8(0x70A),
            cpu.mem.phys_read8(0x70B),
            cpu.mem.phys_read8(0x70C),
            cpu.mem.phys_read8(0x70D),
            cpu.mem.phys_read8(0x70E),
            cpu.mem.phys_read8(0x70F),
        ],
        [
            cpu.mem.phys_read8(0x720),
            cpu.mem.phys_read8(0x721),
            cpu.mem.phys_read8(0x722),
            cpu.mem.phys_read8(0x723),
            cpu.mem.phys_read8(0x724),
            cpu.mem.phys_read8(0x725),
            cpu.mem.phys_read8(0x726),
            cpu.mem.phys_read8(0x727),
            cpu.mem.phys_read8(0x728),
            cpu.mem.phys_read8(0x729),
            cpu.mem.phys_read8(0x72A),
            cpu.mem.phys_read8(0x72B),
            cpu.mem.phys_read8(0x72C),
            cpu.mem.phys_read8(0x72D),
            cpu.mem.phys_read8(0x72E),
            cpu.mem.phys_read8(0x72F),
        ],
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
    let prev_bp = cpu.mem.phys_read16(ss_base + bp) as u32;
    if prev_bp != 0 {
        println!(
            "prev frame: bp={:#06x} [bp..]={:04x?}",
            prev_bp,
            [
                cpu.mem.phys_read16(ss_base + prev_bp),
                cpu.mem.phys_read16(ss_base + prev_bp + 2),
                cpu.mem.phys_read16(ss_base + prev_bp + 4),
                cpu.mem.phys_read16(ss_base + prev_bp + 6),
                cpu.mem.phys_read16(ss_base + prev_bp + 8),
                cpu.mem.phys_read16(ss_base + prev_bp + 10),
                cpu.mem.phys_read16(ss_base + prev_bp + 12),
                cpu.mem.phys_read16(ss_base + prev_bp + 14),
            ],
        );
    }
    if let Some(pc) = cpu
        .mem
        .device_mut()
        .and_then(|device| device.as_any_mut().downcast_mut::<Pc>())
    {
        let debug = pc.machine.debug_state();
        println!(
            "pic/pit: cycles={} imr={:#04x} irr={:#04x} isr={:#04x} reload0={} phase0={} fired={} irq0_raised={} pit0_cmds={} pit0_writes={} last_cmd={:#04x} last_write={:#04x}@{}",
            debug.total_cycles,
            debug.pic_imr,
            debug.pic_irr,
            debug.pic_isr,
            debug.pit_reload0,
            debug.pit_phase0,
            debug.pit_timer_fired,
            debug.timer_irqs_raised,
            debug.pit0_cmds,
            debug.pit0_writes,
            debug.last_pit0_cmd,
            debug.last_pit0_write,
            debug.last_pit0_cycle,
        );
    }
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
