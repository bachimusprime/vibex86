//! Chipset devices: dual 8259 PIC, 8254 PIT, MC146818 CMOS + NMI mask,
//! 8237 DMA, 8042 keyboard controller, UART16550 serial, and the Bochs
//! debug-readout port used by the real Bochs BIOS.
//!
//! The real Bochs BIOS (`BIOS-bochs-latest`) probes all of these during POST
//! (PIT channel 2 rate, CMOS shutdown status, 8042 reset command, ...), so
//! their port behavior matters.

use x86_rs::mem::Device;

/// Interrupt request line -> PIC vector mapping (BIOS remaps master to 0x08,
/// slave to 0x70).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Irq {
    Timer = 0,
    Keyboard = 1,
    Serial2 = 3,
    Serial1 = 4,
    PicSlave = 2,
    Fdc = 6,
    Ide = 14,
    Ps2Mouse = 12,
    Rtc = 8,
}

// ---------------------------------------------------------------------------
// 8259 PIC
// ---------------------------------------------------------------------------

struct PicChip {
    base: u8,
    irr: u8,
    isr: u8,
    imr: u8,
    /// Next ICW index (0 = idle OCW).
    icw_step: u8,
    auto_eoi: bool,
}

impl Default for PicChip {
    fn default() -> Self {
        Self {
            base: 0,
            irr: 0,
            isr: 0,
            imr: 0xFF, // 8259 resets masked
            icw_step: 0,
            auto_eoi: false,
        }
    }
}

impl PicChip {
    fn new(base: u8) -> Self {
        Self {
            base,
            ..Self::default()
        }
    }

    fn raise(&mut self, irq: u8) {
        self.irr |= 1 << irq;
    }

    /// Highest priority unmasked pending request.
    fn lowest_i(&self) -> Option<u8> {
        let pending = self.irr & !self.imr;
        if pending == 0 {
            return None;
        }
        for i in 0..8 {
            if pending & (1 << i) != 0 {
                return Some(i);
            }
        }
        None
    }

    fn ack(&mut self) -> Option<u8> {
        let i = self.lowest_i()?;
        self.irr &= !(1 << i);
        self.isr |= 1 << i;
        Some(self.base + i)
    }

    fn eoi(&mut self) {
        let isr = self.isr;
        if isr != 0 {
            self.isr &= !(isr & isr.wrapping_neg());
        }
    }

    fn write_cmd(&mut self, val: u8) {
        if val & 0x10 != 0 {
            // ICW1: initialize.
            self.icw_step = 1;
            self.imr = 0xFF;
            self.irr = 0;
            self.isr = 0;
            self.auto_eoi = false;
            return;
        }
        if val & 0x08 == 0 && val & 0x04 != 0 && self.icw_step == 0 {
            // OCW2 (EOI etc). Only auto-EOI path: non-specific EOI clears the
            // highest priority in-service.
            if val & 0x20 != 0 {
                let isr = self.isr;
                self.isr &= !(isr & isr.wrapping_neg());
            }
            return;
        }
        // val & 0x08 could be OCW3; the real BIOS writes ICW2/3/4 to the
        // data port after ICW1.
    }

    fn write_data(&mut self, val: u8) {
        // ICW2/3/4 sequence after ICW1, otherwise OCW1 interrupt mask.
        match self.icw_step {
            1 => {
                self.base = val & 0xF8;
                self.icw_step = 2;
            }
            2 => {
                // ICW3 (cascade mask) or ICW4 if single.
                self.icw_step = 3;
            }
            3 => {
                self.auto_eoi = val & 0x02 != 0;
                self.icw_step = 0;
            }
            _ => {
                self.imr = val;
                self.icw_step = 0;
            }
        }
    }
}

/// 8259 PIC pair.
pub struct Pic {
    master: PicChip,
    slave: PicChip,
}

impl Default for Pic {
    fn default() -> Self {
        Self::new()
    }
}

impl Pic {
    pub fn new() -> Self {
        Self {
            master: PicChip::new(0x08),
            slave: PicChip::new(0x70),
        }
    }

    /// Raise an IRQ line (0..15).
    pub fn raise(&mut self, irq: u8) {
        let chip = if irq < 8 {
            &mut self.master
        } else {
            &mut self.slave
        };
        chip.raise(irq & 7);
    }

    /// INTA: return the vector for the highest priority request.
    pub fn ack(&mut self) -> Option<u8> {
        if self.slave.lowest_i().is_some() {
            let v = self.slave.ack();
            self.master.irr &= !(1 << 2);
            self.master.isr |= 1 << 2;
            return v;
        }
        self.master.ack()
    }

    /// End-of-interrupt (non-specific).
    pub fn eoi(&mut self) {
        if self.master.isr & (1 << 2) != 0 {
            self.slave.eoi();
        }
        self.master.eoi();
    }

    fn read(&mut self, port: u16) -> u8 {
        match port {
            0x20 => self.master.isr, // reading ISR; BIOS mostly reads IMR
            0x21 => self.master.imr,
            0xA0 => self.slave.isr,
            0xA1 => self.slave.imr,
            _ => 0xFF,
        }
    }

    fn write(&mut self, port: u16, val: u8) {
        match port {
            0x20 => self.master.write_cmd(val),
            0x21 => self.master.write_data(val),
            0xA0 => self.slave.write_cmd(val),
            0xA1 => self.slave.write_data(val),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// 8254 PIT
// ---------------------------------------------------------------------------

pub struct Pit {
    count: [u16; 3],
    reload: [u16; 3],
    /// Remaining cycles until the next countdown.
    phases: [u64; 3],
    gate: [bool; 3],
    /// Channel 2 output level (toggles on expiry); read via port 0x61 bit 5.
    pub out2: bool,
    latch: [u8; 3],
    latched: [bool; 3],
    /// True when channel 0 expired (drives the timer IRQ).
    pub timer_fired: bool,
    /// Write pointer for 16-bit loads.
    low_byte: [bool; 3],
}

impl Default for Pit {
    fn default() -> Self {
        Self::new()
    }
}

impl Pit {
    pub fn new() -> Self {
        Self {
            count: [0; 3],
            reload: [0; 3],
            phases: [0; 3],
            gate: [true; 3],
            out2: false,
            latch: [0; 3],
            latched: [false; 3],
            timer_fired: false,
            low_byte: [true; 3],
        }
    }

    fn read(&mut self, port: u16) -> u8 {
        let ch = (port - 0x40) as usize;
        if ch >= 3 {
            return 0xFF;
        }
        if self.latched[ch] {
            self.latched[ch] = false;
            return self.latch[ch];
        }
        // Best effort: return the running count low byte.
        (self.count[ch] & 0xFF) as u8
    }

    fn write(&mut self, port: u16, val: u8) {
        let ch = (port - 0x40) as usize;
        if ch >= 3 {
            return;
        }
        if self.low_byte[ch] {
            self.count[ch] = (self.count[ch] & 0xFF00) | val as u16;
        } else {
            self.count[ch] = (self.count[ch] & 0x00FF) | ((val as u16) << 8);
            self.reload[ch] = self.count[ch];
            if self.reload[ch] != 0 {
                self.phases[ch] = self.reload[ch] as u64;
            }
        }
        self.low_byte[ch] = !self.low_byte[ch];
    }

    fn cmd(&mut self, val: u8) {
        if val & 0xC0 == 0xC0 {
            // Read-back / latch command.
            if val & 0x02 == 0 {
                self.latched[0] = true;
                self.latch[0] = (self.count[0] & 0xFF) as u8;
            }
            if val & 0x04 == 0 {
                self.latched[1] = true;
                self.latch[1] = (self.count[1] & 0xFF) as u8;
            }
            if val & 0x08 == 0 {
                self.latched[2] = true;
                self.latch[2] = (self.count[2] & 0xFF) as u8;
            }
            return;
        }
        let ch = ((val >> 6) & 3) as usize;
        if ch == 3 {
            return;
        }
        // Select mode / access mode; we always use the low/high toggle.
        let access = (val >> 4) & 3;
        if access == 0 {
            return; // latch count
        }
        self.low_byte[ch] = true;
        self.count[ch] = 0;
    }

    pub fn tick(&mut self, cycles: u64) {
        self.timer_fired = false;
        for ch in 0..3 {
            if !self.gate[ch] || self.phases[ch] == 0 {
                continue;
            }
            if cycles >= self.phases[ch] {
                self.phases[ch] = self.reload[ch] as u64;
                if ch == 2 {
                    self.out2 = !self.out2;
                }
                if ch == 0 {
                    self.timer_fired = true;
                }
            } else {
                self.phases[ch] -= cycles;
                // keep the phase >= 1 so a partial tick does not spuriously
                // fire; subtract at least 1 if cycles >= 1 and phase given.
            }
            if cycles > 0 && self.phases[ch] > 0 {
                self.phases[ch] = self.phases[ch].saturating_sub(1);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CMOS + NMI mask
// ---------------------------------------------------------------------------

pub struct Cmos {
    pub regs: [u8; 0x80],
    sel: u8,
    pub nmi_mask: u8,
}

impl Default for Cmos {
    fn default() -> Self {
        Self::new()
    }
}

impl Cmos {
    pub fn new() -> Self {
        let mut regs = [0u8; 0x80];
        // RTC time (rough values; the Bochs BIOS mostly polls the UIP bit).
        regs[0x00] = 0x07;
        regs[0x02] = 0x06;
        regs[0x04] = 0x06;
        regs[0x07] = 0x01;
        regs[0x08] = 0x01;
        regs[0x09] = 26;
        regs[0x0A] = 0x26;
        regs[0x0B] = 0x02;
        regs[0x0C] = 0;
        regs[0x0D] = 0x80;
        // Shutdown status: 0 = normal (the 8042/BIOS may clear it on reset).
        regs[0x0F] = 0x00;
        // Drive types: A = 1.44 MB, B = none.
        regs[0x10] = 0x10;
        regs[0x12] = 0x10;
        // Base memory 640K.
        regs[0x15] = 0x00;
        regs[0x16] = 0x28;
        regs[0x17] = 0x00;
        // Extended memory 16128 KB (16 MB - 640K).
        regs[0x18] = 0x00;
        regs[0x19] = 0x3F;
        regs[0x30] = 0x00;
        regs[0x31] = 0x3F;
        regs[0x32] = 0x00;
        regs[0x33] = 0x3F;
        Self {
            regs,
            sel: 0,
            nmi_mask: 1, // NMI enabled by default
        }
    }

    fn read(&mut self, port: u16) -> u8 {
        match port {
            0x70 => self.nmi_mask | 0x80,
            0x71 => self.regs[self.sel as usize],
            _ => 0xFF,
        }
    }

    fn write(&mut self, port: u16, val: u8) {
        match port {
            0x70 => {
                self.sel = val & 0x7F;
                self.nmi_mask = val >> 7;
            }
            0x71 => {
                let sel = self.sel as usize;
                if sel < self.regs.len() {
                    self.regs[sel] = val;
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// 8237 DMA
// ---------------------------------------------------------------------------

pub struct Dma {
    pub page_regs: [u8; 0x10],
    pub mode: u8,
    pub mask: u8,
}

impl Default for Dma {
    fn default() -> Self {
        Self::new()
    }
}

impl Dma {
    pub fn new() -> Self {
        Self {
            page_regs: [0; 0x10],
            mode: 0,
            mask: 0,
        }
    }

    fn read(&mut self, port: u16) -> u8 {
        match port {
            0x08 => (self.mask & 0xF) | 0xF0,
            0xDA => 0x00,
            _ => 0xFF,
        }
    }

    fn write(&mut self, _port: u16, val: u8) {
        self.mode = val;
        let _ = val;
    }
}

// ---------------------------------------------------------------------------
// 8042 keyboard controller
// ---------------------------------------------------------------------------

pub struct Ps2Kbd {
    pub command: u8,
    pub command_byte: u8,
    pub status: u8,
    pub data: u8,
    pub has_data: bool,
    pub scancode_queue: std::collections::VecDeque<u8>,
}

impl Default for Ps2Kbd {
    fn default() -> Self {
        Self::new()
    }
}

impl Ps2Kbd {
    pub fn new() -> Self {
        Self {
            command: 0,
            command_byte: 0,
            status: 0x20, // self-test passed
            data: 0,
            has_data: false,
            scancode_queue: std::collections::VecDeque::new(),
        }
    }

    pub fn inject(&mut self, scancode: u8) {
        self.scancode_queue.push_back(scancode);
        self.status |= 0x01;
    }

    fn read(&mut self, port: u16) -> u8 {
        match port {
            0x60 => {
                if let Some(code) = self.scancode_queue.pop_front() {
                    self.data = code;
                }
                self.status &= !0x01;
                self.has_data = false;
                if !self.scancode_queue.is_empty() {
                    self.status |= 0x01;
                    self.has_data = true;
                }
                self.data
            }
            0x64 => self.status,
            _ => 0xFF,
        }
    }

    fn write(&mut self, port: u16, val: u8) {
        match port {
            0x60 => {
                if self.command == 0x60 {
                    self.command_byte = val;
                    self.command = 0;
                } else {
                    // 8042 data port: A20 line / keyboard commands.
                    self.data = val;
                    match val {
                        0xFF => {
                            self.scancode_queue.push_back(0xFA);
                            self.scancode_queue.push_back(0xAA);
                            self.status |= 0x01;
                            self.has_data = true;
                        }
                        0xF4 | 0xF5 | 0xF6 | 0xED | 0xF3 => {
                            self.scancode_queue.push_back(0xFA);
                            self.status |= 0x01;
                            self.has_data = true;
                        }
                        _ => {}
                    }
                }
            }
            0x64 => {
                self.command = val;
                match val {
                    0x20 => {
                        // Read command byte.
                        self.data = self.command_byte;
                        self.has_data = true;
                        self.status |= 0x01;
                    }
                    0x60 => {
                        // Next write to port 0x60 sets the command byte.
                    }
                    0xAA => {
                        // self-test: reply 0x55 in the output buffer.
                        self.data = 0x55;
                        self.has_data = true;
                        self.status |= 0x01;
                    }
                    0xAB | 0xA9 => {
                        // Interface tests: 0 means OK.
                        self.data = 0x00;
                        self.has_data = true;
                        self.status |= 0x01;
                    }
                    0xAD | 0xAE | 0xA7 | 0xA8 => {
                        // Disable/enable keyboard or auxiliary device.
                    }
                    0xFE => {
                        // pulse reset.
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// UART16550A serial
// ---------------------------------------------------------------------------

pub struct Serial {
    pub thr: u8,
    pub ier: u8,
    pub lcr: u8,
    pub mcr: u8,
    pub lsr: u8,
    pub msr: u8,
    dlab: bool,
    pub out: Vec<u8>,
}

impl Default for Serial {
    fn default() -> Self {
        Self::new()
    }
}

impl Serial {
    pub fn new() -> Self {
        Self {
            thr: 0,
            ier: 0,
            lcr: 0x03,
            mcr: 0,
            lsr: 0x60,
            msr: 0x10,
            dlab: false,
            out: Vec::new(),
        }
    }

    fn read(&mut self, port: u16) -> u8 {
        match port {
            0x3FA => 0x01,
            0x3FB => self.lcr,
            0x3FC => self.mcr,
            0x3FD => self.lsr,
            0x3FE => self.msr,
            _ => 0,
        }
    }

    fn write(&mut self, port: u16, val: u8) {
        match port {
            0x3FB => {
                self.lcr = val;
                self.dlab = val & 0x80 != 0;
            }
            0x3F8 if !self.dlab => {
                self.thr = val;
                self.lsr |= 0x40;
                self.out.push(val);
            }
            0x3F9 => self.ier = val,
            0x3FC => self.mcr = val,
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Machine device implementation
// ---------------------------------------------------------------------------

pub struct Machine {
    pub pic: Pic,
    pub pit: Pit,
    pub cmos: Cmos,
    pub dma: Dma,
    pub ps2: Ps2Kbd,
    pub serial: Serial,
    /// Port E9 debug output from the Bochs BIOS.
    pub debug_bytes: Vec<u8>,
    pub total_cycles: u64,
}

impl Default for Machine {
    fn default() -> Self {
        Self::new()
    }
}

impl Machine {
    pub fn new() -> Self {
        Self {
            pic: Pic::new(),
            pit: Pit::new(),
            cmos: Cmos::new(),
            dma: Dma::new(),
            ps2: Ps2Kbd::new(),
            serial: Serial::new(),
            debug_bytes: Vec::new(),
            total_cycles: 0,
        }
    }

    fn io_read(&mut self, port: u16, _size: u8) -> u32 {
        match port {
            0x20 | 0x21 | 0xA0 | 0xA1 => self.pic.read(port) as u32,
            0x40..=0x43 => self.pit.read(port) as u32,
            0x61 => {
                let mut v = 0u8;
                if self.pit.out2 {
                    v |= 0x20;
                }
                (v | 0x30) as u32
            }
            0x70 | 0x71 => self.cmos.read(port) as u32,
            0x08 | 0x09 | 0x0A | 0x0B | 0x0C | 0x0D | 0xDA => self.dma.read(port) as u32,
            0x60 | 0x64 => self.ps2.read(port) as u32,
            0x3F8 => self.serial.read(port) as u32,
            0x3FA..=0x3FE => self.serial.read(port) as u32,
            0x402 | 0x403 | 0xE9 => 0,
            _ => 0xFF,
        }
    }

    fn io_write(&mut self, port: u16, _size: u8, data: u32) {
        let v = data as u8;
        match port {
            0x20 | 0x21 | 0xA0 | 0xA1 => self.pic.write(port, v),
            0x40..=0x43 => self.pit.write(port, v),
            0x70 | 0x71 => self.cmos.write(port, v),
            0x08 | 0x0A | 0x0B | 0x0C | 0x0D | 0xDA => self.dma.write(port, v),
            0x60 | 0x64 => self.ps2.write(port, v),
            0x3F8 => self.serial.write(port, v),
            0x3FA..=0x3FE => self.serial.write(port, v),
            0x402 | 0x403 | 0xE9 => {
                self.debug_bytes.push(v);
            }
            _ => {}
        }
    }

    fn ack_irq(&mut self) -> Option<u8> {
        self.pic.ack()
    }

    fn tick(&mut self, cycles: u64) {
        self.total_cycles += cycles;
        self.pit.tick(cycles);
        if self.pit.timer_fired {
            self.pic.raise(Irq::Timer as u8);
        }
    }
}

impl Device for Machine {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn io_read(&mut self, port: u16, size: u8) -> u32 {
        self.io_read(port, size)
    }
    fn io_write(&mut self, port: u16, size: u8, data: u32) {
        self.io_write(port, size, data)
    }
    fn ack_irq(&mut self) -> Option<u8> {
        self.ack_irq()
    }
    fn tick(&mut self, cycles: u64) {
        self.tick(cycles)
    }
}
