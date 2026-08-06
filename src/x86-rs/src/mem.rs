//! Physical address space: RAM/ROM/VGA regions plus the host-device I/O bridge.
//!
//! `Mem` is the single addressable memory map owned by `X86`. Physical
//! accesses go through region lookups (RAM fast path); port I/O is delegated
//! to the machine's `Device` implementation (PIC, PIT, VGA, PS/2, … in
//! `vibex86`).

use std::any::Any;

/// A contiguous physical memory region.
pub trait MemRegion: Send {
    fn len(&self) -> usize;
    fn read8(&self, off: u32) -> u8;
    fn write8(&mut self, off: u32, val: u8);
}

/// Plain byte storage (RAM, or a writable ROM image).
pub struct Ram {
    bytes: Box<[u8]>,
}

impl Ram {
    pub fn new(size: usize) -> Ram {
        Ram {
            bytes: vec![0u8; size].into_boxed_slice(),
        }
    }
    pub fn with_data(data: Vec<u8>) -> Ram {
        Ram {
            bytes: data.into_boxed_slice(),
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl MemRegion for Ram {
    #[inline]
    fn len(&self) -> usize {
        self.bytes.len()
    }
    #[inline]
    fn read8(&self, off: u32) -> u8 {
        self.bytes[off as usize]
    }
    #[inline]
    fn write8(&mut self, off: u32, val: u8) {
        self.bytes[off as usize] = val;
    }
}

/// VGA adapter memory (0xA0000..0xC0000). Read/write; the machine's renderer
/// polls it back to produce frames.
pub struct VgaMem {
    bytes: Box<[u8]>,
}

impl VgaMem {
    pub fn new() -> VgaMem {
        VgaMem {
            bytes: vec![0u8; 0x20000].into_boxed_slice(),
        }
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl Default for VgaMem {
    fn default() -> Self {
        Self::new()
    }
}

impl MemRegion for VgaMem {
    #[inline]
    fn len(&self) -> usize {
        self.bytes.len()
    }
    #[inline]
    fn read8(&self, off: u32) -> u8 {
        self.bytes[off as usize]
    }
    #[inline]
    fn write8(&mut self, off: u32, val: u8) {
        self.bytes[off as usize] = val;
    }
}

/// Host machine device: port I/O, IRQ acknowledge, and per-tick clock.
/// Implemented by `vibex86`'s `Machine`.
pub trait Device: Send + Any {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn io_read(&mut self, port: u16, size: u8) -> u32;
    fn io_write(&mut self, port: u16, size: u8, data: u32);
    /// Returns the interrupt vector for the PIC's highest-priority in-service
    /// request and clears it (interrupt acknowledge cycle).
    fn ack_irq(&mut self) -> Option<u8>;
    /// Called from the run loops to advance time in device land.
    fn tick(&mut self, cycles: u64);
}

/// A mapped physical region.
pub struct Region {
    start: u32,
    end: u32, // exclusive
    memory: Box<dyn MemRegion>,
}

impl Region {
    #[inline]
    fn contains(&self, addr: u32) -> bool {
        addr >= self.start && addr < self.end
    }
}

/// The physical memory map.
pub struct Mem {
    regions: Vec<Region>,
    device: Option<Box<dyn Device>>,
}

impl Default for Mem {
    fn default() -> Self {
        Self::new()
    }
}

impl Mem {
    pub fn new() -> Mem {
        Mem {
            regions: Vec::new(),
            device: None,
        }
    }

    /// Install the default PC layout: 16 MB RAM at 0..0x1000000.
    /// (Devices/BIOS/VGA regions are added by the machine crate on top.)
    pub fn install_default_ram(&mut self) {
        self.add_ram(0, 16 * 1024 * 1024);
    }

    pub fn install_device(&mut self, dev: Box<dyn Device>) {
        self.device = Some(dev);
    }

    #[inline]
    pub fn device_mut(&mut self) -> Option<&mut dyn Device> {
        match self.device.as_mut() {
            Some(device) => Some(device.as_mut()),
            None => None,
        }
    }

    /// Map a generic region of `len` bytes at physical `start`.
    pub fn add_region(&mut self, start: u32, len: usize, region: Box<dyn MemRegion>) {
        debug_assert!(region.len() >= len);
        self.regions.push(Region {
            start,
            end: start + len as u32,
            memory: region,
        });
    }

    /// Map fresh RAM at `start`..`start+size`.
    pub fn add_ram(&mut self, start: u32, size: usize) {
        self.add_region(start, size, Box::new(Ram::new(size)));
    }

    /// Map a ROM image (kept writable — real BIOS code writes to its own image
    /// region while shadowing into RAM; that is benign).
    pub fn add_rom(&mut self, start: u32, data: Vec<u8>) {
        let len = data.len();
        self.add_region(start, len, Box::new(Ram::with_data(data)));
    }

    /// Map a VGA framebuffer region.
    pub fn add_vga(&mut self, start: u32) {
        self.add_region(start, 0x20000, Box::new(VgaMem::new()));
    }

    /// Overwrite bytes in a mapped region (used to load firmware/disk images).
    /// Silently ignores unmapped addresses (matches reads returning 0xFF).
    pub fn load_region(&mut self, start: u32, data: &[u8]) {
        for (i, b) in data.iter().enumerate() {
            self.phys_write8(start.wrapping_add(i as u32), *b);
        }
    }

    /// Copy bytes out of the map (used by the CGA renderer to grab VRAM).
    pub fn copy_from_phys(&mut self, start: u32, out: &mut [u8]) {
        for (i, b) in out.iter_mut().enumerate() {
            *b = self.phys_read8(start.wrapping_add(i as u32));
        }
    }

    #[inline]
    fn find(&self, addr: u32) -> Option<usize> {
        self.regions.iter().rposition(|r| r.contains(addr))
    }

    #[inline]
    pub fn phys_read8(&self, addr: u32) -> u8 {
        match self.find(addr) {
            Some(i) => self.regions[i].memory.read8(addr - self.regions[i].start),
            None => 0xFF,
        }
    }

    #[inline]
    pub fn phys_write8(&mut self, addr: u32, val: u8) {
        if let Some(i) = self.find(addr) {
            let region = &mut self.regions[i];
            let off = addr - region.start;
            region.memory.write8(off, val);
        }
    }

    pub fn phys_read16(&self, addr: u32) -> u16 {
        (self.phys_read8(addr) as u16) | ((self.phys_read8(addr + 1) as u16) << 8)
    }

    pub fn phys_read32(&self, addr: u32) -> u32 {
        (self.phys_read16(addr) as u32) | ((self.phys_read16(addr + 2) as u32) << 16)
    }

    pub fn phys_write16(&mut self, addr: u32, val: u16) {
        self.phys_write8(addr, val as u8);
        self.phys_write8(addr + 1, (val >> 8) as u8);
    }

    pub fn phys_write32(&mut self, addr: u32, val: u32) {
        self.phys_write16(addr, val as u16);
        self.phys_write16(addr + 2, (val >> 16) as u16);
    }

    /// Block read used by string instructions (`movs`, `lods`, …).
    pub fn phys_read_block(&mut self, addr: u32, out: &mut [u8]) {
        for (i, b) in out.iter_mut().enumerate() {
            *b = self.phys_read8(addr.wrapping_add(i as u32));
        }
    }

    /// Block write used by string instructions (`movs`, `stos`, …).
    pub fn phys_write_block(&mut self, addr: u32, data: &[u8]) {
        for (i, b) in data.iter().enumerate() {
            self.phys_write8(addr.wrapping_add(i as u32), *b);
        }
    }

    // ---- port I/O ----

    #[inline]
    pub fn io_read(&mut self, port: u16, size: u8) -> u32 {
        match &mut self.device {
            Some(d) => d.io_read(port, size),
            None => 0xFF,
        }
    }

    #[inline]
    pub fn io_write(&mut self, port: u16, size: u8, data: u32) {
        if let Some(d) = &mut self.device {
            d.io_write(port, size, data);
        }
    }

    /// Ask the machine's PIC which vector is in service (INTA cycle).
    #[inline]
    pub fn ack_irq(&mut self) -> Option<u8> {
        match &mut self.device {
            Some(d) => d.ack_irq(),
            None => None,
        }
    }

    /// Walk the machine's timers.
    #[inline]
    pub fn tick_device(&mut self, cycles: u64) {
        if let Some(d) = &mut self.device {
            d.tick(cycles);
        }
    }
}

impl std::fmt::Debug for Mem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mem")
            .field(
                "regions",
                &self
                    .regions
                    .iter()
                    .map(|r| (r.start, r.end))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}
