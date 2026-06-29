/*
 * RP2040 memory map — same for Pico, Pico W, and Pico WH.
 *
 * BOOT2  — 256-byte second-stage bootloader (configures QSPI flash, jumps to app)
 * FLASH  — remaining 2 MB of external QSPI flash
 *          Holds: .text (code), .rodata (const data), initial .data values,
 *          AND the three CYW43 firmware blobs baked in via include_bytes!
 * RAM    — 256 KB on-chip SRAM
 *          Holds: .data, .bss, stack, and embassy task arenas
 *
 * Note: the CYW43 firmware blobs (43439A0.bin ~220 KB, _clm.bin ~5 KB,
 *       _btfw.bin ~6 KB) are included with include_bytes! and live in FLASH.
 *       Make sure your total flash usage stays under ~1.75 MB.
 */
MEMORY {
    BOOT2 : ORIGIN = 0x10000000, LENGTH = 0x100
    FLASH : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    RAM   : ORIGIN = 0x20000000, LENGTH = 256K
}
