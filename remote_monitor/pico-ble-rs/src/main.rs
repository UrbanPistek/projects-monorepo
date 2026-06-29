/*! pico-wh-ble — BLE "Hello, World!" GATT peripheral on the Raspberry Pi Pico WH
 *
 * Overview
 * ────────
 * This firmware turns the Pico WH into a BLE GATT Peripheral.  It:
 *   1. Boots the CYW43439 wireless chip over PIO SPI.
 *   2. Initialises the TrouBLE BLE host stack using cyw43 as the HCI controller.
 *   3. Advertises as "PicoHello".
 *   4. Exposes a custom GATT service with one read-only characteristic that
 *      always returns the UTF-8 bytes of "Hello, World!".
 *   5. Accepts connections and serves reads indefinitely.
 *
 * BLE primer (why this is more complex than a UART bridge)
 * ────────────────────────────────────────────────────────
 * BLE has no concept of a byte stream.  Instead, data lives in a GATT
 * (Generic Attribute Profile) hierarchy:
 *
 *   Device
 *   └─ Service  (identified by a 128-bit UUID)
 *      └─ Characteristic  (also a UUID; has a value + properties like READ/NOTIFY)
 *
 * A BLE Central (your phone) scans for advertising packets, connects, discovers
 * services/characteristics, and then reads or subscribes to them.
 *
 * UUIDs used (randomly generated — safe to change)
 * ────────────────────────────────────────────────
 *   Service UUID        : 12345678-1234-5678-1234-56789abcdef0
 *   Characteristic UUID : 12345678-1234-5678-1234-56789abcdef1
 *
 * Firmware blobs
 * ──────────────
 * The CYW43439 requires three firmware files baked into flash:
 *   43439A0.bin       — Wi-Fi + BT firmware (~220 KB)
 *   43439A0_clm.bin   — CLM (regulatory) blob (~5 KB)
 *   43439A0_btfw.bin  — Bluetooth firmware patch (~6 KB)
 *
 * Download them from:
 *   https://github.com/embassy-rs/embassy/tree/main/cyw43-firmware
 * Place them in a `cyw43-firmware/` directory next to `src/`.
 *
 * Hardware (Pico WH — all wireless pins are fixed, not user-selectable)
 * ──────────────────────────────────────────────────────────────────────
 *   PIN_23  WL_ON / CYW43 power enable
 *   PIN_24  CYW43 SPI data / IRQ line (PIO SPI MISO / IRQ)
 *   PIN_25  CYW43 SPI chip-select
 *   PIN_29  CYW43 SPI clock
 *   DMA_CH0 DMA channel used by PIO SPI transfer
 *   PIO0    PIO block used by cyw43-pio
 */

#![no_std]
#![no_main]

// ── Imports ───────────────────────────────────────────────────────────────────

use bt_hci::controller::ExternalController;
use cyw43_pio::PioSpi;
use defmt::{info, unwrap};
use embassy_executor::Spawner;
use embassy_rp::{
    bind_interrupts,
    gpio::{Level, Output},
    peripherals::{DMA_CH0, PIO0, USB},
    pio::{InterruptHandler as PioInterruptHandler, Pio},
    usb::{Driver as UsbDriver, InterruptHandler as UsbInterruptHandler},
};
use embassy_time::Timer;
use static_cell::StaticCell;
use trouble_host::{
    advertise::{AdStructure, Advertisement, BR_EDR_NOT_SUPPORTED, LE_GENERAL_DISCOVERABLE},
    attribute::{AttributeTable, CharacteristicProp, Service, Uuid},
    gatt::GattEvent,
    BleHostResources, Controller, PacketQos, Stack,
};
use {defmt_rtt as _, panic_probe as _};

// ── Interrupt bindings ────────────────────────────────────────────────────────
//
// `bind_interrupts!` wires RP2040 hardware interrupt vectors to the Embassy
// interrupt handlers for PIO0 (used by cyw43-pio) and USB (used for logging).

bind_interrupts!(struct Irqs {
    PIO0_IRQ_0    => PioInterruptHandler<PIO0>;
    USBCTRL_IRQ   => UsbInterruptHandler<USB>;
});

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum BLE payload size (ATT MTU minus 3 bytes overhead)
const L2CAP_MTU: usize = 251;

/// "Hello, World!" encoded as a static byte slice — this is the GATT value.
const HELLO: &[u8] = b"Hello, World!";

/// 128-bit UUID for our custom service.
/// Format: 12345678-1234-5678-1234-56789abcdef0
const SERVICE_UUID: Uuid = Uuid::new_long([
    0x12, 0x34, 0x56, 0x78,
    0x12, 0x34,
    0x56, 0x78,
    0x12, 0x34,
    0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0,
]);

/// 128-bit UUID for the "Hello" read characteristic.
/// Format: 12345678-1234-5678-1234-56789abcdef1
const HELLO_CHAR_UUID: Uuid = Uuid::new_long([
    0x12, 0x34, 0x56, 0x78,
    0x12, 0x34,
    0x56, 0x78,
    0x12, 0x34,
    0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf1,
]);

// ── Static allocations ────────────────────────────────────────────────────────
//
// Embassy requires certain long-lived objects to live in `'static` memory.
// `StaticCell<T>` is a safe wrapper that initialises the cell exactly once.

static CYW43_STATE: StaticCell<cyw43::State> = StaticCell::new();

// BleHostResources<CONNECTIONS, L2CAP_CHANNELS, L2CAP_MTU>
// - CONNECTIONS=1    : we only need to handle one central at a time
// - L2CAP_CHANNELS=3 : enough for ATT + SM + one spare
type MyBleResources = BleHostResources<ExternalController<cyw43::BtDriver<'static>, 10>, 1, 3, L2CAP_MTU>;
static BLE_RESOURCES: StaticCell<MyBleResources> = StaticCell::new();

// ── Tasks ─────────────────────────────────────────────────────────────────────

/// Drives the CYW43 chip's internal event loop.  Must run concurrently with
/// everything else — if this task is not polled the BLE/Wi-Fi stack stalls.
#[embassy_executor::task]
async fn cyw43_task(runner: cyw43::Runner<'static, Output<'static>>) -> ! {
    runner.run().await
}

/// Runs the USB serial logger so defmt log output appears on your PC.
#[embassy_executor::task]
async fn logger_task(driver: UsbDriver<'static, USB>) {
    embassy_usb_logger::run!(1024, log::LevelFilter::Info, driver);
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // ── 1. Initialise Embassy-RP HAL ─────────────────────────────────────────
    //
    // `embassy_rp::init` configures the system clock, resets peripherals, and
    // returns a `Peripherals` struct that grants ownership of every RP2040
    // peripheral exactly once (Rust's ownership model enforces this).
    let p = embassy_rp::init(Default::default());

    // ── 2. Start USB logger ───────────────────────────────────────────────────
    //
    // This lets you see `info!(...)` messages over the USB serial port without
    // needing a debug probe.  Connect via `minicom`, `screen`, or any serial
    // terminal at any baud rate (USB CDC is baud-rate agnostic).
    let usb_driver = UsbDriver::new(p.USB, Irqs);
    spawner.spawn(logger_task(usb_driver)).unwrap();

    // ── 3. Load CYW43 firmware blobs from flash ───────────────────────────────
    //
    // `include_bytes!` embeds the files at compile time.  The paths are relative
    // to the *Cargo.toml*, so place the `cyw43-firmware/` directory there.
    //
    // Download from: https://github.com/embassy-rs/embassy/tree/main/cyw43-firmware
    let fw   = include_bytes!("../../cyw43-firmware/43439A0.bin");
    let clm  = include_bytes!("../../cyw43-firmware/43439A0_clm.bin");
    let btfw = include_bytes!("../../cyw43-firmware/43439A0_btfw.bin");

    // ── 4. Set up PIO SPI for the CYW43439 ───────────────────────────────────
    //
    // The CYW43439 uses a non-standard half-duplex SPI protocol.  cyw43-pio
    // implements this using the RP2040's PIO (Programmable I/O) block, which
    // can run custom state-machine programs for arbitrary bit-level protocols.
    //
    // All four SPI pins (PWR, CS, CLK, DATA/IRQ) are fixed on the Pico W/WH
    // PCB — they cannot be remapped.
    let pwr = Output::new(p.PIN_23, Level::Low);   // CYW43 power enable (active-high)
    let cs  = Output::new(p.PIN_25, Level::High);  // SPI chip-select (active-low)

    let mut pio = Pio::new(p.PIO0, Irqs);

    // PioSpi::new programs PIO0's state machine 0 with cyw43-pio's SPI program.
    // PIN_24 is the combined MISO/IRQ line; PIN_29 is CLK.
    let spi = PioSpi::new(
        &mut pio.common,
        pio.sm0,
        pio.irq0,
        cs,
        p.PIN_24,  // DATA / IRQ
        p.PIN_29,  // CLK
        p.DMA_CH0, // DMA channel for bulk transfers
    );

    // ── 5. Initialise the CYW43 driver with Bluetooth enabled ────────────────
    //
    // `new_with_bluetooth` returns four items:
    //   _net_device — the Wi-Fi network device (unused here)
    //   bt_device   — the BLE HCI controller (passed to trouble-host)
    //   control     — a handle to control the chip (LED, CLM upload, etc.)
    //   runner      — the event-loop task (must be spawned)
    let state = CYW43_STATE.init(cyw43::State::new());
    let (_net_device, bt_device, mut control, runner) =
        cyw43::new_with_bluetooth(state, pwr, spi, fw, btfw).await;

    // Spawn the CYW43 runner — this MUST be running at all times.
    unwrap!(spawner.spawn(cyw43_task(runner)));

    // Upload the CLM (Country/Regulatory) blob.  Required before the chip will
    // transmit; calling this with the wrong blob causes silent RF failures.
    control.init(clm).await;

    info!("CYW43 initialised, BLE controller ready");

    // ── 6. Build the TrouBLE BLE host stack ───────────────────────────────────
    //
    // `ExternalController` wraps the cyw43 BT device so it implements the
    // bt-hci `Controller` trait that trouble-host consumes.
    let controller: ExternalController<_, 10> = ExternalController::new(bt_device);

    // `BleHostResources` provides all the static buffers the host stack needs
    // (packet queues, connection tables, L2CAP channel buffers).
    let resources = BLE_RESOURCES.init(MyBleResources::new(PacketQos::None));

    // `Stack::new` creates the BLE host.  The `address` argument is `None` here,
    // which makes the stack use a random static address generated from the RP2040
    // unique ID — safe for development use.
    let (stack, mut peripheral, _, runner) =
        trouble_host::new(controller, resources)
            .set_random_address(trouble_host::Address::random([0x42, 0x00, 0x01, 0x02, 0x03, 0x04]))
            .build();

    info!("BLE host stack built");

    // ── 7. Build the GATT attribute table ────────────────────────────────────
    //
    // `AttributeTable` is a fixed-size compile-time structure that describes
    // all the GATT services and characteristics this device exposes.
    // The generic parameter is the total number of attribute handles.
    let mut table: AttributeTable<'_, _, 10> = AttributeTable::new();

    // ── 7a. Generic Access service (mandatory per BLE spec) ───────────────────
    //
    // Every BLE peripheral must expose a Generic Access service (UUID 0x1800)
    // with a Device Name characteristic (UUID 0x2A00).  The device name is what
    // appears in your phone's Bluetooth scanner.
    let id = b"PicoHello";
    let appearance = [0x00u8, 0x00]; // Generic Unknown appearance category
    let mut svc = table.add_service(Service::new(0x1800_u16));  // Generic Access
    let _ = svc.add_characteristic_ro(0x2A00_u16, id);          // Device Name
    let _ = svc.add_characteristic_ro(0x2A01_u16, &appearance); // Appearance
    svc.build();

    // ── 7b. Generic Attribute service (mandatory) ─────────────────────────────
    table.add_service(Service::new(0x1801_u16)).build(); // Generic Attribute

    // ── 7c. Our custom "Hello" service ───────────────────────────────────────
    let mut hello_svc = table.add_service(Service::new(SERVICE_UUID));

    // Add a single READ-only characteristic holding "Hello, World!".
    // `add_characteristic_ro` takes the UUID and a `&[u8]` value reference.
    let hello_handle = hello_svc
        .add_characteristic_ro(HELLO_CHAR_UUID, HELLO)
        .build();

    hello_svc.build();

    info!("GATT table built — service UUID ends in f0, char UUID ends in f1");

    // ── 8. Compose the advertising payload ───────────────────────────────────
    //
    // BLE advertising packets are a list of AD (Advertising Data) structures.
    // We include:
    //   - Flags: LE General Discoverable, BR/EDR not supported (BLE-only device)
    //   - Complete Local Name: "PicoHello"
    //
    // The total must fit in 31 bytes (BLE advertising PDU limit).
    let mut adv_data = [0u8; 31];
    let adv_len = AdStructure::encode_slice(
        &[
            AdStructure::Flags(LE_GENERAL_DISCOVERABLE | BR_EDR_NOT_SUPPORTED),
            AdStructure::CompleteLocalName(b"PicoHello"),
        ],
        &mut adv_data,
    )
    .unwrap();

    info!("Starting BLE advertising as 'PicoHello'");

    // ── 9. Main BLE loop ──────────────────────────────────────────────────────
    //
    // We run the BLE host runner and the application logic concurrently using
    // `embassy_futures::join::join`.  The host runner processes HCI events from
    // the CYW43; the app_task handles advertising and GATT requests.

    let host_fut = runner.run();

    let app_fut = async {
        loop {
            // Start advertising.  This call returns an `Advertiser` future that
            // resolves when a Central connects to us.
            let advertiser = peripheral
                .advertise(
                    // Advertising interval: 100 ms (160 × 0.625 ms)
                    &trouble_host::advertise::AdvertisementParameters {
                        interval_min: trouble_host::advertise::Duration::from_millis(100),
                        interval_max: trouble_host::advertise::Duration::from_millis(100),
                        ..Default::default()
                    },
                    Advertisement::ConnectableScannableUndirected {
                        adv_data: &adv_data[..adv_len],
                        scan_data: &[], // no scan-response data needed
                    },
                )
                .await
                .unwrap();

            info!("Advertising... waiting for a BLE Central to connect");

            // `advertiser.accept()` suspends until a connection arrives.
            let conn = advertiser.accept().await.unwrap();
            info!("Central connected!");

            // ── GATT event loop ────────────────────────────────────────────
            //
            // Once connected, we handle incoming GATT events.  For our
            // read-only characteristic, the only event we care about is
            // `GattEvent::Read` — the central asked to read a characteristic.
            //
            // The `stack.run_gatt_server(&conn, &table)` call drives the ATT
            // layer (handles MTU exchange, discovery, reads, writes) and yields
            // `GattEvent` items for events the application must handle.
            let server = stack.gatt_server::<_, _, L2CAP_MTU>(&conn, &table);

            loop {
                match server.next().await {
                    Ok(GattEvent::Read(event)) => {
                        // Check if the read is targeting our Hello characteristic.
                        if event.handle() == hello_handle {
                            // Reply with the "Hello, World!" bytes.
                            // `reply` sets the response value that will be sent
                            // back in the ATT Read Response PDU.
                            if let Err(e) = event.reply(Ok(HELLO)) {
                                defmt::warn!("GATT read reply error: {:?}", e);
                            } else {
                                info!("Replied to READ with: {:?}", HELLO);
                            }
                        }
                    }
                    Ok(GattEvent::Write(_event)) => {
                        // Our characteristic is read-only; writes will be
                        // rejected by the ATT layer before reaching here,
                        // but we handle the arm to be exhaustive.
                    }
                    Err(e) => {
                        // Connection dropped or BLE protocol error.
                        defmt::warn!("GATT connection ended: {:?}", e);
                        break; // go back to advertising
                    }
                }
            }

            info!("Central disconnected — resuming advertising");

            // Small delay before re-advertising to avoid thrashing
            Timer::after_millis(200).await;
        }
    };

    // Run both futures concurrently.  Neither returns, so this line never exits.
    embassy_futures::join::join(host_fut, app_fut).await;
}
