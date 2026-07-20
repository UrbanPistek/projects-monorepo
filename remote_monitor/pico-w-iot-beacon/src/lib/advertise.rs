//! BLE beacon advertising via the TrouBLE host stack.
//! The CYW43439
//! is the *controller* and speaks HCI. `ExternalController` bridges the two.
//!
//! Using the **non-connectable** advertising: scanners receive our payload without
//! opening a connection, which suits a short broadcast window.

use ::cyw43::bluetooth::BtDriver;
use bt_hci::controller::ExternalController;
use embassy_time::{Duration, Timer};
use static_cell::StaticCell;
use trouble_host::advertise::{
    AdStructure, Advertisement, AdvertisementParameters, BR_EDR_NOT_SUPPORTED, LE_GENERAL_DISCOVERABLE,
};
use trouble_host::prelude::*;
use trouble_host::{Address, Controller, HostResources, PacketPool, Stack};

// Internal modules
use crate::pwm::FlowMeasurements;

/// Bluetooth "company identifier" for manufacturer-specific data.
const COMPANY_ID: u16 = 0x000D; // 13

/// Device name shown in BLE scanner apps.
const DEVICE_NAME: &[u8] = b"pico-w-ub";

// 0xC2 = 11xxxxxx — valid static random address
// The top two bits of byte 0 must be `11` for a static random address.
const DEVICE_ADDRESS: [u8; 6] = [0xC2, 0x00, 0x01, 0x02, 0x03, 0x04];

type BleController = ExternalController<BtDriver<'static>, 10>;
type BleResources = HostResources<BleController, DefaultPacketPool, 0, 0>;

static BLE_RESOURCES: StaticCell<BleResources> = StaticCell::new();

/// Builds the TrouBLE host stack and returns it.
///
/// The stack must stay alive for the entire firmware run. `HostResources` lives
/// in a `StaticCell` so the borrow can be `'static`.
pub fn setup(bt_device: BtDriver<'static>) -> Stack<'static, BleController, DefaultPacketPool> {
    let resources = BLE_RESOURCES.init(BleResources::new());
    let controller = ExternalController::new(bt_device);

    // Uses the static address defined in the DEVICE_ADDRESS constant.
    // n BLE, this is a static random address (fixed until reboot), not a public IEEE address. 
    // TrouBLE exposes it via Address::random(...), which marks the address type as random. 
    // During host init it sends the HCI LE Set Random Address command with your bytes
    trouble_host::new(controller, resources)
        .set_random_address(Address::random(DEVICE_ADDRESS))
        .build()
}

/// Encodes the 31-byte BLE advertising PDU payload.
fn encode_adv_data(wake_count: u8, buf: &mut [u8; 31], measurements: FlowMeasurements) -> usize {
    
    /*
    wake_count - 1 byte
    avg_flow_rate_litres_per_min - 4 bytes
    total_volumne_litres - 4 bytes
    total = 9 bytes
    payload = [wake_count (1), avg_flow_rate_litres_per_min (4), total_volumne_litres (4)]
     */
    let mut data: [u8; 9] = [0u8; 9];
    data[0..1].copy_from_slice(&wake_count.to_be_bytes());
    data[1..5].copy_from_slice(&measurements.avg_flow_rate_litres_per_min.to_be_bytes());
    data[5..9].copy_from_slice(&measurements.total_volumne_litres.to_be_bytes());

    /*
    total = 14 bytes, still have 17 bytes useable for future data
     */
    AdStructure::encode_slice(
        &[
            AdStructure::Flags(LE_GENERAL_DISCOVERABLE | BR_EDR_NOT_SUPPORTED), // 3 bytes
            AdStructure::ManufacturerSpecificData {
                company_identifier: COMPANY_ID, // 2 bytes
                payload: &data, // 9 bytes
            },
            AdStructure::ShortenedLocalName(DEVICE_NAME),
        ],
        buf,
    )
    .expect("advertising data fits in 31 bytes")
}

/// Advertises a beacon payload for `duration`, then stops.
///
/// Starting advertising returns an [`Advertiser`] guard. While it is alive the
/// controller transmits our PDU. Dropping it (or letting it go out of scope)
/// sends the HCI command to disable advertising — that is how we end each window cleanly.
pub async fn run_burst<C, P>(
    peripheral: &mut trouble_host::peripheral::Peripheral<'_, C, P>,
    wake_count: u8,
    measurements: FlowMeasurements, 
    duration: Duration,
) where
    C: Controller,
    P: PacketPool,
{
    // Legacy BLE only supports 31 bytes packet
    let mut adv_buf = [0u8; 31];
    let adv_len = encode_adv_data(wake_count, &mut adv_buf, measurements);

    // 100 ms interval is a reasonable trade-off between discoverability and power.
    let params = AdvertisementParameters {
        interval_min: Duration::from_millis(100),
        interval_max: Duration::from_millis(100),
        ..Default::default()
    };

    let advertiser = peripheral
        .advertise(
            &params,
            Advertisement::NonconnectableNonscannableUndirected {
                adv_data: &adv_buf[..adv_len],
            },
        )
        .await
        .expect("advertise");

    Timer::after(duration).await;

    // Drop stops advertising. No connection was accepted — beacon-only mode.
    drop(advertiser);
}

