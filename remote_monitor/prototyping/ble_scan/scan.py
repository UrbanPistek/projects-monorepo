import asyncio
import sys
import time
import struct
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

TARGET_NAME = "pico-w-ub"

# Format string breakdown:
#   '<'  = little-endian (change to '>' for big-endian)
#   'I'  = unsigned 32-bit int  (4 bytes) → wake_count
#   'd'  = 64-bit double float  (8 bytes) → avg_flow_rate_litres_per_min
#   'd'  = 64-bit double float  (8 bytes) → total_volume_litres

FORMAT = '>Bff'  # Total: 20 bytes

def unpack_payload(payload: bytes) -> dict:
    assert len(payload) == struct.calcsize(FORMAT), \
        f"Expected {struct.calcsize(FORMAT)} bytes, got {len(payload)}"

    wake_count, avg_flow_rate, total_volume = struct.unpack(FORMAT, payload)

    return {
        "wake_count":                  wake_count,
        "avg_flow_rate_litres_per_min": avg_flow_rate,
        "total_volume_litres":          total_volume,
    }

def on_detection(device: BLEDevice, adv: AdvertisementData):
    if device.name != TARGET_NAME:
        return

    print(f"\n{'='*50}")
    print(f"time: [{time.time()}]")
    print(f"Device:  {device.name}")
    print(f"Address: {device.address}")
    print(f"RSSI:    {adv.rssi} dBm")

    if adv.local_name:
        print(f"Local Name: {adv.local_name}")

    if adv.service_uuids:
        print(f"Service UUIDs:")
        for uuid in adv.service_uuids:
            print(f"  {uuid}")

    if adv.service_data:
        print(f"Service Data:")
        for uuid, data in adv.service_data.items():
            print(f"  {uuid}: {data.hex()} ({list(data)})")

    if adv.manufacturer_data:
        print(f"Manufacturer Data:")
        for company_id, data in adv.manufacturer_data.items():
            print(f"  Company ID: 0x{company_id:04X} ({company_id})")
            print(f"  Raw bytes:  {data.hex()}")
            
            # Decode bytes
            decode_struct = unpack_payload(data)
            wake_count = decode_struct["wake_count"]
            avg_flow_rate_litres_per_min = decode_struct["avg_flow_rate_litres_per_min"]
            total_volume_litres = decode_struct["total_volume_litres"]
            
            # Print results:
            print(f"  Wake Count: {wake_count}")
            print(f"  Average Flow: {avg_flow_rate_litres_per_min} L/min")
            print(f"  Total Volumne: {total_volume_litres} L")

    if adv.tx_power:
        print(f"TX Power: {adv.tx_power} dBm")

    print(f"{'='*50}")


async def main():
    print(f"Scanning for '{TARGET_NAME}'... (Ctrl+C to stop)\n")

    async with BleakScanner(detection_callback=on_detection):
        try:
            await asyncio.sleep(float("inf"))
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScan stopped.")
        sys.exit(0)