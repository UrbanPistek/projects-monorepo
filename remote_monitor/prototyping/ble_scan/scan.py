import asyncio
import sys
import time
from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

TARGET_NAME = "pico-w-iot-beacon"

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
            print(f"  As ints:    {list(data)}")
            if len(data) >= 4:
                import struct
                value = struct.unpack_from("<I", data)[0]
                print(f"  As u32 LE:  {value}")

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