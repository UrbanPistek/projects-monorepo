# Sources

1. [rp-hal-boards](https://github.com/rp-rs/rp-hal-boards/tree/main/boards/rp-pico)
2. [pico pinout](https://picow.pinout.xyz/)
3. [pico programming guide](https://raspberrytips.com/getting-started-with-raspberry-pi-pico/)
4. [rp2040 rust template](https://github.com/rp-rs/rp2040-project-template)
5. [rp serial demo](https://pico.implrust.com/usb-serial/code-with-rp-hal.html)
6. [Embassy-rs](https://github.com/embassy-rs/embassy/tree/main/examples/rp)
7. [hall-effect flow meter sensors](https://www.ti.com/content/dam/videos/external-videos/en-us/8/3816841626001/6299510807001.mp4/subassets/flow_meter_design_using_hall-effect_sensors.pdf)
8. [hall effect sensors](https://blog.productsforautomation.com/hall-effect-sensors/)
9. [struct chars](https://docs.python.org/3/library/struct.html#format-characters)
10. [raspberry pico docs](https://pip-assets.raspberrypi.com/categories/610-raspberry-pi-pico/documents/RP-008307-DS-2-pico-datasheet.pdf)
11. [beacon ble example](https://github.com/embassy-rs/trouble/blob/main/examples/apps/src/ble_beacon.rs)
12. [embassy examples](https://github.com/embassy-rs/embassy/blob/main/examples/rp/src/bin/gpio_async.rs)

# flash

1. Hold BOOTSEL on pico board - while holding plug USB into computer
2. Wait until device shows on computer, BOOTSEL can be released, then run `cargo run --release`
3. Baord should now be flashed - will no longer show on the computer

Debug Build Size: 850.50 KB
Release Build Size: 835.50 KB

## flow meter

More details about the flow meter:
F=(5.5*Q)±2%, Q=L/Min, error: ±2%
Working range: 1-60L/min
Working voltage: DC 5-24 V
Water Pressure: ≤1.2Mpa
Liquid temperature: 0-100℃
Maximum current consumption: 15 mA(DC 5V)
Wire length: 15 cm
Size: 66mm x 38mm(L*W)
Refer to the wiring:
Red wire: VCC(+)
Black wire: GND(-)
Yellow wire: Signal output
