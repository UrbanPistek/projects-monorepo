# Sources

1. [rp-hal-boards](https://github.com/rp-rs/rp-hal-boards/tree/main/boards/rp-pico)
2. [pico pinout](https://picow.pinout.xyz/)
3. [pico programming guide](https://raspberrytips.com/getting-started-with-raspberry-pi-pico/)
4. [rp2040 rust template](https://github.com/rp-rs/rp2040-project-template)
5. [rp serial demo](https://pico.implrust.com/usb-serial/code-with-rp-hal.html)
6. [Embassy-rs](https://github.com/embassy-rs/embassy/tree/main/examples/rp)
7. [hall-effect flow meter sensors](https://www.ti.com/content/dam/videos/external-videos/en-us/8/3816841626001/6299510807001.mp4/subassets/flow_meter_design_using_hall-effect_sensors.pdf)
8. [hall effect sensors](https://blog.productsforautomation.com/hall-effect-sensors/)

# targets

## bilnky

Run blinky with a LED connected to gpio.

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
