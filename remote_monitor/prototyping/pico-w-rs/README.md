# Sources

1. [rp-hal-boards](https://github.com/rp-rs/rp-hal-boards/tree/main/boards/rp-pico)
2. [pico pinout](https://picow.pinout.xyz/)
3. [pico programming guide](https://raspberrytips.com/getting-started-with-raspberry-pi-pico/)
4. [rp2040 rust template](https://github.com/rp-rs/rp2040-project-template)
5. [rp serial demo](https://pico.implrust.com/usb-serial/code-with-rp-hal.html)

# targets

## bilnky

Run blinky with a LED connected to gpio.

```
cargo run --bin blinky
```

## serial-echo

Open a serial connection and be able to check communication with the device.

```
cargo run --bin serial-echo
```

`sudo apt install tio`
`tio /dev/ttyACM0`
`ls /dev | grep ttyACM`
