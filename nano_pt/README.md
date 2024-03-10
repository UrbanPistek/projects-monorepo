# Nano Power Station

Board: [KeeYees Development Board ESP-WROOM-32 Chip CP2102](https://www.amazon.ca/gp/product/B07QCP2451/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8)

USB Driver: [CP2101 Driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=downloads)

## Setup

1. Check USB drivers, if you are using Ubuntu the UART to USB driver for the CP2101 chip should be present, to check run:

```
ls -al /lib/modules/"$(uname -r)"/kernel/drivers/usb/serial/cp210x.ko
modinfo cp210x
```

Check other usb serial drivers:

```
ls -al /lib/modules/"$(uname -r)"/kernel/drivers/usb/serial/usbserial.kerkour
modinfo usbserial
```

If needed, manually load the drivers:

```
sudo modprobe usbserial
sudo modprobe cp210x
```

If the driver is not present: [Driver Download](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=downloads) & compile & load it yourself:

```
cd Linux-3-x-x-VCP-Driver-Source
make
cp cp210x.ko to /lib/modules/"$(uname -r)"/kernel/drivers/usb/serial/
sudo modprobe usbserial
sudo modprobe cp210x
```

2. Add the board to arduino using: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`

3. Run the following blinky program, use the board `DOIT ESP32 DEVKIT V1`, you may need to press the EN button on the ESP32. With this sketch you should see the LED turn on/off and the messages print to the serial monitor.

```
int ledPin = 2;

void setup() {

  Serial.begin(115200);
  pinMode(ledPin, OUTPUT);
  
  Serial.println("Setup done");
}

void loop() {
  
  digitalWrite(ledPin, HIGH);
  Serial.println("LED ON");
  delay(1000);
  
  digitalWrite(ledPin, LOW);
  Serial.println("LED OFF");
  delay(1000);
}
```

## MQTT

[Arduino MQTT](https://docs.arduino.cc/tutorials/uno-wifi-rev2/uno-wifi-r2-mqtt-device-to-device/)

Start the broker:

```
docker compose --file mqtt_broker.yml up -d
```

[MQTT CLI](https://www.hivemq.com/blog/mqtt-cli/)
[Hive MQTT CLI](https://github.com/hivemq/mqtt-cli)

Install:

```
wget https://github.com/hivemq/mqtt-cli/releases/download/v4.25.0/mqtt-cli-4.25.0.deb
sudo apt install ./mqtt-cli-4.25.0.deb
```

### Use MQTT CLI

Terminal 1:

```
mqtt sub -t test -h 0.0.0.0 -p 1883 -u <user> -pw <password>
```

Terminal 2:

```
mqtt pub -t test -m "hi" -h 0.0.0.0 -p 1883 -u <user> -pw <password>
```

### eclipse-mosquitto

[Documentation](https://mosquitto.org/documentation/)

Install: `sudo apt-get install mosquitto mosquitto-clients`

Create password for mqtt:

```
sudo mosquitto_passwd -c ./config/mosquitto.passwd <user_name>
```

Subscribe
```
mosquitto_sub -h 127.0.0.1 -p 1883 -t test -u <user> -P <password> 
```

Publish
```
mosquitto_pub -h 127.0.0.1 -p 1883 -t test -m 0xFF -u <user> -P <password> 
```

### Resources

[ESP32 Dev Board](https://www.adafruit.com/product/3269)
[Pico Board]()

Embedded Rust
- https://www.rust-lang.org/what/Embedded
- https://kerkour.com/rust-on-esp32

[Zephyr ESP32 Support](https://docs.zephyrproject.org/latest/boards/espressif/esp32_devkitc_wroom/doc/index.html)
[ESP32 Dev Board Arduino](https://randomnerdtutorials.com/getting-started-with-esp32/)

Give all permissions to a file / folder in linux:

```
chmod ugo+rwx <path>
```

Links
- https://askubuntu.com/questions/941594/installing-cp210x-driver
- https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers?tab=downloads
- https://lastminuteengineers.com/esp32-arduino-ide-tutorial/
- https://randomnerdtutorials.com/getting-started-with-esp32/
- https://hub.docker.com/_/eclipse-mosquitto
- 

