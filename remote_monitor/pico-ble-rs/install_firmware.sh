#!/bin/bash

mkdir cyw43-firmware
BASE="https://raw.githubusercontent.com/embassy-rs/embassy/main/cyw43-firmware"
curl -L "$BASE/43439A0.bin"      -o cyw43-firmware/43439A0.bin
curl -L "$BASE/43439A0_clm.bin"  -o cyw43-firmware/43439A0_clm.bin
curl -L "$BASE/43439A0_btfw.bin" -o cyw43-firmware/43439A0_btfw.bin
