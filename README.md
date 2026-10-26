# CROW: Community Recording Online Workflow

The CROW app is used for configuring AudioMoth devices and analyzing generated results from AudioMoths. See [Greater Seattle Soundsacpe Monitoring Project](https://seattle-soundscape.uw.edu) for more information about the project.

## Flashing AudioMoths

After logging in, you can flash any connected AudioMoth to the preset setting with the latest firmware:
Sample rate: 192 kHz
Duty cycle: 10 min / 50 min
Schedule: 00:00-24:00

## Analyzing Data

The app can analyze data from an AudioMoth SD card structure ready for upload to the webportal. You will be prompted to select a location and inventory item before, logging in and selecting the correct settings is essential to populate the META.TXT file properly. BirdNET 2.4 and BattyBirdNET 1.0 USA-WEST are used.

## Signing in

The app currently requires signing in to configure AudioMoths and analyze data using the models. Use the same credentials as you would on the webportal.

## OS Specifics

- **macOS:** The app may be blocked on open, go to **System Settings -> Privacy & Security** and click **Open Anyway**.
- **Linux:** To connect AudioMoths, create `/etc/udev/rules.d/99-audiomoth.rules` containing:
  ```
  SUBSYSTEM=="usb", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="0002", MODE="0666"
  SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="0003", MODE="0666"
  ```
  Then run `sudo udevadm control --reload-rules` and reconnect the AudioMoth.

## Licenses

`audiomoth-hid` and the bundled `flash` binaries are MIT-licensed by Open Acoustic Devices. The EFM32-Flash license is in `bin/`.

The BirdNET and BattyBirdNET models are licensed [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only, with attribution (see `models/README.md`).