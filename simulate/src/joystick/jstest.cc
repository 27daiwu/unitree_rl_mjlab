#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>

#include "joystick.h"
#include "joystick_mapping.h"

namespace
{
using Profile = joystick_mapping::CustomSwitchPro;

const char *keyName(unsigned int code)
{
  switch (code)
  {
    case Profile::a_key: return "A";
    case Profile::b_key: return "B";
    case Profile::x_key: return "X";
    case Profile::y_key: return "Y";
    case Profile::l1_key: return "L1";
    case Profile::r1_key: return "R1";
    case Profile::l2_key: return "L2";
    case Profile::r2_key: return "R2";
    case Profile::select_key: return "SELECT";
    case Profile::start_key: return "START";
    default: return "unmapped";
  }
}

const char *axisName(unsigned int code)
{
  switch (code)
  {
    case Profile::lx_abs: return "LX";
    case Profile::ly_abs: return "LY";
    case Profile::rx_abs: return "RX";
    case Profile::ry_abs: return "RY";
    case Profile::dpad_x_abs: return "DPAD_X";
    case Profile::dpad_y_abs: return "DPAD_Y";
    default: return "unmapped";
  }
}
} // namespace

int main(int argc, char **argv)
{
  const std::string device = argc > 1
      ? argv[1]
      : "/dev/input/by-id/usb-057e_THUNDEROBOT_G30-event-joystick";
  EvdevJoystick joystick(device);
  if (!joystick.isFound())
  {
    std::cerr << "Failed to open " << device << '\n';
    return 1;
  }

  std::cout << "device=" << device << " profile=" << Profile::name
            << " backend=linux_evdev dpad_backend=ABS_HAT0X/ABS_HAT0Y\n";
  std::cout << std::fixed << std::setprecision(4)
            << "initial axes: LX=" << joystick.normalizedAxis(Profile::lx_abs)
            << " LY=" << joystick.normalizedAxis(Profile::ly_abs)
            << " RX=" << joystick.normalizedAxis(Profile::rx_abs)
            << " RY=" << joystick.normalizedAxis(Profile::ry_abs)
            << " DPAD_X=" << joystick.axis_[Profile::dpad_x_abs]
            << " DPAD_Y=" << joystick.axis_[Profile::dpad_y_abs] << '\n';
  std::cout << "Move every stick and D-pad direction, then press and release every button.\n";

  while (true)
  {
    input_event event{};
    if (!joystick.sample(&event))
    {
      usleep(1000);
      continue;
    }
    joystick.applyEvent(event);

    if (event.type == EV_ABS)
    {
      std::cout << "raw EV_ABS code=" << event.code
                << " value=" << event.value
                << " logical=" << axisName(event.code);
      if (event.code == Profile::lx_abs || event.code == Profile::ly_abs ||
          event.code == Profile::rx_abs || event.code == Profile::ry_abs)
        std::cout << " normalized=" << joystick.normalizedAxis(event.code);
      std::cout << '\n';
    }
    else if (event.type == EV_KEY)
    {
      std::cout << "raw EV_KEY code=" << event.code
                << " state=" << (event.value ? "pressed" : "released")
                << " value=" << event.value
                << " logical=" << keyName(event.code) << '\n';
    }
  }
}
