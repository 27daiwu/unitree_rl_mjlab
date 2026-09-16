#pragma once

#include <linux/input-event-codes.h>

namespace joystick_mapping
{

struct CustomSwitchPro
{
    static constexpr const char *name = "custom_switchpro";

    // SDL/Pygame IDs from the validated Python implementation.
    static constexpr int pygame_lx_axis = 0;
    static constexpr int pygame_ly_axis = 1;
    static constexpr int pygame_rx_axis = 2;
    static constexpr int pygame_ry_axis = 3;

    static constexpr int pygame_a_button = 0;
    static constexpr int pygame_b_button = 1;
    static constexpr int pygame_y_button = 2;
    static constexpr int pygame_x_button = 3;
    static constexpr int pygame_l1_button = 5;
    static constexpr int pygame_r1_button = 6;
    static constexpr int pygame_l2_button = 7;
    static constexpr int pygame_r2_button = 8;
    static constexpr int pygame_select_button = 9;
    static constexpr int pygame_start_button = 10;

    // Linux evdev codes exposed by hid_nintendo for this controller.
    static constexpr int lx_abs = ABS_X;
    static constexpr int ly_abs = ABS_Y;
    static constexpr int rx_abs = ABS_RX;
    static constexpr int ry_abs = ABS_RY;
    static constexpr int dpad_x_abs = ABS_HAT0X;
    static constexpr int dpad_y_abs = ABS_HAT0Y;

    static constexpr int a_key = BTN_SOUTH;
    static constexpr int b_key = BTN_EAST;
    static constexpr int y_key = BTN_NORTH;
    static constexpr int x_key = BTN_WEST;
    static constexpr int l1_key = BTN_TL;
    static constexpr int r1_key = BTN_TR;
    static constexpr int l2_key = BTN_TL2;
    static constexpr int r2_key = BTN_TR2;
    static constexpr int select_key = BTN_SELECT;
    static constexpr int start_key = BTN_START;
};

} // namespace joystick_mapping
