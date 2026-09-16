#pragma once

#include <iostream>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>
#include "joystick/joystick.h"
#include "joystick/joystick_mapping.h"
#include <memory>


class XBoxJoystick : public unitree::common::UnitreeJoystick
{
public:
    XBoxJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[6]);
        start(js_->button_[7]);
        LB(js_->button_[4]);
        RB(js_->button_[5]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[2]);
        Y(js_->button_[3]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[2] > 0);
        RT(js_->axis_[5] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[3]) / max_value_);
        ry(-double(js_->axis_[4]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


class SwitchJoystick : public unitree::common::UnitreeJoystick
{
public:
    SwitchJoystick(std::string device, int bits = 15)
	: unitree::common::UnitreeJoystick()
	{
		js_ = std::make_unique<Joystick>(device);
		if(!js_->isFound()) {
			std::cout << "Error: Joystick open failed." << std::endl;
			exit(1);
		}
        max_value_ = 1 << (bits - 1);
	}

    void update() override
    {
        js_->getState();
        back(js_->button_[10]);
        start(js_->button_[11]);
        LB(js_->button_[6]);
        RB(js_->button_[7]);
        A(js_->button_[0]);
        B(js_->button_[1]); 
        X(js_->button_[3]);
        Y(js_->button_[4]);
        up(js_->axis_[7] < 0);
        down(js_->axis_[7] > 0);
        left(js_->axis_[6] < 0);
        right(js_->axis_[6] > 0);
        LT(js_->axis_[5] > 0);
        RT(js_->axis_[4] > 0);
        lx(double(js_->axis_[0]) / max_value_);
        ly(-double(js_->axis_[1]) / max_value_);
        rx(double(js_->axis_[2]) / max_value_);
        ry(-double(js_->axis_[3]) / max_value_);
    }
private:
	std::unique_ptr<Joystick> js_;
	int max_value_;
};


class CustomSwitchProJoystick : public unitree::common::UnitreeJoystick
{
public:
    CustomSwitchProJoystick(std::string device, int bits = 15)
    : unitree::common::UnitreeJoystick()
    {
        (void)bits;
        js_ = std::make_unique<EvdevJoystick>(device);
        if(!js_->isFound()) {
            std::cout << "Error: Joystick open failed." << std::endl;
            exit(1);
        }
    }

    void update() override
    {
        using Profile = joystick_mapping::CustomSwitchPro;
        js_->getState();
        back(js_->key_[Profile::select_key]);
        start(js_->key_[Profile::start_key]);
        LB(js_->key_[Profile::l1_key]);
        RB(js_->key_[Profile::r1_key]);
        LT(js_->key_[Profile::l2_key]);
        RT(js_->key_[Profile::r2_key]);
        A(js_->key_[Profile::a_key]);
        B(js_->key_[Profile::b_key]);
        X(js_->key_[Profile::x_key]);
        Y(js_->key_[Profile::y_key]);
        up(js_->axis_[Profile::dpad_y_abs] < 0);
        right(js_->axis_[Profile::dpad_x_abs] > 0);
        down(js_->axis_[Profile::dpad_y_abs] > 0);
        left(js_->axis_[Profile::dpad_x_abs] < 0);
        lx(js_->normalizedAxis(Profile::lx_abs));
        ly(-js_->normalizedAxis(Profile::ly_abs));
        rx(js_->normalizedAxis(Profile::rx_abs));
        ry(-js_->normalizedAxis(Profile::ry_abs));
    }

private:
    std::unique_ptr<EvdevJoystick> js_;
};
