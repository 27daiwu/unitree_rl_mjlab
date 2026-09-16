// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright Drew Noakes 2013-2016

#ifndef __JOYSTICK_H__
#define __JOYSTICK_H__

#include <iostream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sstream>
#include <map>
#include <array>
#include <linux/input.h>
#include <linux/joystick.h>
#include "unistd.h"

class JoystickEvent
{
public:
  /** Minimum value of axes range */
  static const short MIN_AXES_VALUE = -32768;

  /** Maximum value of axes range */
  static const short MAX_AXES_VALUE = 32767;

  /**
   * The timestamp of the event, in milliseconds.
   */
  unsigned int time;

  /**
   * The value associated with this joystick event.
   * For buttons this will be either 1 (down) or 0 (up).
   * For axes, this will range between MIN_AXES_VALUE and MAX_AXES_VALUE.
   */
  short value;

  /**
   * The event type.
   */
  unsigned char type;

  /**
   * The axis/button number.
   */
  unsigned char number;

  /**
   * Returns true if this event is the result of a button press.
   */
  bool isButton() const
  {
    return (type & JS_EVENT_BUTTON) != 0;
  }

  /**
   * Returns true if this event is the result of an axis movement.
   */
  bool isAxis() const
  {
    return (type & JS_EVENT_AXIS) != 0;
  }

  /**
   * Returns true if this event is part of the initial state obtained when
   * the joystick is first connected to.
   */
  bool isInitialState() const
  {
    return (type & JS_EVENT_INIT) != 0;
  }

  /**
   * The ostream inserter needs to be a friend so it can access the
   * internal data structures.
   */
  friend std::ostream &operator<<(std::ostream &os, const JoystickEvent &e);
};

/**
 * Stream insertion function so you can do this:
 *    cout << event << endl;
 */
std::ostream &operator<<(std::ostream &os, const JoystickEvent &e);

/**
 * Represents a joystick device. Allows data to be sampled from it.
 */
class Joystick
{
private:
  void openPath(std::string devicePath, bool blocking = false);
  int _fd;

public:
  ~Joystick();

  /**
   * Initialises an instance for the first joystick: /dev/input/js0
   */
  Joystick();

  /**
   * Initialises an instance for the joystick with the specified,
   * zero-indexed number.
   */
  Joystick(int joystickNumber);

  /**
   * Initialises an instance for the joystick device specified.
   */
  Joystick(std::string devicePath);

  /**
   * Joystick objects cannot be copied
   */
  Joystick(Joystick const &) = delete;

  /**
   * Joystick objects can be moved
   */
  Joystick(Joystick &&) = default;

  /**
   * Initialises an instance for the joystick device specified and provide
   * the option of blocking I/O.
   */
  Joystick(std::string devicePath, bool blocking);

  /**
   * Returns true if the joystick was found and may be used, otherwise false.
   */
  bool isFound();

  /** Returns the /dev/input/jsN axis number for a Linux ABS_* code. */
  int axisNumberForAbsCode(unsigned char absCode) const;

  /**
   * Attempts to populate the provided JoystickEvent instance with data
   * from the joystick. Returns true if data is available, otherwise false.
   */

  void getState();

  JoystickEvent event_;
  std::array<int, KEY_MAX - BTN_MISC + 1> button_{};
  std::array<int, ABS_MAX + 1> axis_{};

  bool sample(JoystickEvent *event);

  void applyEvent(const JoystickEvent &event);

private:
  std::array<unsigned char, ABS_MAX + 1> axisMap_{};
  unsigned char axisCount_ = 0;
};

/** Linux evdev backend used when the kernel does not create /dev/input/jsN. */
class EvdevJoystick
{
public:
  explicit EvdevJoystick(const std::string &devicePath);
  ~EvdevJoystick();

  EvdevJoystick(EvdevJoystick const &) = delete;
  EvdevJoystick(EvdevJoystick &&) = default;

  bool isFound() const;
  bool sample(input_event *event);
  void applyEvent(const input_event &event);
  void getState();
  double normalizedAxis(unsigned int code) const;

  std::array<int, KEY_MAX + 1> key_{};
  std::array<int, ABS_MAX + 1> axis_{};

private:
  int fd_ = -1;
  std::array<input_absinfo, ABS_MAX + 1> absInfo_{};
  std::array<bool, ABS_MAX + 1> hasAbs_{};
};

#endif
