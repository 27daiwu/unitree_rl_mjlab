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

#include "joystick.h"

#include <algorithm>
#include <cmath>
#include <sys/ioctl.h>

Joystick::Joystick()
{
  openPath("/dev/input/js0");
}

Joystick::Joystick(int joystickNumber)
{
  std::stringstream sstm;
  sstm << "/dev/input/js" << joystickNumber;
  openPath(sstm.str());
}

Joystick::Joystick(std::string devicePath)
{
  openPath(devicePath);
}

Joystick::Joystick(std::string devicePath, bool blocking)
{
  openPath(devicePath, blocking);
}

void Joystick::openPath(std::string devicePath, bool blocking)
{
  // Open the device using either blocking or non-blocking
  _fd = open(devicePath.c_str(), blocking ? O_RDONLY : O_RDONLY | O_NONBLOCK);
  if (_fd >= 0)
  {
    ioctl(_fd, JSIOCGAXES, &axisCount_);
    ioctl(_fd, JSIOCGAXMAP, axisMap_.data());
  }
}

bool Joystick::sample(JoystickEvent *event)
{
  int bytes = read(_fd, event, sizeof(*event));

  if (bytes == -1)
    return false;

  // NOTE if this condition is not met, we're probably out of sync and this
  // Joystick instance is likely unusable
  return bytes == sizeof(*event);
}

bool Joystick::isFound()
{
  return _fd >= 0;
}

int Joystick::axisNumberForAbsCode(unsigned char absCode) const
{
  const auto end = axisMap_.begin() + std::min<std::size_t>(axisCount_, axisMap_.size());
  const auto it = std::find(axisMap_.begin(), end, absCode);
  return it == end ? -1 : static_cast<int>(std::distance(axisMap_.begin(), it));
}

void Joystick::applyEvent(const JoystickEvent &event)
{
  if (event.isButton() && event.number < button_.size())
  {
    button_[event.number] = event.value;
  }
  else if (event.isAxis() && event.number < axis_.size())
  {
    axis_[event.number] = event.value;
  }
}

void Joystick::getState()
{
  if (sample(&event_))
  {
    applyEvent(event_);
  }
}

Joystick::~Joystick()
{
  if (_fd >= 0)
    close(_fd);
}

std::ostream &operator<<(std::ostream &os, const JoystickEvent &e)
{
  os << "type=" << static_cast<int>(e.type)
     << " number=" << static_cast<int>(e.number)
     << " value=" << static_cast<int>(e.value);
  return os;
}

EvdevJoystick::EvdevJoystick(const std::string &devicePath)
{
  fd_ = open(devicePath.c_str(), O_RDONLY | O_NONBLOCK);
  if (fd_ < 0)
    return;

  for (unsigned int code = 0; code <= ABS_MAX; ++code)
  {
    input_absinfo info{};
    if (ioctl(fd_, EVIOCGABS(code), &info) == 0)
    {
      absInfo_[code] = info;
      axis_[code] = info.value;
      hasAbs_[code] = true;
    }
  }
}

EvdevJoystick::~EvdevJoystick()
{
  if (fd_ >= 0)
    close(fd_);
}

bool EvdevJoystick::isFound() const
{
  return fd_ >= 0;
}

bool EvdevJoystick::sample(input_event *event)
{
  return read(fd_, event, sizeof(*event)) == sizeof(*event);
}

void EvdevJoystick::applyEvent(const input_event &event)
{
  if (event.type == EV_KEY && event.code <= KEY_MAX)
    key_[event.code] = event.value;
  else if (event.type == EV_ABS && event.code <= ABS_MAX)
    axis_[event.code] = event.value;
}

void EvdevJoystick::getState()
{
  input_event event{};
  while (sample(&event))
    applyEvent(event);
}

double EvdevJoystick::normalizedAxis(unsigned int code) const
{
  if (code > ABS_MAX || !hasAbs_[code])
    return 0.0;

  const auto &info = absInfo_[code];
  const double center = (static_cast<double>(info.minimum) + info.maximum) / 2.0;
  const double value = static_cast<double>(axis_[code]) - center;
  const double scale = value >= 0.0 ? info.maximum - center : center - info.minimum;
  if (scale <= 0.0)
    return 0.0;
  return std::clamp(value / scale, -1.0, 1.0);
}
