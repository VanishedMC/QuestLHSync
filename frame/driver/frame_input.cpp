#include "frame_input.h"

#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

void Log(const char *fmt, ...) {
  char b[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  if (vr::VRDriverLog()) vr::VRDriverLog()->Log((std::string("questlhsync input: ") + b).c_str());
}

struct Quat {
  double w = 1, x = 0, y = 0, z = 0;
};

Quat Mul(const Quat &a, const Quat &b) {
  return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
          a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
          a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

void Rotate(const Quat &q, const double v[3], double out[3]) {
  const double ux = q.x, uy = q.y, uz = q.z;
  const double cx = uy * v[2] - uz * v[1];
  const double cy = uz * v[0] - ux * v[2];
  const double cz = ux * v[1] - uy * v[0];
  const double dx = uy * cz - uz * cy;
  const double dy = uz * cx - ux * cz;
  const double dz = ux * cy - uy * cx;
  out[0] = v[0] + 2 * (q.w * cx + dx);
  out[1] = v[1] + 2 * (q.w * cy + dy);
  out[2] = v[2] + 2 * (q.w * cz + dz);
}

Quat FromMatrix(const vr::HmdMatrix34_t &m) {
  Quat q;
  const double tr = m.m[0][0] + m.m[1][1] + m.m[2][2];
  if (tr > 0) {
    const double s = std::sqrt(tr + 1.0) * 2;
    q.w = 0.25 * s;
    q.x = (m.m[2][1] - m.m[1][2]) / s;
    q.y = (m.m[0][2] - m.m[2][0]) / s;
    q.z = (m.m[1][0] - m.m[0][1]) / s;
  } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
    const double s = std::sqrt(1.0 + m.m[0][0] - m.m[1][1] - m.m[2][2]) * 2;
    q.w = (m.m[2][1] - m.m[1][2]) / s;
    q.x = 0.25 * s;
    q.y = (m.m[0][1] + m.m[1][0]) / s;
    q.z = (m.m[0][2] + m.m[2][0]) / s;
  } else if (m.m[1][1] > m.m[2][2]) {
    const double s = std::sqrt(1.0 + m.m[1][1] - m.m[0][0] - m.m[2][2]) * 2;
    q.w = (m.m[0][2] - m.m[2][0]) / s;
    q.x = (m.m[0][1] + m.m[1][0]) / s;
    q.y = 0.25 * s;
    q.z = (m.m[1][2] + m.m[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + m.m[2][2] - m.m[0][0] - m.m[1][1]) * 2;
    q.w = (m.m[1][0] - m.m[0][1]) / s;
    q.x = (m.m[0][2] + m.m[2][0]) / s;
    q.y = (m.m[1][2] + m.m[2][1]) / s;
    q.z = 0.25 * s;
  }
  return q;
}

bool Finite(double v) { return std::isfinite(v); }

// The knuckle aim/tip pose, relative to the grip pose. This is the same offset the
// host used to send as the device pose, and the same "tip" in the Index render model.
vr::HmdMatrix34_t AimOffsetMatrix() {
  constexpr double d = 3.141592653589793 / 180.0;
  const double cx = std::cos(-40 * d), sx = std::sin(-40 * d);
  const double cy = std::cos(5 * d), sy = std::sin(5 * d);
  vr::HmdMatrix34_t m{};
  m.m[0][0] = cy;        m.m[0][1] = 0;  m.m[0][2] = sy;
  m.m[1][0] = sx * sy;   m.m[1][1] = cx; m.m[1][2] = -sx * cy;
  m.m[2][0] = -cx * sy;  m.m[2][1] = sx; m.m[2][2] = cx * cy;
  m.m[0][3] = -0.006;
  m.m[1][3] = -0.015;
  m.m[2][3] = 0.020;
  return m;
}

}  // namespace

double FrameInput::Now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

FrameInput::~FrameInput() { Stop(); }

bool FrameInput::Start() {
  const char *rt = getenv("XDG_RUNTIME_DIR");
  if (!rt || !*rt) {
    Log("no XDG_RUNTIME_DIR");
    return false;
  }
  const std::string dir = std::string(rt) + "/questlhsync";
  mkdir(dir.c_str(), 0700);
  path_ = dir + "/input.sock";
  fd_ = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
  if (fd_ < 0) {
    Log("socket: %s", strerror(errno));
    return false;
  }
  sockaddr_un a{};
  a.sun_family = AF_UNIX;
  if (path_.size() >= sizeof a.sun_path) {
    Log("socket path too long");
    close(fd_);
    fd_ = -1;
    return false;
  }
  memcpy(a.sun_path, path_.c_str(), path_.size() + 1);
  unlink(path_.c_str());
  if (bind(fd_, (sockaddr *)&a, sizeof a) != 0) {
    Log("bind %s: %s", path_.c_str(), strerror(errno));
    close(fd_);
    fd_ = -1;
    return false;
  }
  chmod(path_.c_str(), 0600);
  if (!vr::VRServerDriverHost()->TrackedDeviceAdded("QLHS-KNUCKLES-R",
                                                    vr::TrackedDeviceClass_Controller, &device_)) {
    Log("TrackedDeviceAdded failed");
    close(fd_);
    unlink(path_.c_str());
    fd_ = -1;
    return false;
  }
  run_ = true;
  thread_ = std::thread(&FrameInput::Serve, this);
  Log("listening on %s", path_.c_str());
  return true;
}

void FrameInput::Stop() {
  run_ = false;
  if (thread_.joinable()) thread_.join();
  if (fd_ >= 0) {
    close(fd_);
    unlink(path_.c_str());
    fd_ = -1;
  }
}

void FrameInput::Serve() {
  while (run_) {
    pollfd p{fd_, POLLIN, 0};
    if (poll(&p, 1, 200) <= 0) continue;
    char b[512];
    const ssize_t n = recv(fd_, b, sizeof b - 1, 0);
    if (n <= 0) continue;
    b[n] = 0;
    Parse(b, (size_t)n);
  }
}

void FrameInput::Parse(const char *line, size_t len) {
  (void)len;
  int link = 0;
  if (line[0] == 'G' && line[1] == ' ' && sscanf(line, "G %d", &link) == 1) {
    const bool up = link != 0;
    if (link_up_.exchange(up) != up) Log("input link %s", up ? "up" : "down");
    return;
  }
  int hand = 0;
  unsigned seq = 0;
  long long qpc = 0;
  unsigned buttons = 0;
  float trigger = 0, grip = 0, sx = 0, sy = 0;
  float pad_x = 0, pad_y = 0;
  int pad_touch = 0;
  float battery = -1;
  int charging = 0;
  double qw = 1, qx = 0, qy = 0, qz = 0, px = 0, py = 0, pz = 0;
  const int n = sscanf(line,
                       "X %d %u %lld %u %f %f %f %f %lf %lf %lf %lf %lf %lf %lf %f %f %d %f %d",
                       &hand, &seq, &qpc, &buttons, &trigger, &grip, &sx, &sy, &qw, &qx, &qy,
                       &qz, &px, &py, &pz, &pad_x, &pad_y, &pad_touch, &battery, &charging);
  if (n < 15) return;
  if (hand != 1) return;  // right hand only for the dashboard
  const double norm = qw * qw + qx * qx + qy * qy + qz * qz;
  if (!Finite(trigger) || !Finite(grip) || !Finite(sx) || !Finite(sy) || !Finite(norm) ||
      std::abs(norm - 1) > 0.1 || !Finite(px) || !Finite(py) || !Finite(pz) || !Finite(pad_x) ||
      !Finite(pad_y))
    return;
  Sample s;
  s.seq = seq;
  s.buttons = buttons;
  s.trigger = std::max(0.f, std::min(1.f, trigger));
  s.grip = std::max(0.f, std::min(1.f, grip));
  s.stick_x = std::max(-1.f, std::min(1.f, sx));
  s.stick_y = std::max(-1.f, std::min(1.f, sy));
  s.pad_x = std::max(-1.f, std::min(1.f, pad_x));
  s.pad_y = std::max(-1.f, std::min(1.f, pad_y));
  s.pad_touch = pad_touch != 0;
  s.battery = (n >= 20 && Finite(battery) && battery >= 0.f)
                  ? std::max(0.f, std::min(1.f, battery))
                  : -1.f;
  s.charging = charging != 0;
  s.q[0] = qw; s.q[1] = qx; s.q[2] = qy; s.q[3] = qz;
  s.p[0] = px; s.p[1] = py; s.p[2] = pz;
  s.rx = std::chrono::steady_clock::now();
  if (s.buttons) Log("X buttons 0x%x", s.buttons);
  pending_buttons_.fetch_or(s.buttons);
  {
    std::lock_guard<std::mutex> g(m_);
    latest_ = s;
    have_sample_ = true;
  }
  (void)qpc;
  if (seq % 120 == 0) Log("sample %u hand=%d", seq, hand);
}

void FrameInput::RunFrame() {
  Sample s;
  bool have = false;
  {
    std::lock_guard<std::mutex> g(m_);
    s = latest_;
    have = have_sample_;
  }
  const double now = Now();
  bool fresh = false;
  if (have && link_up_.load()) {
    const auto age = std::chrono::duration<double>(std::chrono::steady_clock::now() - s.rx).count();
    fresh = age >= 0 && age < 0.5;
  }
  device_.Tick(s, fresh, now, pending_buttons_.exchange(0));
}

void FrameInput::Device::SetRole(bool active) {
  if (container_ == vr::k_ulInvalidPropertyContainer) return;
  vr::VRProperties()->SetInt32Property(container_, vr::Prop_ControllerRoleHint_Int32,
                                       active ? vr::TrackedControllerRole_RightHand
                                              : vr::TrackedControllerRole_OptOut);
  vr::VRProperties()->SetInt32Property(container_, vr::Prop_ControllerHandSelectionPriority_Int32,
                                       active ? 1000 : -1000);
}

void FrameInput::Device::SetBool(vr::VRInputComponentHandle_t h, bool v) {
  if (h) vr::VRDriverInput()->UpdateBooleanComponent(h, v, 0);
}
void FrameInput::Device::SetScalar(vr::VRInputComponentHandle_t h, float v) {
  if (h) vr::VRDriverInput()->UpdateScalarComponent(h, v, 0);
}

void FrameInput::Device::ReleaseInputs() {
  SetBool(system_click_, false); SetBool(system_touch_, false);
  SetBool(a_click_, false); SetBool(a_touch_, false);
  SetBool(b_click_, false); SetBool(b_touch_, false);
  SetScalar(trigger_value_, 0); SetBool(trigger_click_, false); SetBool(trigger_touch_, false);
  SetScalar(grip_value_, 0); SetBool(grip_click_, false); SetBool(grip_touch_, false);
  SetScalar(stick_x_, 0); SetScalar(stick_y_, 0); SetBool(stick_click_, false); SetBool(stick_touch_, false);
  SetScalar(pad_x_, 0); SetScalar(pad_y_, 0); SetBool(pad_touch_, false); SetBool(pad_click_, false);
  system_pulse_.Reset(); a_pulse_.Reset(); b_pulse_.Reset(); stick_pulse_.Reset(); pad_pulse_.Reset();
  trigger_pulse_.Reset(); last_trigger_click_ = false;
}

void FrameInput::Device::Tick(const Sample &s, bool fresh, double now, uint32_t pulse_bits) {
  if (index_ == vr::k_unTrackedDeviceIndexInvalid) return;
  if (!fresh) {
    if (active_) {
      Log("input stale: releasing right hand");
      active_ = false;
      active_frames_ = 0;
      pending_buttons_ = 0;
      ReleaseInputs();
      SetRole(false);
    }
    vr::DriverPose_t p{};
    p.qWorldFromDriverRotation.w = 1;
    p.qDriverFromHeadRotation.w = 1;
    p.deviceIsConnected = false;
    p.poseIsValid = false;
    p.result = vr::TrackingResult_Uninitialized;
    last_pose_ = p;
    vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_, p, sizeof p);
    return;
  }
  if (!active_) {
    active_ = true;
    active_frames_ = 0;
    last_seq_ = s.seq - 1;
    SetRole(true);
    Log("input live: taking right hand");
  }
  vr::TrackedDevicePose_t hmd{};
  vr::VRServerDriverHost()->GetRawTrackedDevicePoses(0.f, &hmd, 1);
  Quat hq;
  double hp[3] = {0, 0, 0};
  if (hmd.bPoseIsValid) {
    hq = FromMatrix(hmd.mDeviceToAbsoluteTracking);
    hp[0] = hmd.mDeviceToAbsoluteTracking.m[0][3];
    hp[1] = hmd.mDeviceToAbsoluteTracking.m[1][3];
    hp[2] = hmd.mDeviceToAbsoluteTracking.m[2][3];
  }
  double wp[3], rel[3] = {s.p[0], s.p[1], s.p[2]};
  Rotate(hq, rel, wp);
  for (int i = 0; i < 3; i++) wp[i] += hp[i];
  const Quat wq = Mul(hq, Quat{s.q[0], s.q[1], s.q[2], s.q[3]});
  vr::DriverPose_t p{};
  p.qWorldFromDriverRotation.w = 1;
  p.qDriverFromHeadRotation.w = 1;
  p.vecPosition[0] = wp[0]; p.vecPosition[1] = wp[1]; p.vecPosition[2] = wp[2];
  p.qRotation = {wq.w, wq.x, wq.y, wq.z};
  p.deviceIsConnected = true;
  p.poseIsValid = true;
  p.result = vr::TrackingResult_Running_OK;
  last_pose_ = p;
  vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_, p, sizeof p);
  if (aim_pose_) {
    const auto offset = AimOffsetMatrix();
    vr::VRDriverInput()->UpdatePoseComponent(aim_pose_, &offset, 0);
  }

  last_seq_ = s.seq;
  pending_buttons_ |= pulse_bits;
  if (active_frames_ < 2) {
    ++active_frames_;
  } else if (pending_buttons_) {
    Log("pulse bits 0x%x", pending_buttons_);
    if (pending_buttons_ & 1) system_pulse_.Fire(now);
    if (pending_buttons_ & 2) a_pulse_.Fire(now);
    if (pending_buttons_ & 4) b_pulse_.Fire(now);
    if (pending_buttons_ & 8) stick_pulse_.Fire(now);
    if (pending_buttons_ & 16) pad_pulse_.Fire(now);
    if (pending_buttons_ & 32) trigger_pulse_.Fire(now);
    pending_buttons_ = 0;
  }
  const bool sys_down = system_pulse_.Update(now);
  if (sys_down && !last_system_) Log("system click down");
  last_system_ = sys_down;
  SetBool(system_click_, sys_down);
  SetBool(system_touch_, sys_down);
  SetBool(a_click_, a_pulse_.Update(now));
  SetBool(a_touch_, a_pulse_.Update(now));
  SetBool(b_click_, b_pulse_.Update(now));
  SetBool(b_touch_, b_pulse_.Update(now));
  const bool trig_click = trigger_pulse_.Update(now);
  if (trig_click && !last_trigger_click_)
    Log("trigger click down (%.2f)%s", s.trigger, trigger_pulse_.down ? " hid" : "");
  else if (!trig_click && last_trigger_click_)
    Log("trigger click up");
  last_trigger_click_ = trig_click;
  SetScalar(trigger_value_, s.trigger);
  SetBool(trigger_click_, trig_click);
  SetBool(trigger_touch_, s.trigger > 0.05f || trigger_pulse_.down);
  SetScalar(grip_value_, s.grip);
  SetBool(grip_click_, s.grip >= 0.5f);
  SetBool(grip_touch_, s.grip > 0.05f);
  SetScalar(stick_x_, s.stick_x);
  SetScalar(stick_y_, s.stick_y);
  SetBool(stick_click_, stick_pulse_.Update(now));
  SetBool(stick_touch_, std::hypot(s.stick_x, s.stick_y) > 0.1f);
  const bool pad_click = pad_pulse_.Update(now);
  SetBool(pad_click_, pad_click);
  SetBool(pad_touch_, s.pad_touch || pad_click);
  SetScalar(pad_x_, s.pad_x);
  SetScalar(pad_y_, s.pad_y);
  auto *props = vr::VRProperties();
  if (s.battery >= 0 && (!battery_seen_ || std::fabs(s.battery - last_battery_) > 0.001f)) {
    props->SetFloatProperty(container_, vr::Prop_DeviceBatteryPercentage_Float, s.battery);
    last_battery_ = s.battery;
    battery_seen_ = true;
  }
  if (s.battery >= 0 && s.charging != last_charging_) {
    props->SetBoolProperty(container_, vr::Prop_DeviceIsCharging_Bool, s.charging);
    last_charging_ = s.charging;
  }
}

FrameInput::Device::Device() {
  last_pose_.qWorldFromDriverRotation.w = 1;
  last_pose_.qDriverFromHeadRotation.w = 1;
  last_pose_.result = vr::TrackingResult_Uninitialized;
}

vr::EVRInitError FrameInput::Device::Activate(uint32_t id) {
  index_ = id;
  container_ = vr::VRProperties()->TrackedDeviceToPropertyContainer(id);
  if (container_ == vr::k_ulInvalidPropertyContainer) return vr::VRInitError_Driver_Failed;
  auto props = vr::VRProperties();
  props->SetStringProperty(container_, vr::Prop_SerialNumber_String, "QLHS-KNUCKLES-R");
  props->SetStringProperty(container_, vr::Prop_TrackingSystemName_String, "questlhsync");
  props->SetStringProperty(container_, vr::Prop_ModelNumber_String, "Knuckles (QuestLHSync)");
  props->SetStringProperty(container_, vr::Prop_ManufacturerName_String, "QuestLHSync");
  props->SetStringProperty(container_, vr::Prop_ControllerType_String, "questlhsync_knuckles");
  props->SetStringProperty(container_, vr::Prop_InputProfilePath_String,
                           "{questlhsync_frame}/input/questlhsync_knuckles_profile.json");
  props->SetStringProperty(container_, vr::Prop_RenderModelName_String,
                           "{indexcontroller}valve_controller_knu_1_0_right");
  props->SetBoolProperty(container_, vr::Prop_DeviceProvidesBatteryStatus_Bool, true);
  props->SetBoolProperty(container_, vr::Prop_DeviceIsWireless_Bool, true);
  SetRole(false);

  auto input = vr::VRDriverInput();
  auto boolean = [&](const char *path) {
    vr::VRInputComponentHandle_t h = 0;
    const auto e = input->CreateBooleanComponent(container_, path, &h);
    if (e != vr::VRInputError_None) Log("create %s: %d", path, (int)e);
    return h;
  };
  auto scalar = [&](const char *path, bool two_sided) {
    vr::VRInputComponentHandle_t h = 0;
    const auto e = input->CreateScalarComponent(
        container_, path, &h, vr::VRScalarType_Absolute,
        two_sided ? vr::VRScalarUnits_NormalizedTwoSided : vr::VRScalarUnits_NormalizedOneSided);
    if (e != vr::VRInputError_None) Log("create %s: %d", path, (int)e);
    return h;
  };
  system_click_ = boolean("/input/system/click");
  system_touch_ = boolean("/input/system/touch");
  a_click_ = boolean("/input/a/click");
  a_touch_ = boolean("/input/a/touch");
  b_click_ = boolean("/input/b/click");
  b_touch_ = boolean("/input/b/touch");
  trigger_value_ = scalar("/input/trigger/value", false);
  trigger_click_ = boolean("/input/trigger/click");
  trigger_touch_ = boolean("/input/trigger/touch");
  grip_value_ = scalar("/input/grip/value", false);
  grip_click_ = boolean("/input/grip/click");
  grip_touch_ = boolean("/input/grip/touch");
  stick_x_ = scalar("/input/thumbstick/x", true);
  stick_y_ = scalar("/input/thumbstick/y", true);
  stick_click_ = boolean("/input/thumbstick/click");
  stick_touch_ = boolean("/input/thumbstick/touch");
  pad_x_ = scalar("/input/trackpad/x", true);
  pad_y_ = scalar("/input/trackpad/y", true);
  pad_touch_ = boolean("/input/trackpad/touch");
  pad_click_ = boolean("/input/trackpad/click");
  const auto pe = input->CreatePoseComponent(container_, "/pose/aim", &aim_pose_);
  if (pe != vr::VRInputError_None) Log("create /pose/aim: %d", (int)pe);
  if (aim_pose_) {
    const auto offset = AimOffsetMatrix();
    input->UpdatePoseComponent(aim_pose_, &offset, 0);
  }
  ReleaseInputs();
  Log("virtual controller activated as device %u", id);
  return vr::VRInitError_None;
}

void FrameInput::Device::Deactivate() {
  index_ = vr::k_unTrackedDeviceIndexInvalid;
  container_ = vr::k_ulInvalidPropertyContainer;
}

void FrameInput::Device::EnterStandby() {
  if (index_ != vr::k_unTrackedDeviceIndexInvalid) {
    vr::DriverPose_t p{};
    p.qWorldFromDriverRotation.w = 1;
    p.qDriverFromHeadRotation.w = 1;
    p.result = vr::TrackingResult_Uninitialized;
    vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_, p, sizeof p);
  }
}

void FrameInput::Device::DebugRequest(const char *request, char *response, uint32_t size) {
  (void)request;
  if (!response || !size) return;
  snprintf(response, size, "connected=%d active=%d seq=%u", last_pose_.deviceIsConnected ? 1 : 0,
           active_ ? 1 : 0, last_seq_);
}
