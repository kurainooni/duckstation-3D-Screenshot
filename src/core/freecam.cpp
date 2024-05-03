#include "freecam.h"
#include "settings.h"

#include "imgui.h"
#include "util/imgui_manager.h"

#include <cmath>

namespace Freecam {
namespace {

struct Config
{
  bool enabled = false;
  s32 angle_x = 0;
  s32 angle_y = 0;
  s32 angle_z = 0;
  s32 move_x = 0;
  s32 move_y = 0;
  s32 move_z = 0;
};

} // namespace

static Config s_config;

static void Rotate2D(s32 angle, s64& x, s64& y)
{
  const float radians = angle * (2.f * M_PI / 360.f);
  const s64 s = roundf(sinf(radians) * 4096.f);
  const s64 c = roundf(cosf(radians) * 4096.f);

  const s64 new_x = ( c * x + s * y) >> 12;
  const s64 new_y = (-s * x + c * y) >> 12;

  x = new_x;
  y = new_y;
}

// x,y,z are in high precision, before the 12 fractional bits are
// shifted off.
void ApplyToVertex(s64& x, s64& y, s64& z)
{
  // Disable when the GUI hidden to avoid confusion
  if (!g_settings.debugging.show_freecam)
    return;

  if (!s_config.enabled)
    return;

  x += (s64)s_config.move_x << 12;
  y -= (s64)s_config.move_y << 12;
  z -= (s64)s_config.move_z << 12;

  Rotate2D(s_config.angle_x, y, z);
  Rotate2D(s_config.angle_y, x, z);
  Rotate2D(s_config.angle_z, x, y);
}

// Config is double-buffered to prevent "tearing" (not sure this is
// necessary). This is the back-buffer that the GUI modifies.
static Config s_ui_config;

void NextFrame()
{
  s_config = s_ui_config;
}

void DrawGuiWindow()
{
  const float framebuffer_scale = Host::GetOSDScale();

  ImGui::SetNextWindowSize(
    ImVec2(500.0f * framebuffer_scale, 233.0f * framebuffer_scale),
    ImGuiCond_Once
  );

  if (!ImGui::Begin("Freecam", nullptr))
  {
    ImGui::End();
    return;
  }

  ImGui::Checkbox("Enabled", &s_ui_config.enabled);

  if (!s_ui_config.enabled)
    ImGui::BeginDisabled();

  ImGui::SliderInt("Rotation X", &s_ui_config.angle_x, -180, 180);
  ImGui::SliderInt("Rotation Y", &s_ui_config.angle_y, -180, 180);
  ImGui::SliderInt("Rotation Z", &s_ui_config.angle_z, -180, 180);

  ImGui::SliderInt("Move X", &s_ui_config.move_x, -4000, 4000);
  ImGui::SliderInt("Move Y", &s_ui_config.move_y, -4000, 4000);
  ImGui::SliderInt("Move Z", &s_ui_config.move_z, -4000, 4000);

  if (!s_ui_config.enabled)
    ImGui::EndDisabled();

  if (ImGui::Button("Reset Rotation"))
    s_ui_config.angle_x = s_ui_config.angle_y = s_ui_config.angle_z = 0;
  ImGui::SameLine();
  if (ImGui::Button("Reset Move"))
    s_ui_config.move_x = s_ui_config.move_y = s_ui_config.move_z = 0;

  ImGui::End();
}

} // namespace Freecam
