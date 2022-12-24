// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "common_host.h"
#include "IconsFontAwesome5.h"
#include "common/assert.h"
#include "common/byte_stream.h"
#include "common/crash_handler.h"
#include "common/file_system.h"
#include "common/log.h"
#include "common/path.h"
#include "common/string_util.h"
#include "core/cdrom.h"
#include "core/cheats.h"
#include "core/controller.h"
#include "core/cpu_code_cache.h"
#include "core/dma.h"
#include "core/gpu.h"
#include "core/gte.h"
#include "core/host.h"
#include "core/host_display.h"
#include "core/host_settings.h"
#include "core/mdec.h"
#include "core/pgxp.h"
#include "core/save_state_version.h"
#include "core/settings.h"
#include "core/spu.h"
#include "core/system.h"
#include "core/texture_replacements.h"
#include "core/timers.h"
#include "fullscreen_ui.h"
#include "game_list.h"
#include "icon.h"
#include "imgui.h"
#include "imgui_fullscreen.h"
#include "imgui_manager.h"
#include "imgui_overlays.h"
#include "input_manager.h"
#include "platform_misc.h"
#include "scmversion/scmversion.h"
#include "util/audio_stream.h"
#include "util/ini_settings_interface.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

#ifdef WITH_DISCORD_PRESENCE
#include "discord_rpc.h"
#endif

#ifdef WITH_CHEEVOS
#include "achievements.h"
#endif

#ifdef _WIN32
#include "common/windows_headers.h"
#include "frontend-common/d3d11_host_display.h"
#include "frontend-common/d3d12_host_display.h"
#include <KnownFolders.h>
#include <ShlObj.h>
#include <mmsystem.h>
#endif

#ifdef WITH_OPENGL
#include "frontend-common/opengl_host_display.h"
#endif

#ifdef WITH_VULKAN
#include "frontend-common/vulkan_host_display.h"
#endif

Log_SetChannel(CommonHostInterface);

namespace CommonHost {
static void UpdateSessionTime(const std::string& new_serial);

#ifdef WITH_DISCORD_PRESENCE
static void SetDiscordPresenceEnabled(bool enabled);
static void InitializeDiscordPresence();
static void ShutdownDiscordPresence();
static void UpdateDiscordPresence(bool rich_presence_only);
static void PollDiscordPresence();
#endif
} // namespace CommonHost

// Used to track play time. We use a monotonic timer here, in case of clock changes.
static u64 s_session_start_time = 0;
static std::string s_session_serial;

#ifdef WITH_DISCORD_PRESENCE
// discord rich presence
bool m_discord_presence_enabled = false;
bool m_discord_presence_active = false;
#ifdef WITH_CHEEVOS
std::string m_discord_presence_cheevos_string;
#endif
#endif

void CommonHost::Initialize()
{
  // This will call back to Host::LoadSettings() -> ReloadSources().
  System::LoadSettings(false);

#ifdef WITH_CHEEVOS
#ifdef WITH_RAINTEGRATION
  if (Host::GetBaseBoolSettingValue("Cheevos", "UseRAIntegration", false))
    Achievements::SwitchToRAIntegration();
#endif
  if (g_settings.achievements_enabled)
    Achievements::Initialize();
#endif
}

void CommonHost::Shutdown()
{
#ifdef WITH_DISCORD_PRESENCE
  CommonHost::ShutdownDiscordPresence();
#endif

#ifdef WITH_CHEEVOS
  Achievements::Shutdown();
#endif

  InputManager::CloseSources();
}

void CommonHost::PumpMessagesOnCPUThread()
{
  InputManager::PollSources();

#ifdef WITH_DISCORD_PRESENCE
  PollDiscordPresence();
#endif

#ifdef WITH_CHEEVOS
  if (Achievements::IsActive())
    Achievements::FrameUpdate();
#endif
}

std::unique_ptr<HostDisplay> Host::CreateDisplayForAPI(RenderAPI api)
{
  switch (api)
  {
#ifdef WITH_VULKAN
    case RenderAPI::Vulkan:
      return std::make_unique<VulkanHostDisplay>();
#endif

#ifdef WITH_OPENGL
    case RenderAPI::OpenGL:
    case RenderAPI::OpenGLES:
      return std::make_unique<OpenGLHostDisplay>();
#endif

#ifdef _WIN32
    case RenderAPI::D3D12:
      return std::make_unique<D3D12HostDisplay>();

    case RenderAPI::D3D11:
      return std::make_unique<D3D11HostDisplay>();
#endif

    default:
#if defined(_WIN32) && defined(_M_ARM64)
      return std::make_unique<D3D12HostDisplay>();
#elif defined(_WIN32)
      return std::make_unique<D3D11HostDisplay>();
#elif defined(WITH_OPENGL)
      return std::make_unique<OpenGLHostDisplay>();
#elif defined(WITH_VULKAN)
      return std::make_unique<VulkanHostDisplay>();
#else
      return {};
#endif
  }
}

bool CommonHost::CreateHostDisplayResources()
{
  return true;
}

void CommonHost::ReleaseHostDisplayResources()
{
  SaveStateSelectorUI::DestroyTextures();
}

#ifndef __ANDROID__

std::unique_ptr<AudioStream> Host::CreateAudioStream(AudioBackend backend, u32 sample_rate, u32 channels, u32 buffer_ms,
                                                     u32 latency_ms, AudioStretchMode stretch)
{
  switch (backend)
  {
#ifdef WITH_CUBEB
    case AudioBackend::Cubeb:
      return CommonHost::CreateCubebAudioStream(sample_rate, channels, buffer_ms, latency_ms, stretch);
#endif

#ifdef _WIN32
    case AudioBackend::XAudio2:
      return CommonHost::CreateXAudio2Stream(sample_rate, channels, buffer_ms, latency_ms, stretch);
#endif

    case AudioBackend::Null:
      return AudioStream::CreateNullStream(sample_rate, channels, buffer_ms);

    default:
      return nullptr;
  }
}

#endif

void CommonHost::UpdateLogSettings()
{
  Log::SetFilterLevel(g_settings.log_level);
  Log::SetConsoleOutputParams(g_settings.log_to_console,
                              g_settings.log_filter.empty() ? nullptr : g_settings.log_filter.c_str(),
                              g_settings.log_level);
  Log::SetDebugOutputParams(g_settings.log_to_debug,
                            g_settings.log_filter.empty() ? nullptr : g_settings.log_filter.c_str(),
                            g_settings.log_level);

  if (g_settings.log_to_file)
  {
    Log::SetFileOutputParams(g_settings.log_to_file, Path::Combine(EmuFolders::DataRoot, "duckstation.log").c_str(),
                             true, g_settings.log_filter.empty() ? nullptr : g_settings.log_filter.c_str(),
                             g_settings.log_level);
  }
  else
  {
    Log::SetFileOutputParams(false, nullptr);
  }
}

void CommonHost::OnSystemStarting()
{
  //
}

void CommonHost::OnSystemStarted()
{
  FullscreenUI::OnSystemStarted();

  if (g_settings.inhibit_screensaver)
    FrontendCommon::SuspendScreensaver(g_host_display->GetWindowInfo());
}

void CommonHost::OnSystemPaused()
{
  FullscreenUI::OnSystemPaused();

  InputManager::PauseVibration();

  if (g_settings.inhibit_screensaver)
    FrontendCommon::ResumeScreensaver();
}

void CommonHost::OnSystemResumed()
{
  FullscreenUI::OnSystemResumed();

  if (g_settings.inhibit_screensaver)
    FrontendCommon::SuspendScreensaver(g_host_display->GetWindowInfo());
}

void CommonHost::OnSystemDestroyed()
{
  Host::ClearOSDMessages();

  SaveStateSelectorUI::Close(true);
  FullscreenUI::OnSystemDestroyed();

  InputManager::PauseVibration();

  if (g_settings.inhibit_screensaver)
    FrontendCommon::ResumeScreensaver();
}

void CommonHost::OnGameChanged(const std::string& disc_path, const std::string& game_serial,
                               const std::string& game_name)
{
#ifdef WITH_DISCORD_PRESENCE
  UpdateDiscordPresence(false);
#endif

  UpdateSessionTime(game_serial);

  SaveStateSelectorUI::RefreshList();
}

void CommonHost::SetDefaultSettings(SettingsInterface& si)
{
#ifdef WITH_DISCORD_PRESENCE
  si.SetBoolValue("Main", "EnableDiscordPresence", false);
#endif

#if defined(WITH_CHEEVOS) && defined(WITH_RAINTEGRATION)
  si.SetBoolValue("Cheevos", "UseRAIntegration", false);
#endif
}

void CommonHost::SetDefaultControllerSettings(SettingsInterface& si)
{
  InputManager::SetDefaultConfig(si);

  // Global Settings
  si.SetStringValue("ControllerPorts", "MultitapMode", Settings::GetMultitapModeName(Settings::DEFAULT_MULTITAP_MODE));
  si.SetFloatValue("ControllerPorts", "PointerXScale", 8.0f);
  si.SetFloatValue("ControllerPorts", "PointerYScale", 8.0f);
  si.SetBoolValue("ControllerPorts", "PointerXInvert", false);
  si.SetBoolValue("ControllerPorts", "PointerYInvert", false);

  // Default pad types and parameters.
  for (u32 i = 0; i < NUM_CONTROLLER_AND_CARD_PORTS; i++)
  {
    const std::string section(Controller::GetSettingsSection(i));
    si.ClearSection(section.c_str());
    si.SetStringValue(section.c_str(), "Type", Controller::GetDefaultPadType(i));
  }

#ifndef __ANDROID__
  // Use the automapper to set this up.
  InputManager::MapController(si, 0, InputManager::GetGenericBindingMapping("Keyboard"));
#endif
}

void CommonHost::SetDefaultHotkeyBindings(SettingsInterface& si)
{
  si.ClearSection("Hotkeys");

#ifndef __ANDROID__
  si.SetStringValue("Hotkeys", "FastForward", "Keyboard/Tab");
  si.SetStringValue("Hotkeys", "TogglePause", "Keyboard/Space");
  si.SetStringValue("Hotkeys", "Screenshot", "Keyboard/F10");
  si.SetStringValue("Hotkeys", "ToggleFullscreen", "Keyboard/F11");

  si.SetStringValue("Hotkeys", "OpenPauseMenu", "Keyboard/Escape");
  si.SetStringValue("Hotkeys", "LoadSelectedSaveState", "Keyboard/F1");
  si.SetStringValue("Hotkeys", "SaveSelectedSaveState", "Keyboard/F2");
  si.SetStringValue("Hotkeys", "SelectPreviousSaveStateSlot", "Keyboard/F3");
  si.SetStringValue("Hotkeys", "SelectNextSaveStateSlot", "Keyboard/F4");
#endif
}

void CommonHost::LoadSettings(SettingsInterface& si, std::unique_lock<std::mutex>& lock)
{
  UpdateLogSettings();
  InputManager::ReloadSources(si, lock);
  InputManager::ReloadBindings(si, *Host::GetSettingsInterfaceForBindings());

#ifdef WITH_DISCORD_PRESENCE
  SetDiscordPresenceEnabled(si.GetBoolValue("Main", "EnableDiscordPresence", false));
#endif
}

void CommonHost::CheckForSettingsChanges(const Settings& old_settings)
{
  if (System::IsValid())
  {
    if (g_settings.inhibit_screensaver != old_settings.inhibit_screensaver)
    {
      if (g_settings.inhibit_screensaver)
        FrontendCommon::SuspendScreensaver(g_host_display->GetWindowInfo());
      else
        FrontendCommon::ResumeScreensaver();
    }
  }

#ifdef WITH_CHEEVOS
  Achievements::UpdateSettings(old_settings);
#endif

  FullscreenUI::CheckForConfigChanges(old_settings);

  if (g_settings.log_level != old_settings.log_level || g_settings.log_filter != old_settings.log_filter ||
      g_settings.log_to_console != old_settings.log_to_console ||
      g_settings.log_to_debug != old_settings.log_to_debug || g_settings.log_to_window != old_settings.log_to_window ||
      g_settings.log_to_file != old_settings.log_to_file)
  {
    UpdateLogSettings();
  }
}

void CommonHost::UpdateSessionTime(const std::string& new_serial)
{
  if (s_session_serial == new_serial)
    return;

  const u64 ctime = Common::Timer::GetCurrentValue();
  if (!s_session_serial.empty())
  {
    // round up to seconds
    const std::time_t etime =
      static_cast<std::time_t>(std::round(Common::Timer::ConvertValueToSeconds(ctime - s_session_start_time)));
    const std::time_t wtime = std::time(nullptr);
    GameList::AddPlayedTimeForSerial(s_session_serial, wtime, etime);
  }

  s_session_serial = new_serial;
  s_session_start_time = ctime;
}

u64 CommonHost::GetSessionPlayedTime()
{
  const u64 ctime = Common::Timer::GetCurrentValue();
  return static_cast<u64>(std::round(Common::Timer::ConvertValueToSeconds(ctime - s_session_start_time)));
}

void Host::SetPadVibrationIntensity(u32 pad_index, float large_or_single_motor_intensity, float small_motor_intensity)
{
  InputManager::SetPadVibrationIntensity(pad_index, large_or_single_motor_intensity, small_motor_intensity);
}

void Host::DisplayLoadingScreen(const char* message, int progress_min /*= -1*/, int progress_max /*= -1*/,
                                int progress_value /*= -1*/)
{
  const auto& io = ImGui::GetIO();
  const float scale = ImGuiManager::GetGlobalScale();
  const float width = (400.0f * scale);
  const bool has_progress = (progress_min < progress_max);

  // eat the last imgui frame, it might've been partially rendered by the caller.
  ImGui::EndFrame();
  ImGui::NewFrame();

  const float logo_width = 260.0f * scale;
  const float logo_height = 260.0f * scale;

  ImGui::SetNextWindowSize(ImVec2(logo_width, logo_height), ImGuiCond_Always);
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, (io.DisplaySize.y * 0.5f) - (50.0f * scale)),
                          ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  if (ImGui::Begin("LoadingScreenLogo", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoBackground))
  {
    GPUTexture* tex = ImGuiFullscreen::GetCachedTexture("images/duck.png");
    if (tex)
      ImGui::Image(tex, ImVec2(logo_width, logo_height));
  }
  ImGui::End();

  ImGui::SetNextWindowSize(ImVec2(width, (has_progress ? 50.0f : 30.0f) * scale), ImGuiCond_Always);
  ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, (io.DisplaySize.y * 0.5f) + (100.0f * scale)),
                          ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  if (ImGui::Begin("LoadingScreen", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing))
  {
    if (has_progress)
    {
      ImGui::Text("%s: %d/%d", message, progress_value, progress_max);
      ImGui::ProgressBar(static_cast<float>(progress_value) / static_cast<float>(progress_max - progress_min),
                         ImVec2(-1.0f, 0.0f), "");
      Log_InfoPrintf("%s: %d/%d", message, progress_value, progress_max);
    }
    else
    {
      const ImVec2 text_size(ImGui::CalcTextSize(message));
      ImGui::SetCursorPosX((width - text_size.x) / 2.0f);
      ImGui::TextUnformatted(message);
      Log_InfoPrintf("%s", message);
    }
  }
  ImGui::End();

  ImGui::EndFrame();
  g_host_display->Render(false);
  ImGui::NewFrame();
}

void ImGuiManager::RenderDebugWindows()
{
  if (System::IsValid())
  {
    if (g_settings.debugging.show_gpu_state)
      g_gpu->DrawDebugStateWindow();
    if (g_settings.debugging.show_cdrom_state)
      g_cdrom.DrawDebugWindow();
    if (g_settings.debugging.show_timers_state)
      g_timers.DrawDebugStateWindow();
    if (g_settings.debugging.show_spu_state)
      SPU::DrawDebugStateWindow();
    if (g_settings.debugging.show_mdec_state)
      MDEC::DrawDebugStateWindow();
    if (g_settings.debugging.show_dma_state)
      g_dma.DrawDebugStateWindow();
  }
}

#ifdef WITH_DISCORD_PRESENCE

void CommonHost::SetDiscordPresenceEnabled(bool enabled)
{
  if (m_discord_presence_enabled == enabled)
    return;

  m_discord_presence_enabled = enabled;
  if (enabled)
    InitializeDiscordPresence();
  else
    ShutdownDiscordPresence();
}

void CommonHost::InitializeDiscordPresence()
{
  if (m_discord_presence_active)
    return;

  DiscordEventHandlers handlers = {};
  Discord_Initialize("705325712680288296", &handlers, 0, nullptr);
  m_discord_presence_active = true;

  UpdateDiscordPresence(false);
}

void CommonHost::ShutdownDiscordPresence()
{
  if (!m_discord_presence_active)
    return;

  Discord_ClearPresence();
  Discord_Shutdown();
  m_discord_presence_active = false;
#ifdef WITH_CHEEVOS
  m_discord_presence_cheevos_string.clear();
#endif
}

void CommonHost::UpdateDiscordPresence(bool rich_presence_only)
{
  if (!m_discord_presence_active)
    return;

#ifdef WITH_CHEEVOS
  // Update only if RetroAchievements rich presence has changed
  const std::string& new_rich_presence = Achievements::GetRichPresenceString();
  if (new_rich_presence == m_discord_presence_cheevos_string && rich_presence_only)
  {
    return;
  }
  m_discord_presence_cheevos_string = new_rich_presence;
#else
  if (rich_presence_only)
  {
    return;
  }
#endif

  // https://discord.com/developers/docs/rich-presence/how-to#updating-presence-update-presence-payload-fields
  DiscordRichPresence rp = {};
  rp.largeImageKey = "duckstation_logo";
  rp.largeImageText = "DuckStation PS1/PSX Emulator";
  rp.startTimestamp = std::time(nullptr);

  SmallString details_string;
  if (!System::IsShutdown())
  {
    details_string.AppendFormattedString("%s (%s)", System::GetRunningTitle().c_str(),
                                         System::GetRunningSerial().c_str());
  }
  else
  {
    details_string.AppendString("No Game Running");
  }

#ifdef WITH_CHEEVOS
  SmallString state_string;
  // Trim to 128 bytes as per Discord-RPC requirements
  if (m_discord_presence_cheevos_string.length() >= 128)
  {
    // 124 characters + 3 dots + null terminator
    state_string = m_discord_presence_cheevos_string.substr(0, 124);
    state_string.AppendString("...");
  }
  else
  {
    state_string = m_discord_presence_cheevos_string;
  }

  rp.state = state_string;
#endif
  rp.details = details_string;

  Discord_UpdatePresence(&rp);
}

void CommonHost::PollDiscordPresence()
{
  if (!m_discord_presence_active)
    return;

  UpdateDiscordPresence(true);

  Discord_RunCallbacks();
}

#endif

static void HotkeyModifyResolutionScale(s32 increment)
{
  const u32 new_resolution_scale = std::clamp<u32>(
    static_cast<u32>(static_cast<s32>(g_settings.gpu_resolution_scale) + increment), 1, GPU::MAX_RESOLUTION_SCALE);
  if (new_resolution_scale == g_settings.gpu_resolution_scale)
    return;

  g_settings.gpu_resolution_scale = new_resolution_scale;

  if (System::IsValid())
  {
    g_gpu->RestoreGraphicsAPIState();
    g_gpu->UpdateSettings();
    g_gpu->ResetGraphicsAPIState();
    System::ClearMemorySaveStates();
    Host::InvalidateDisplay();
  }
}

static void HotkeyLoadStateSlot(bool global, s32 slot)
{
  if (!System::IsValid())
    return;

  if (!global && System::GetRunningSerial().empty())
  {
    Host::AddKeyedOSDMessage("LoadState", TRANSLATABLE("OSDMessage", "Cannot load state for game without serial."),
                             5.0f);
    return;
  }

  std::string path(global ? System::GetGlobalSaveStateFileName(slot) :
                            System::GetGameSaveStateFileName(System::GetRunningSerial(), slot));
  if (!FileSystem::FileExists(path.c_str()))
  {
    Host::AddKeyedOSDMessage("LoadState",
                             fmt::format(TRANSLATABLE("OSDMessage", "No save state found in slot {}."), slot), 5.0f);
    return;
  }

  System::LoadState(path.c_str());
}

static void HotkeySaveStateSlot(bool global, s32 slot)
{
  if (!System::IsValid())
    return;

  if (!global && System::GetRunningSerial().empty())
  {
    Host::AddKeyedOSDMessage("LoadState", TRANSLATABLE("OSDMessage", "Cannot save state for game without serial."),
                             5.0f);
    return;
  }

  std::string path(global ? System::GetGlobalSaveStateFileName(slot) :
                            System::GetGameSaveStateFileName(System::GetRunningSerial(), slot));
  System::SaveState(path.c_str(), g_settings.create_save_state_backups);
}

BEGIN_HOTKEY_LIST(g_common_hotkeys)
#ifndef __ANDROID__
DEFINE_HOTKEY("OpenPauseMenu", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Open Pause Menu"),
              [](s32 pressed) {
                if (!pressed)
                  FullscreenUI::OpenPauseMenu();
              })
#endif

DEFINE_HOTKEY("FastForward", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Fast Forward"),
              [](s32 pressed) {
                if (pressed < 0)
                  return;
                System::SetFastForwardEnabled(pressed > 0);
              })

DEFINE_HOTKEY("ToggleFastForward", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Toggle Fast Forward"),
              [](s32 pressed) {
                if (!pressed)
                  System::SetFastForwardEnabled(!System::IsFastForwardEnabled());
              })

DEFINE_HOTKEY("Turbo", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Turbo"), [](s32 pressed) {
  if (pressed < 0)
    return;
  System::SetTurboEnabled(pressed > 0);
})

DEFINE_HOTKEY("ToggleTurbo", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Toggle Turbo"),
              [](s32 pressed) {
                if (!pressed)
                  System::SetTurboEnabled(!System::IsTurboEnabled());
              })

#ifndef __ANDROID__
DEFINE_HOTKEY("ToggleFullscreen", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Toggle Fullscreen"),
              [](s32 pressed) {
                if (!pressed)
                  Host::SetFullscreen(!Host::IsFullscreen());
              })

DEFINE_HOTKEY("TogglePause", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Toggle Pause"),
              [](s32 pressed) {
                if (!pressed)
                  System::PauseSystem(!System::IsPaused());
              })

DEFINE_HOTKEY("PowerOff", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Power Off System"),
              [](s32 pressed) {
                if (!pressed)
                  Host::RequestSystemShutdown(true, g_settings.save_state_on_exit);
              })
#endif

DEFINE_HOTKEY("Screenshot", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Save Screenshot"),
              [](s32 pressed) {
                if (!pressed)
                  System::SaveScreenshot();
              })

#if !defined(__ANDROID__) && defined(WITH_CHEEVOS)
DEFINE_HOTKEY("OpenAchievements", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Open Achievement List"),
              [](s32 pressed) {
                if (!pressed)
                {
                  if (!FullscreenUI::OpenAchievementsWindow())
                  {
                    Host::AddOSDMessage(
                      Host::TranslateStdString("OSDMessage", "Achievements are disabled or unavailable for  game."),
                      10.0f);
                  }
                }
              })

DEFINE_HOTKEY("OpenLeaderboards", TRANSLATABLE("Hotkeys", "General"), TRANSLATABLE("Hotkeys", "Open Leaderboard List"),
              [](s32 pressed) {
                if (!pressed)
                {
                  if (!FullscreenUI::OpenLeaderboardsWindow())
                  {
                    Host::AddOSDMessage(
                      Host::TranslateStdString("OSDMessage", "Leaderboards are disabled or unavailable for  game."),
                      10.0f);
                  }
                }
              })
#endif // !defined(__ANDROID__) && defined(WITH_CHEEVOS)

DEFINE_HOTKEY("Reset", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Reset System"), [](s32 pressed) {
  if (!pressed)
    Host::RunOnCPUThread(System::ResetSystem);
})

DEFINE_HOTKEY("ChangeDisc", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Change Disc"), [](s32 pressed) {
  if (!pressed && System::IsValid() && System::HasMediaSubImages())
  {
    const u32 current = System::GetMediaSubImageIndex();
    const u32 next = (current + 1) % System::GetMediaSubImageCount();
    if (current != next)
      Host::RunOnCPUThread([next]() { System::SwitchMediaSubImage(next); });
  }
})

DEFINE_HOTKEY("SwapMemoryCards", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Swap Memory Card Slots"),
              [](s32 pressed) {
                if (!pressed)
                  System::SwapMemoryCards();
              })

#ifndef __ANDROID__
DEFINE_HOTKEY("FrameStep", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Frame Step"), [](s32 pressed) {
  if (!pressed)
    System::DoFrameStep();
})
#endif

DEFINE_HOTKEY("Rewind", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Rewind"), [](s32 pressed) {
  if (pressed < 0)
    return;
  System::SetRewindState(pressed > 0);
})

#ifndef __ANDROID__
DEFINE_HOTKEY("ToggleCheats", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Toggle Cheats"),
              [](s32 pressed) {
                if (!pressed)
                  System::DoToggleCheats();
              })
#else
DEFINE_HOTKEY("TogglePatchCodes", TRANSLATABLE("Hotkeys", "System"), TRANSLATABLE("Hotkeys", "Toggle Patch Codes"),
              [](s32 pressed) {
                if (!pressed)
                  System::DoToggleCheats();
              })
#endif

DEFINE_HOTKEY("ToggleOverclocking", TRANSLATABLE("Hotkeys", "System"),
              TRANSLATABLE("Hotkeys", "Toggle Clock Speed Control (Overclocking)"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.cpu_overclock_enable = !g_settings.cpu_overclock_enable;
                  g_settings.UpdateOverclockActive();
                  System::UpdateOverclock();

                  if (g_settings.cpu_overclock_enable)
                  {
                    const u32 percent = g_settings.GetCPUOverclockPercent();
                    const double clock_speed =
                      ((static_cast<double>(System::MASTER_CLOCK) * static_cast<double>(percent)) / 100.0) / 1000000.0;
                    Host::AddKeyedFormattedOSDMessage(
                      "ToggleOverclocking", 5.0f,
                      Host::TranslateString("OSDMessage", "CPU clock speed control enabled (%u%% / %.3f MHz)."),
                      percent, clock_speed);
                  }
                  else
                  {
                    Host::AddKeyedFormattedOSDMessage(
                      "ToggleOverclocking", 5.0f,
                      Host::TranslateString("OSDMessage", "CPU clock speed control disabled (%.3f MHz)."),
                      static_cast<double>(System::MASTER_CLOCK) / 1000000.0);
                  }
                }
              })

DEFINE_HOTKEY("IncreaseEmulationSpeed", TRANSLATABLE("Hotkeys", "System"),
              TRANSLATABLE("Hotkeys", "Increase Emulation Speed"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.emulation_speed += 0.1f;
                  System::UpdateSpeedLimiterState();
                  Host::AddKeyedFormattedOSDMessage("EmulationSpeedChange", 5.0f,
                                                    Host::TranslateString("OSDMessage", "Emulation speed set to %u%%."),
                                                    static_cast<u32>(std::lround(g_settings.emulation_speed * 100.0f)));
                }
              })

DEFINE_HOTKEY("DecreaseEmulationSpeed", TRANSLATABLE("Hotkeys", "System"),
              TRANSLATABLE("Hotkeys", "Decrease Emulation Speed"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.emulation_speed = std::max(g_settings.emulation_speed - 0.1f, 0.1f);
                  System::UpdateSpeedLimiterState();
                  Host::AddKeyedFormattedOSDMessage("EmulationSpeedChange", 5.0f,
                                                    Host::TranslateString("OSDMessage", "Emulation speed set to %u%%."),
                                                    static_cast<u32>(std::lround(g_settings.emulation_speed * 100.0f)));
                }
              })

DEFINE_HOTKEY("ResetEmulationSpeed", TRANSLATABLE("Hotkeys", "System"),
              TRANSLATABLE("Hotkeys", "Reset Emulation Speed"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.emulation_speed = Host::GetFloatSettingValue("Main", "EmulationSpeed", 1.0f);
                  System::UpdateSpeedLimiterState();
                  Host::AddKeyedFormattedOSDMessage("EmulationSpeedChange", 5.0f,
                                                    Host::TranslateString("OSDMessage", "Emulation speed set to %u%%."),
                                                    static_cast<u32>(std::lround(g_settings.emulation_speed * 100.0f)));
                }
              })

DEFINE_HOTKEY("ToggleSoftwareRendering", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Toggle Software Rendering"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                  System::ToggleSoftwareRendering();
              })

DEFINE_HOTKEY("TogglePGXP", TRANSLATABLE("Hotkeys", "Graphics"), TRANSLATABLE("Hotkeys", "Toggle PGXP"),
              [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.gpu_pgxp_enable = !g_settings.gpu_pgxp_enable;
                  g_gpu->RestoreGraphicsAPIState();
                  g_gpu->UpdateSettings();
                  g_gpu->ResetGraphicsAPIState();
                  System::ClearMemorySaveStates();
                  Host::AddKeyedOSDMessage("TogglePGXP",
                                           g_settings.gpu_pgxp_enable ?
                                             Host::TranslateStdString("OSDMessage", "PGXP is now enabled.") :
                                             Host::TranslateStdString("OSDMessage", "PGXP is now disabled."),
                                           5.0f);

                  if (g_settings.gpu_pgxp_enable)
                    PGXP::Initialize();
                  else
                    PGXP::Shutdown();

                  // we need to recompile all blocks if pgxp is toggled on/off
                  if (g_settings.IsUsingCodeCache())
                    CPU::CodeCache::Flush();
                }
              })

DEFINE_HOTKEY("IncreaseResolutionScale", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Increase Resolution Scale"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                  HotkeyModifyResolutionScale(1);
              })

DEFINE_HOTKEY("DecreaseResolutionScale", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Decrease Resolution Scale"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                  HotkeyModifyResolutionScale(-1);
              })

DEFINE_HOTKEY("TogglePostProcessing", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Toggle Post-Processing"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                  System::TogglePostProcessing();
              })

DEFINE_HOTKEY("ReloadPostProcessingShaders", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Reload Post Processing Shaders"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                  System::ReloadPostProcessingShaders();
              })

DEFINE_HOTKEY("ReloadTextureReplacements", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Reload Texture Replacements"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  Host::AddKeyedOSDMessage("ReloadTextureReplacements",
                                           Host::TranslateStdString("OSDMessage", "Texture replacements reloaded."),
                                           10.0f);
                  g_texture_replacements.Reload();
                }
              })

DEFINE_HOTKEY("ToggleWidescreen", TRANSLATABLE("Hotkeys", "Graphics"), TRANSLATABLE("Hotkeys", "Toggle Widescreen"),
              [](s32 pressed) {
                if (!pressed)
                  System::ToggleWidescreen();
              })

DEFINE_HOTKEY("TogglePGXPDepth", TRANSLATABLE("Hotkeys", "Graphics"),
              TRANSLATABLE("Hotkeys", "Toggle PGXP Depth Buffer"), [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.gpu_pgxp_depth_buffer = !g_settings.gpu_pgxp_depth_buffer;
                  if (!g_settings.gpu_pgxp_enable)
                    return;

                  g_gpu->RestoreGraphicsAPIState();
                  g_gpu->UpdateSettings();
                  g_gpu->ResetGraphicsAPIState();
                  System::ClearMemorySaveStates();
                  Host::AddKeyedOSDMessage(
                    "TogglePGXPDepth",
                    g_settings.gpu_pgxp_depth_buffer ?
                      Host::TranslateStdString("OSDMessage", "PGXP Depth Buffer is now enabled.") :
                      Host::TranslateStdString("OSDMessage", "PGXP Depth Buffer is now disabled."),
                    5.0f);
                }
              })

DEFINE_HOTKEY("TogglePGXPCPU", TRANSLATABLE("Hotkeys", "Graphics"), TRANSLATABLE("Hotkeys", "Toggle PGXP CPU Mode"),
              [](s32 pressed) {
                if (pressed && System::IsValid())
                {
                  g_settings.gpu_pgxp_cpu = !g_settings.gpu_pgxp_cpu;
                  if (!g_settings.gpu_pgxp_enable)
                    return;

                  g_gpu->RestoreGraphicsAPIState();
                  g_gpu->UpdateSettings();
                  g_gpu->ResetGraphicsAPIState();
                  System::ClearMemorySaveStates();
                  Host::AddKeyedOSDMessage("TogglePGXPCPU",
                                           g_settings.gpu_pgxp_cpu ?
                                             Host::TranslateStdString("OSDMessage", "PGXP CPU mode is now enabled.") :
                                             Host::TranslateStdString("OSDMessage", "PGXP CPU mode is now disabled."),
                                           5.0f);

                  PGXP::Shutdown();
                  PGXP::Initialize();

                  // we need to recompile all blocks if pgxp is toggled on/off
                  if (g_settings.IsUsingCodeCache())
                    CPU::CodeCache::Flush();
                }
              })

DEFINE_HOTKEY("AudioMute", TRANSLATABLE("Hotkeys", "Audio"), TRANSLATABLE("Hotkeys", "Toggle Mute"), [](s32 pressed) {
  if (!pressed && System::IsValid())
  {
    g_settings.audio_output_muted = !g_settings.audio_output_muted;
    const s32 volume = System::GetAudioOutputVolume();
    SPU::GetOutputStream()->SetOutputVolume(volume);
    if (g_settings.audio_output_muted)
    {
      Host::AddIconOSDMessage("AudioControlHotkey", ICON_FA_VOLUME_MUTE,
                              Host::TranslateStdString("OSDMessage", "Volume: Muted"), 5.0f);
    }
    else
    {
      Host::AddIconOSDMessage("AudioControlHotkey", ICON_FA_VOLUME_UP,
                              fmt::format(Host::TranslateString("OSDMessage", "Volume: {}%").GetCharArray(), volume),
                              5.0f);
    }
  }
})
DEFINE_HOTKEY("AudioCDAudioMute", TRANSLATABLE("Hotkeys", "Audio"), TRANSLATABLE("Hotkeys", "Toggle CD Audio Mute"),
              [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.cdrom_mute_cd_audio = !g_settings.cdrom_mute_cd_audio;
                  Host::AddIconOSDMessage(
                    "AudioControlHotkey", g_settings.cdrom_mute_cd_audio ? ICON_FA_VOLUME_MUTE : ICON_FA_VOLUME_UP,
                    g_settings.cdrom_mute_cd_audio ? Host::TranslateStdString("OSDMessage", "CD Audio Muted.") :
                                                     Host::TranslateStdString("OSDMessage", "CD Audio Unmuted."),
                    2.0f);
                }
              })
DEFINE_HOTKEY("AudioVolumeUp", TRANSLATABLE("Hotkeys", "Audio"), TRANSLATABLE("Hotkeys", "Volume Up"), [](s32 pressed) {
  if (!pressed && System::IsValid())
  {
    g_settings.audio_output_muted = false;

    const s32 volume = std::min<s32>(System::GetAudioOutputVolume() + 10, 100);
    g_settings.audio_output_volume = volume;
    g_settings.audio_fast_forward_volume = volume;
    SPU::GetOutputStream()->SetOutputVolume(volume);
    Host::AddIconOSDMessage("AudioControlHotkey", ICON_FA_VOLUME_UP,
                            fmt::format(Host::TranslateString("OSDMessage", "Volume: {}%").GetCharArray(), volume),
                            5.0f);
  }
})
DEFINE_HOTKEY("AudioVolumeDown", TRANSLATABLE("Hotkeys", "Audio"), TRANSLATABLE("Hotkeys", "Volume Down"),
              [](s32 pressed) {
                if (!pressed && System::IsValid())
                {
                  g_settings.audio_output_muted = false;

                  const s32 volume = std::max<s32>(System::GetAudioOutputVolume() - 10, 0);
                  g_settings.audio_output_volume = volume;
                  g_settings.audio_fast_forward_volume = volume;
                  SPU::GetOutputStream()->SetOutputVolume(volume);
                  Host::AddIconOSDMessage(
                    "AudioControlHotkey", ICON_FA_VOLUME_DOWN,
                    fmt::format(Host::TranslateString("OSDMessage", "Volume: {}%").GetCharArray(), volume), 5.0f);
                }
              })

// NOTE: All save/load state hotkeys are deferred, because it can trigger setting reapply, which reloads bindings.
DEFINE_HOTKEY("LoadSelectedSaveState", TRANSLATABLE("Hotkeys", "Save States"),
              TRANSLATABLE("Hotkeys", "Load From Selected Slot"), [](s32 pressed) {
                if (!pressed)
                  Host::RunOnCPUThread(SaveStateSelectorUI::LoadCurrentSlot);
              })
DEFINE_HOTKEY("SaveSelectedSaveState", TRANSLATABLE("Hotkeys", "Save States"),
              TRANSLATABLE("Hotkeys", "Save To Selected Slot"), [](s32 pressed) {
                if (!pressed)
                  Host::RunOnCPUThread(SaveStateSelectorUI::SaveCurrentSlot);
              })
DEFINE_HOTKEY("SelectPreviousSaveStateSlot", TRANSLATABLE("Hotkeys", "Save States"),
              TRANSLATABLE("Hotkeys", "Select Previous Save Slot"), [](s32 pressed) {
                if (!pressed)
                  Host::RunOnCPUThread(SaveStateSelectorUI::SelectPreviousSlot);
              })
DEFINE_HOTKEY("SelectNextSaveStateSlot", TRANSLATABLE("Hotkeys", "Save States"),
              TRANSLATABLE("Hotkeys", "Select Next Save Slot"), [](s32 pressed) {
                if (!pressed)
                  Host::RunOnCPUThread(SaveStateSelectorUI::SelectNextSlot);
              })

DEFINE_HOTKEY("UndoLoadState", TRANSLATABLE("Hotkeys", "Save States"), TRANSLATABLE("Hotkeys", "Undo Load State"),
              [](s32 pressed) {
                if (!pressed)
                  Host::RunOnCPUThread(System::UndoLoadState);
              })

#define MAKE_LOAD_STATE_HOTKEY(global, slot, name)                                                                     \
  DEFINE_HOTKEY(global ? "LoadGameState" #slot : "LoadGlobalState" #slot, TRANSLATABLE("Hotkeys", "Save States"),      \
                name, [](s32 pressed) {                                                                                \
                  if (!pressed)                                                                                        \
                    Host::RunOnCPUThread([]() { HotkeyLoadStateSlot(global, slot); });                                 \
                })
#define MAKE_SAVE_STATE_HOTKEY(global, slot, name)                                                                     \
  DEFINE_HOTKEY(global ? "SaveGameState" #slot : "SaveGlobalState" #slot, TRANSLATABLE("Hotkeys", "Save States"),      \
                name, [](s32 pressed) {                                                                                \
                  if (!pressed)                                                                                        \
                    Host::RunOnCPUThread([]() { HotkeySaveStateSlot(global, slot); });                                 \
                })

MAKE_LOAD_STATE_HOTKEY(false, 1, TRANSLATABLE("Hotkeys", "Load Game State 1"))
MAKE_SAVE_STATE_HOTKEY(false, 1, TRANSLATABLE("Hotkeys", "Save Game State 1"))
MAKE_LOAD_STATE_HOTKEY(false, 2, TRANSLATABLE("Hotkeys", "Load Game State 2"))
MAKE_SAVE_STATE_HOTKEY(false, 2, TRANSLATABLE("Hotkeys", "Save Game State 2"))
MAKE_LOAD_STATE_HOTKEY(false, 3, TRANSLATABLE("Hotkeys", "Load Game State 3"))
MAKE_SAVE_STATE_HOTKEY(false, 3, TRANSLATABLE("Hotkeys", "Save Game State 3"))
MAKE_LOAD_STATE_HOTKEY(false, 4, TRANSLATABLE("Hotkeys", "Load Game State 4"))
MAKE_SAVE_STATE_HOTKEY(false, 4, TRANSLATABLE("Hotkeys", "Save Game State 4"))
MAKE_LOAD_STATE_HOTKEY(false, 5, TRANSLATABLE("Hotkeys", "Load Game State 5"))
MAKE_SAVE_STATE_HOTKEY(false, 5, TRANSLATABLE("Hotkeys", "Save Game State 5"))
MAKE_LOAD_STATE_HOTKEY(false, 6, TRANSLATABLE("Hotkeys", "Load Game State 6"))
MAKE_SAVE_STATE_HOTKEY(false, 6, TRANSLATABLE("Hotkeys", "Save Game State 6"))
MAKE_LOAD_STATE_HOTKEY(false, 7, TRANSLATABLE("Hotkeys", "Load Game State 7"))
MAKE_SAVE_STATE_HOTKEY(false, 7, TRANSLATABLE("Hotkeys", "Save Game State 7"))
MAKE_LOAD_STATE_HOTKEY(false, 8, TRANSLATABLE("Hotkeys", "Load Game State 8"))
MAKE_SAVE_STATE_HOTKEY(false, 8, TRANSLATABLE("Hotkeys", "Save Game State 8"))
MAKE_LOAD_STATE_HOTKEY(false, 9, TRANSLATABLE("Hotkeys", "Load Game State 9"))
MAKE_SAVE_STATE_HOTKEY(false, 9, TRANSLATABLE("Hotkeys", "Save Game State 9"))
MAKE_LOAD_STATE_HOTKEY(false, 10, TRANSLATABLE("Hotkeys", "Load Game State 10"))
MAKE_SAVE_STATE_HOTKEY(false, 10, TRANSLATABLE("Hotkeys", "Save Game State 10"))

MAKE_LOAD_STATE_HOTKEY(true, 1, TRANSLATABLE("Hotkeys", "Load Global State 1"))
MAKE_SAVE_STATE_HOTKEY(true, 1, TRANSLATABLE("Hotkeys", "Save Global State 1"))
MAKE_LOAD_STATE_HOTKEY(true, 2, TRANSLATABLE("Hotkeys", "Load Global State 2"))
MAKE_SAVE_STATE_HOTKEY(true, 2, TRANSLATABLE("Hotkeys", "Save Global State 2"))
MAKE_LOAD_STATE_HOTKEY(true, 3, TRANSLATABLE("Hotkeys", "Load Global State 3"))
MAKE_SAVE_STATE_HOTKEY(true, 3, TRANSLATABLE("Hotkeys", "Save Global State 3"))
MAKE_LOAD_STATE_HOTKEY(true, 4, TRANSLATABLE("Hotkeys", "Load Global State 4"))
MAKE_SAVE_STATE_HOTKEY(true, 4, TRANSLATABLE("Hotkeys", "Save Global State 4"))
MAKE_LOAD_STATE_HOTKEY(true, 5, TRANSLATABLE("Hotkeys", "Load Global State 5"))
MAKE_SAVE_STATE_HOTKEY(true, 5, TRANSLATABLE("Hotkeys", "Save Global State 5"))
MAKE_LOAD_STATE_HOTKEY(true, 6, TRANSLATABLE("Hotkeys", "Load Global State 6"))
MAKE_SAVE_STATE_HOTKEY(true, 6, TRANSLATABLE("Hotkeys", "Save Global State 6"))
MAKE_LOAD_STATE_HOTKEY(true, 7, TRANSLATABLE("Hotkeys", "Load Global State 7"))
MAKE_SAVE_STATE_HOTKEY(true, 7, TRANSLATABLE("Hotkeys", "Save Global State 7"))
MAKE_LOAD_STATE_HOTKEY(true, 8, TRANSLATABLE("Hotkeys", "Load Global State 8"))
MAKE_SAVE_STATE_HOTKEY(true, 8, TRANSLATABLE("Hotkeys", "Save Global State 8"))
MAKE_LOAD_STATE_HOTKEY(true, 9, TRANSLATABLE("Hotkeys", "Load Global State 9"))
MAKE_SAVE_STATE_HOTKEY(true, 9, TRANSLATABLE("Hotkeys", "Save Global State 9"))
MAKE_LOAD_STATE_HOTKEY(true, 10, TRANSLATABLE("Hotkeys", "Load Global State 10"))
MAKE_SAVE_STATE_HOTKEY(true, 10, TRANSLATABLE("Hotkeys", "Save Global State 10"))

#undef MAKE_SAVE_STATE_HOTKEY
#undef MAKE_LOAD_STATE_HOTKEY

END_HOTKEY_LIST()
