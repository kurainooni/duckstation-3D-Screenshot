// SPDX-FileCopyrightText: 2019-2022 Connor McLaughlin <stenzek@gmail.com>
// SPDX-License-Identifier: (GPL-3.0 OR CC-BY-NC-ND-4.0)

#include "common/window_info.h"

namespace FrontendCommon {
void SuspendScreensaver(const WindowInfo& wi);
void ResumeScreensaver();

/// Abstracts platform-specific code for asynchronously playing a sound.
/// On Windows, this will use PlaySound(). On Linux, it will shell out to aplay. On MacOS, it uses NSSound.
bool PlaySoundAsync(const char* path);
} // namespace FrontendCommon
