#pragma once
#include "types.h"

namespace Freecam {

void NextFrame();

// Called in GTE.RTPS after rotating and translating, but before
// projecting.
void ApplyToVertex(s64& x, s64& y, s64& z);

void DrawGuiWindow();

} // namespace Freecam
