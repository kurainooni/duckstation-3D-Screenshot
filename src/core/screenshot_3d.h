#pragma once
#include "types.h"
#include "gpu_types.h"

namespace Screenshot3D {

void NextFrame();
void Shutdown();

// Use to make GTE.NCLIP always be positive
bool ShouldDisableCulling();

// Called in GTE.RTPS to track 3D verts
void PushVertex(s32& Sx, s32& Sy, float x, float y, float z);

// Called when a polygon is drawn
bool WantsPolygon();
void DrawPolygon(
  GPURenderCommand rc,
  const GPUBackendDrawPolygonCommand::Vertex verts[4],
  GPUDrawModeReg mode_reg,
  u16 palette_reg,
  GPUTextureWindow texture_window
);

// Called once just before NextFrame to fill texture data from the
// contents of VRAM.
//
// Textures always use the content of VRAM at frame end. Tracking of
// mid-frame VRAM changes via dirty rects, etc. is not implemented.
bool WantsUpdateFromVRAM();
void UpdateFromVRAM(const u16* vram_ptr);

void DrawGuiWindow();

} // namespace Screenshot3D
