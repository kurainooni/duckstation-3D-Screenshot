#include "screenshot_3d.h"
#include "host.h"
#include "system.h"

#include "common/file_system.h"
#include "common/hash_combine.h"
#include "common/image.h"
#include "common/path.h"

#include "util/imgui_manager.h"

#include "fmt/format.h"
#include "fmt/chrono.h"
#include "imgui.h"
#include "xxhash.h"
#if defined(CPU_ARCH_X86) || defined(CPU_ARCH_X64)
#include "xxh_x86dispatch.h"
#endif

#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace Screenshot3D {
namespace {

struct Config
{
  int num_frames_to_capture = 1;
  bool disable_culling = true;
  bool dump_full_textures = true;
  bool in_2d_mode = false;
  bool is_dry_run = false;
};

struct Vertex
{
  s16 Sx, Sy;      // screen coordinate, output of RTPS
  float x, y, z;   // XYZ before projection, high precision
  u32 generation;  // generation number, used to recycle old verts
};

struct Poly
{
  GPUBackendDrawPolygonCommand::Vertex v[4];

  u8 is_quad : 1;
  u8 texture_enable : 1;
  u8 transparency_enable : 1;

  GPUTransparencyMode transparency_mode;

  u32 texture_index;

  int NumVerts() const { return is_quad ? 4 : 3; }
};

struct TextureState
{
  u16 texture_page_x_base : 4;
  u16 texture_page_y_base : 1;
  u16 transparency_enable : 1;
  u16 texture_mode: 2;

  u16 palette;

  GPUTextureWindow window;

  ALWAYS_INLINE u16 GetTexturePageBaseX() const { return ZeroExtend16(texture_page_x_base) * 64; }
  ALWAYS_INLINE u16 GetTexturePageBaseY() const { return ZeroExtend16(texture_page_y_base) * 256; }
  GPUTextureMode GetTextureMode() const { return GPUTextureMode(texture_mode); }
  GPUTexturePaletteReg GetPaletteReg() const
  {
    GPUTexturePaletteReg reg;
    reg.bits = palette;
    return reg;
  }

  // For use as unordered_map key
  u64 PackIntoU64() const
  {
    return
      (u64(texture_page_x_base) << 0) |
      (u64(texture_page_y_base) << 4) |
      (u64(transparency_enable) << 5) |
      (u64(texture_mode) << 6) |
      (u64(palette) << 16) |
      (u64(window.and_x) << 32) |
      (u64(window.and_y) << 40) |
      (u64(window.or_x) << 48) |
      (u64(window.or_y) << 56);
  }
};

// UV bounding box of polys drawn with the same TextureState
struct UVBlob
{
  u8 min_u, max_u;
  u8 min_v, max_v;

  u32 Width() const { return u32(max_u) - u32(min_u) + 1; }
  u32 Height() const { return u32(max_v) - u32(min_v) + 1; }

  void Init(const Poly& poly)
  {
    min_u = max_u = poly.v[0].u;
    min_v = max_v = poly.v[0].v;
    UpdateBoundingBox(poly);
  }

  void UpdateBoundingBox(const Poly& poly)
  {
    for (int i = 0; i < poly.NumVerts(); i++)
    {
      min_u = std::min(min_u, poly.v[i].u);
      max_u = std::max(max_u, poly.v[i].u);
      min_v = std::min(min_v, poly.v[i].v);
      max_v = std::max(max_v, poly.v[i].v);
    }
  }
};

struct Texture
{
  TextureState tstate;
  UVBlob blob;
  Common::RGBA8Image image;
  std::string filename;
};

} // namespace

// Games are generally pipelined so the polys processed on frame N
// don't get drawn until frame N+k.
//
// This has two consequences.
//
// First, if we start running on frame N, the first frame we can rip
// is N+k. We have to wait until all the unknown data is out of the
// pipe or we'll get missing verts, etc. So there is a "warmup" period
// during which we're running, but we're not shooting yet.
//
// Second, we need to keep vertices in the cache for k frames,
// otherwise they'll be evicted before they are drawn. Verts older
// than k frames are recycled to free up their screen coordinate.
//
// Higher values of k are safer, but increase screenshot latency
// (warmup period), and increase vertex cache occupancy, which makes
// vertex jitter really slow and increases the change of "2D litter".
//
// Most games work with k=1. FF7's battle screen needs k=2.
static constexpr u32 WARMUP_PERIOD = 2;
static constexpr u32 VERTEX_RECYCLE_PERIOD = 2;

static Config s_config;
static bool s_running = false;
static int s_shots_taken = 0;
static u32 s_frame_counter = 0;

using ScreenXYKey = u32;
using TextureKey = u64;
using TextureIndex = u32;

static std::vector<Poly> s_poly_buffer;
static std::unordered_map<ScreenXYKey, Vertex> s_vertex_cache;
static std::vector<Texture> s_textures;
static std::unordered_map<TextureKey, TextureIndex> s_texture_cache;

// UI doesn't directly control the state. It schedules updates for the
// next frame end (double buffered) with these variables.
static Config s_ui_config;
static bool s_run_requested = false;

////////////////////////////////////
// Main lifecycle events
////////////////////////////////////
static void TurnOff()
{
  s_running = false;
  s_shots_taken = 0;
  s_frame_counter = 0;
  s_vertex_cache.clear();
  s_poly_buffer.clear();
  s_textures.clear();
  s_texture_cache.clear();
}

static void TurnOn()
{
  TurnOff();
  s_running = true;
}

void Shutdown()
{
  TurnOff();
}

static bool IsWarmedUp()
{
  return s_running && s_frame_counter >= WARMUP_PERIOD;
}

static bool IsDoneShooting()
{
  return s_shots_taken >= s_config.num_frames_to_capture;
}

////////////////////////////////////
// Misc GTE hooks
////////////////////////////////////
bool ShouldDisableCulling()
{
  return s_running && s_config.disable_culling;
}

////////////////////////////////////
// Vertex/RTPS tracking
////////////////////////////////////
static u32 PackSXYIntoU32(s32 Sx, s32 Sy)
{
  return u32(u16(Sx)) | (u32(u16(Sy)) << 16);
}

// Jitter the output of RTPS around so no two verts land on the same
// pixel. Thus a screen pixel uniquely indentifies the vert that
// produced it.
Vertex& FindFreeScreenCoord(s32& Sx, s32& Sy)
{
  const u32 key = PackSXYIntoU32(Sx, Sy);
  auto [it, success] = s_vertex_cache.try_emplace(key);
  if (success)
    return it->second;

  // Seed RNG, different per vertex
  std::size_t seed = 0xb0ba7ea;
  hash_combine(seed, s_frame_counter, s_vertex_cache.size());
  u32 rng = seed;

  for (u32 attempts = 0; true; ++attempts)
  {
    rng *= 0x343fdU;
    rng += 0x269ec3U;

    // Move up/down/left/right at random
    const bool axis = (rng >> 20) & 1;
    const bool sign = (rng >> 21) & 1;
    // Move faster if we've failed repeatedly
    const s32 dist = 1 + ((rng >> 22) & (attempts < 5 ? 3 : 7));
    (axis ? Sx : Sy) += sign ? dist : -dist;

    // Saturate
    Sx = Sx < -1024 ? -1024 : Sx > 1023 ? 1023 : Sx;
    Sy = Sy < -1024 ? -1024 : Sy > 1023 ? 1023 : Sy;

    const u32 key = PackSXYIntoU32(Sx, Sy);
    auto [it, success] = s_vertex_cache.try_emplace(key);
    if (success)
      return it->second;
  }
}

void PushVertex(s32& Sx, s32& Sy, float x, float y, float z)
{
  if (!s_running || s_config.in_2d_mode)
    return;

  // Saturate
  Sx = Sx < -1024 ? -1024 : Sx > 1023 ? 1023 : Sx;
  Sy = Sy < -1024 ? -1024 : Sy > 1023 ? 1023 : Sy;

  Vertex& v = FindFreeScreenCoord(Sx, Sy);
  v.Sx = Sx;
  v.Sy = Sy;
  v.x = x;
  v.y = y;
  v.z = z;
  v.generation = s_frame_counter;
}

////////////////////////////////////
// Polygon tracking
////////////////////////////////////
bool WantsPolygon()
{
  return s_running;
}

static TextureIndex AssignPolyToTexture(TextureState tstate, const Poly& poly)
{
  const u64 key = tstate.PackIntoU64();
  const u32 next_texture_index = s_textures.size();
  const auto [it, inserted] = s_texture_cache.insert({key, next_texture_index});
  if (!inserted)
  {
    // Update existing texture
    const u32 texture_index = it->second;
    s_textures[texture_index].blob.UpdateBoundingBox(poly);
    return texture_index;
  }

  Texture texture;
  texture.tstate = tstate;
  texture.blob.Init(poly);
  s_textures.push_back(texture);

  return next_texture_index;
}

void DrawPolygon(
  GPURenderCommand rc,
  const GPUBackendDrawPolygonCommand::Vertex verts[4],
  GPUDrawModeReg mode_reg,
  u16 palette_reg,
  GPUTextureWindow texture_window
) {
  if (!s_running)
    return;

  Poly poly;

  poly.is_quad = rc.quad_polygon;
  poly.texture_enable = rc.texture_enable;
  poly.transparency_enable = rc.transparency_enable;
  poly.transparency_mode = mode_reg.transparency_mode;

  for (int i = 0; i < poly.NumVerts(); i++)
  {
    poly.v[i] = verts[i];

    // Raw texturing is the same as shading with 128
    if (rc.texture_enable && rc.raw_texture_enable)
    {
      poly.v[i].r = 128;
      poly.v[i].g = 128;
      poly.v[i].b = 128;
    }
  }

  if (poly.texture_enable)
  {
    TextureState tstate;

    tstate.texture_page_x_base = mode_reg.texture_page_x_base;
    tstate.texture_page_y_base = mode_reg.texture_page_y_base;
    tstate.texture_mode = (u16)mode_reg.texture_mode.GetValue();
    tstate.transparency_enable = poly.transparency_enable;
    tstate.palette = palette_reg;
    tstate.window = texture_window;

    poly.texture_index = AssignPolyToTexture(tstate, poly);
  }

  s_poly_buffer.push_back(poly);
}

////////////////////////////////////
// Textures
////////////////////////////////////
bool WantsUpdateFromVRAM()
{
  return s_running && !s_config.is_dry_run && IsWarmedUp();
}

static void FillTextureFromVRAM(Texture& texture, const u16* vram_ptr)
{
#define GetPixel(x, y) (vram_ptr[VRAM_WIDTH * (y) + (x)])
  const auto& draw_mode = texture.tstate;
  const auto palette = texture.tstate.GetPaletteReg();
  const auto texture_mode = texture.tstate.GetTextureMode();
  const auto& window = texture.tstate.window;
  auto& image = texture.image;

  image.SetSize(texture.blob.Width(), texture.blob.Height());

  for (u32 image_y = 0; image_y < image.GetHeight(); ++image_y)
  {
    for (u32 image_x = 0; image_x < image.GetWidth(); ++image_x)
    {
      u8 texcoord_x = texture.blob.min_u + image_x;
      u8 texcoord_y = texture.blob.min_v + image_y;

      // Apply texture window
      texcoord_x = (texcoord_x & window.and_x) | window.or_x;
      texcoord_y = (texcoord_y & window.and_y) | window.or_y;

      u16 texture_color;
      if (texture_mode == GPUTextureMode::Palette4Bit)
      {
        const u16 palette_value = GetPixel(
          (draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 4)) % VRAM_WIDTH,
          (draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT
        );
        const u16 palette_index = (palette_value >> ((texcoord_x % 4) * 4)) & 0x0Fu;

        texture_color = GetPixel(
          (palette.GetXBase() + ZeroExtend32(palette_index)) % VRAM_WIDTH,
          palette.GetYBase()
        );
      }
      else if (texture_mode == GPUTextureMode::Palette8Bit)
      {
          const u16 palette_value = GetPixel(
            (draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x / 2)) % VRAM_WIDTH,
            (draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT
          );
          const u16 palette_index = (palette_value >> ((texcoord_x % 2) * 8)) & 0xFFu;
          texture_color = GetPixel(
            (palette.GetXBase() + ZeroExtend32(palette_index)) % VRAM_WIDTH,
            palette.GetYBase()
          );
      }
      else {
        texture_color = GetPixel(
          (draw_mode.GetTexturePageBaseX() + ZeroExtend32(texcoord_x)) % VRAM_WIDTH,
          (draw_mode.GetTexturePageBaseY() + ZeroExtend32(texcoord_y)) % VRAM_HEIGHT
        );
      }

      // Black with transparent bit clear is always discarded
      // => Alpha = 0%
      if (texture_color == 0)
      {
        image.SetPixel(image_x, image_y, 0);  // transparent black
        continue;
      }

      // Tranparency is enabled iff the transparent bit is set on BOTH
      // the poly and the texel
      // => Alpha = 50%
      //
      // Assuming we're interpreting the alpha channel as normal alpha
      // compositing, 50% alpha is correct only for B/2+F/2 mode, but
      // there's no way to represent the other modes in a normal OBJ
      // file. The correct transparency mode will still be written
      // into the MTL name.
      u8 alpha;
      if (draw_mode.transparency_enable && (texture_color & 0x8000u))
        alpha = 128;
      else
        alpha = 255;

      const u32 rgb = VRAMRGBA5551ToRGBA8888(texture_color) & 0x00FFFFFFu;
      image.SetPixel(image_x, image_y, rgb | (alpha << 24));
    }
  }
#undef GetPixel
}

void UpdateFromVRAM(const u16* vram_ptr)
{
  for (auto& texture : s_textures)
  {
    if (texture.image.IsValid())
      continue;

    if (s_config.dump_full_textures)
    {
      // Expand to maximum UV range
      texture.blob.min_u = 0;
      texture.blob.min_v = 0;
      texture.blob.max_u = 255;
      texture.blob.max_v = 255;
    }

    FillTextureFromVRAM(texture, vram_ptr);
  }

  // All existing textures now closed for editing
  // (But doesn't matter since we only do this once)
  s_texture_cache.clear();
}

static void HashTextureAndAssignFileName(Texture& texture)
{
  if (!texture.filename.empty())
    return;

  if (!texture.image.IsValid())
    return;

  const u32 width = texture.image.GetWidth();
  const u32 height = texture.image.GetHeight();
  const size_t size = width * height * sizeof(u32);
  XXH128_hash_t hash = XXH3_128bits(texture.image.GetPixels(), size);

  texture.filename = fmt::format(
    "{:016X}{:016X}{}",
    hash.high64,
    hash.low64 ^ width,  // add dependence on dimensions
    texture.tstate.transparency_enable ? "t" : ""
  );
}

static void DumpTextures(const std::string& dump_directory)
{
  for (auto& texture : s_textures)
  {
    HashTextureAndAssignFileName(texture);

    if (texture.filename.empty())
      continue;

    const std::string path = Path::Combine(dump_directory, fmt::format("{}.png", texture.filename));

    if (FileSystem::FileExists(path.c_str()))
      continue;

    if (!texture.image.SaveToFile(path.c_str()))
      Host::AddFormattedOSDMessage(10.0f, "Couldn't save texture to '%s'", path.c_str());
  }
}

////////////////////////////////////
// Write OBJ/MTL
////////////////////////////////////
template <class T>
static bool VectorContains(const std::vector<T> vec, const T& x)
{
  for (const auto& elem : vec)
    if (elem == x)
      return true;
  return false;
}

static void FormatMtlNameForPoly(std::string& mtl_name, const Poly& poly)
{
  mtl_name.clear();

  if (poly.transparency_enable)
  {
    switch (poly.transparency_mode)
    {
      case GPUTransparencyMode::HalfBackgroundPlusHalfForeground:
        mtl_name += "B/2+F/2,";
        break;
      case GPUTransparencyMode::BackgroundPlusForeground:
        mtl_name += "B+F,";
        break;
      case GPUTransparencyMode::BackgroundMinusForeground:
        mtl_name += "B-F,";
        break;
      case GPUTransparencyMode::BackgroundPlusQuarterForeground:
        mtl_name += "B+F/4,";
        break;
    }
  }

  if (poly.texture_enable)
    mtl_name += s_textures[poly.texture_index].filename;
  else
    mtl_name += "Untextured";
}

static void WriteNewmtlForPoly(FILE* fp, const Poly& poly, const std::string& mtl_name)
{
  fmt::print(fp, "newmtl {}\n", mtl_name);
  if (poly.texture_enable)
  {
    const Texture& texture = s_textures[poly.texture_index];
    fmt::print(fp, "map_Kd {}.png\n", texture.filename);
    fmt::print(fp, "map_d -imfchan m {}.png\n", texture.filename);
  }
  else
  {
    fmt::print(fp, "Kd 1 1 1\n");
  }
  fmt::print(fp, "\n");
}

// Converts an integer texcoord in [0, width) onto a normalized
// [0.0, 1.0] UV space.
//
// The nth texel index is mapped to the nth texel center, ie.
// (n+0.5)/w.
//
//  +---------------+---------------+----   ----+---------------+
//  |       0       |       1       |    ...    |      w-1      |
//  +-------'-------+-------'-------+----   ----+-------'-------+
//          '               '                           '
//          '               '                           '
//          '               '                           '
//  +-------O-------+-------O-------+--- ... ---+-------O-------+
//  0     0.5/w    1/w    1.5/w    2/w       1-1/w   1-0.5/w    1
//
// The Avocado feature/3d-screenshot branch instead rescales so 0
// becomes 0.0 and w-1 becomes 1.0, ie. n/(w-1).
//
//  +---------------+---------------+----   ----+---------------+
//  |       0       |       1       |    ...    |      w-1      |
//  +------.'-------+------.'-------+----   ----+-------'.------+
//       .'               /                               '.
//     .'               .'                                  '.
//   .'                /                                      '.
//  O---------------+-O-------------+--- ... ---+---------------O
//  0              1/w             2/w        1-1/w             1
//
// I didn't use this because the alignment wrt the texel grid changes
// depending on the position, from the left edge at 0 to the right
// edge at 1. This also means it isn't invariant when cropping. For
// example, 3 in [0,5) maps to near a texel center, but if you crop to
// [3,5), 3 now maps to a texel edge.
//
// What we really want to use is the unknowble inverse of whatever the
// gamedevs used in their model converter...
static float ConvertTexcoord(u32 texcoord, u32 width)
{
  return (float(texcoord) + 0.5f) / float(width);
}

static void Lookup3DVertsForPoly(const Vertex* v3d[4], const Poly& poly)
{
  for (int i = 0; i < poly.NumVerts(); ++i)
  {
    // NOTE: here is where we would apply a correction if the RTPS
    // output and the triangle coords are out of phase
    const s32 sx = poly.v[i].x;
    const s32 sy = poly.v[i].y;

    const u32 key = PackSXYIntoU32(sx, sy);
    auto find_result = s_vertex_cache.find(key);

    if (find_result == s_vertex_cache.end())
    {
      v3d[0] = nullptr;
      return;
    }

    v3d[i] = &find_result->second;
  }
}

static void WriteOBJ(
  const std::string& dump_directory,
  const std::string& filename
)
{
  const std::string obj_path = Path::Combine(dump_directory, fmt::format("{}.obj", filename));
  const std::string mtl_path = Path::Combine(dump_directory, fmt::format("{}.mtl", filename));

  auto obj_file = FileSystem::OpenManagedCFile(obj_path.c_str(), "wb");
  auto mtl_file = FileSystem::OpenManagedCFile(mtl_path.c_str(), "wb");
  if (!obj_file || !mtl_file)
  {
    Host::AddFormattedOSDMessage(10.0f, "Couldn't open OBJ file in '%s'", dump_directory.c_str());
    return;
  }

  const auto obj_fp = obj_file.get();
  const auto mtl_fp = mtl_file.get();

  fmt::print(obj_fp, "# PS1 {} Screenshot\n", s_config.in_2d_mode ? "2D" : "3D");
  fmt::print(obj_fp, "mtllib {}.mtl\n\n", filename);

  const Poly* prev_poly = nullptr;
  u32 stat_obj_faces = 0;
  std::string mtl_name;
  std::vector<std::string> mtls_already_written;

  for (u32 poly_index = 0; poly_index != s_poly_buffer.size(); ++poly_index)
  {
    const Poly& poly = s_poly_buffer[poly_index];

    // v - Vertex position & color
    if (!s_config.in_2d_mode)
    {
      const Vertex* v3d[4];
      Lookup3DVertsForPoly(v3d, poly);

      // Skip if at least one corner didn't have a 3D vertex
      if (!v3d[0])
        continue;

      for (int i = 0; i < poly.NumVerts(); ++i)
      {
        fmt::print(
          obj_fp,
          "v {} {} {} {:.3f} {:.3f} {:.3f}\n",
          v3d[i]->x,
          -v3d[i]->y,
          -v3d[i]->z,
          poly.v[i].r / 255.f,
          poly.v[i].g / 255.f,
          poly.v[i].b / 255.f
        );
      }
    }
    else // 2D mode
    {
      // Convert draw order to Z depth
      const float z_step = 0.5f;
      const float z_min = -z_step * s_poly_buffer.size();
      const float z = z_min + poly_index * z_step;

      for (int i = 0; i < poly.NumVerts(); ++i)
      {
        fmt::print(
          obj_fp,
          "v {} {} {} {:.3f} {:.3f} {:.3f}\n",
          poly.v[i].x,
          -poly.v[i].y,
          z,
          poly.v[i].r / 255.f,
          poly.v[i].g / 255.f,
          poly.v[i].b / 255.f
        );
      }
    }

    // vt - Texture coordinates
    for (int i = 0; i < poly.NumVerts(); ++i)
    {
      if (poly.texture_enable)
      {
        const UVBlob& blob = s_textures[poly.texture_index].blob;
        const float u = ConvertTexcoord(poly.v[i].u - blob.min_u, blob.Width());
        const float v = ConvertTexcoord(poly.v[i].v - blob.min_v, blob.Height());

        fmt::print(obj_fp, "vt {:.4f} {:.4f}\n", u, 1.f - v);
      }
      else
      {
        fmt::print(obj_fp, "vt 0 0\n");
      }
    }

    // usemtl - Material
    const bool needs_usemtl = (
      prev_poly == nullptr ||
      poly.texture_enable != prev_poly->texture_enable ||
      poly.texture_index != prev_poly->texture_index ||
      poly.transparency_enable != prev_poly->transparency_enable ||
      poly.transparency_mode != prev_poly->transparency_mode
    );
    if (needs_usemtl)
    {
      FormatMtlNameForPoly(mtl_name, poly);
      fmt::print(obj_fp, "usemtl {}\n", mtl_name);

      if (!VectorContains(mtls_already_written, mtl_name))
      {
        WriteNewmtlForPoly(mtl_fp, poly, mtl_name);
        mtls_already_written.push_back(mtl_name);
      }
    }

    // f - Face
    if (poly.is_quad)
      fmt::print(obj_fp, "f -4/-4 -2/-2 -1/-1 -3/-3\n");
    else
      fmt::print(obj_fp, "f -3/-3 -1/-1 -2/-2\n");

    fmt::print(obj_fp, "\n");

    stat_obj_faces++;
    prev_poly = &poly;
  }

  fmt::print("Wrote {} OBJ faces.\n", stat_obj_faces);

  std::fflush(obj_fp);
  std::fflush(mtl_fp);
}

////////////////////////////////////
// File stuff
////////////////////////////////////

static std::string GetDumpDirectory()
{
  const std::string game_name = System::GetGameTitle();
  if (game_name.empty())
    return "";
  return Path::Combine(EmuFolders::Screenshots3D, game_name);
}

// Millisecond precision allows multiple shots a second
static std::string GetTimestampStringForFileName()
{
  namespace chrono = std::chrono;

  const auto now = chrono::system_clock::now();
  const auto t = chrono::system_clock::to_time_t(now);

  const auto d = now.time_since_epoch();
  const auto secs = chrono::duration_cast<chrono::seconds>(d);
  const auto millis = chrono::duration_cast<chrono::milliseconds>(d - secs).count();

  return fmt::format("{:%Y-%m-%d_%H-%M-%S}-{:03d}", fmt::localtime(t), millis);
}

////////////////////////////////////
// Frame End processing
////////////////////////////////////
static void TakeShot()
{
  if (s_poly_buffer.empty())
    return;

  const std::string dump_directory = GetDumpDirectory();

  if (dump_directory.empty())
  {
    Host::AddFormattedOSDMessage(10.0f, "Couldn't get directory for 3D screenshots");
    TurnOff();
    return;
  }

  if (!FileSystem::EnsureDirectoryExists(dump_directory.c_str(), false))
  {
    Host::AddFormattedOSDMessage(10.0f, "Couldn't get 3D screenshot directory '%s'", dump_directory.c_str());
    TurnOff();
    return;
  }

  const std::string filename = fmt::format("shot_{}", GetTimestampStringForFileName());

  DumpTextures(dump_directory);
  WriteOBJ(dump_directory, filename);

  s_shots_taken++;

  // Show success message after the last shot
  if (IsDoneShooting())
    Host::AddFormattedOSDMessage(5.0f, "Saved 3D screenshot to '%s'", dump_directory.c_str());
}

static void RecycleVertsAtFrameEnd()
{
  if (s_frame_counter < VERTEX_RECYCLE_PERIOD)
    return;

  std::erase_if(s_vertex_cache, [](const auto& kv) {
    return kv.second.generation <= s_frame_counter - VERTEX_RECYCLE_PERIOD;
  });
}

static void FinishFrame()
{
  if (!s_running)
    return;

  fmt::print(
   "Got {} vertices, {} polys, {} textures.\n",
   s_vertex_cache.size(),
   s_poly_buffer.size(),
   s_textures.size()
  );

  if (!s_config.is_dry_run)
  {
    if (IsWarmedUp() && !IsDoneShooting())
      TakeShot();

    if (IsDoneShooting())
    {
      TurnOff();
      return;
    }
  }

  RecycleVertsAtFrameEnd();

  s_poly_buffer.clear();
  s_textures.clear();
  s_texture_cache.clear();
  s_frame_counter++;
}

void NextFrame()
{
  FinishFrame();

  // Process UI actions
  if (s_run_requested && !s_running)
  {
    s_config = s_ui_config;
    TurnOn();
    s_run_requested = false;
  }
}

////////////////////////////////////
// GUI
////////////////////////////////////
static void HelpIcon(const char* msg)
{
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip(msg);
}

void DrawGuiWindow()
{
  const float framebuffer_scale = Host::GetOSDScale();

  ImGui::SetNextWindowSize(
    ImVec2(400.0f * framebuffer_scale, 200.0f * framebuffer_scale),
    ImGuiCond_Once
  );

  if (!ImGui::Begin("3D Screenshot", nullptr))
  {
    ImGui::End();
    return;
  }

  // [Take Screenshot] [Cancel] [Progress Bar]
  if (!s_running && !s_run_requested)
  {
    if (ImGui::Button("Take Screenshot"))
      s_run_requested = true;
  }
  else
  {
    ImGui::BeginDisabled();
    ImGui::Button("Take Screenshot");
    ImGui::EndDisabled();

    ImGui::SameLine();

    if (ImGui::Button("Cancel"))
    {
      TurnOff();
      s_run_requested = false;
    }

    ImGui::SameLine();

    if (s_running)
    {
      if (s_config.is_dry_run)
        ImGui::ProgressBar(0.0, ImVec2(-FLT_MIN, 0), "Dry Run...");
      else if (!IsWarmedUp())
        ImGui::ProgressBar(0.0, ImVec2(-FLT_MIN, 0), "Warming Up...");
      else
        ImGui::ProgressBar(float(s_shots_taken) / s_config.num_frames_to_capture);
    }
  }

  if (s_running)
    ImGui::BeginDisabled();

  ImGui::SliderInt("Frames to Capture", &s_ui_config.num_frames_to_capture, 1, 20);

  ImGui::Checkbox("Disable Culling", &s_ui_config.disable_culling);
  HelpIcon(
    "Disables culling of tris which are back-facing or zero-size in\n"
    "screen space. Required or your screenshot will be missing all\n"
    "tris that weren't facing the camera."
  );

  ImGui::Checkbox("Dump Full 256x256 Textures", &s_ui_config.dump_full_textures);
  HelpIcon(
    "Dumps the entire 256x256 texture space instead of cropping\n"
    "to just the part that is used.\n"
    "\n"
    "The cropped region may be of interest, or it may contain\n"
    "garbage in the wrong color format or palette.\n"
    "\n"
    "This also avoids making tons of tiny image files during\n"
    "UV scroll animations where the \"part that is used\" is\n"
    "changing every frame."
  );

  ImGui::Checkbox("2D Mode", &s_ui_config.in_2d_mode);
  HelpIcon(
    "Mostly for debug use.\n"
    "\n"
    "Dumps polys with raw 2D screen coords instead of the\n"
    "reconstructed 3D verts. Drawing order is converted\n"
    "to Z depth.\n"
    "\n"
    "Expect UV issues on 2D elements, especially along the\n"
    "bottom-right edges.\n"
    "\n"
    "Vertices do not jitter in this mode."
  );

  ImGui::Checkbox("Dry Run", &s_ui_config.is_dry_run);
  HelpIcon(
    "Debug use. Runs without taking shots. Can be used\n"
    "to test performance of the jitter implementation."
  );

  if (s_running)
    ImGui::EndDisabled();

  ImGui::End();
}

} // namespace Screenshot3D
