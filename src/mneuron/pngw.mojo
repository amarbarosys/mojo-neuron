"""PNG writer through libpng's simplified API (png_image: 104 B, version at 8, width 12, height 16, format 20,
colormap_entries 28). Lifted from png_write in ~/Projects/tools/AgentModder/mods/avatars/splat/make.mojo (same owner;
that file points back here). Difference: fopen mode "wbx" (O_EXCL) and png_image_write_to_stdio, so an existing path
is refused with no check-then-write race.

png_write_rgba: 8-bit RGBA. png_write_mask: the same pixels as a 2-entry palette (transparent, red) with tRNS; a mask
is two-valued and 4x less data for zlib (6144 x 3456: 448 ms as RGBA)."""

from std.ffi import OwnedDLHandle, external_call

comptime LIB = "libpng16.so.16"
comptime MODE = "wbx"  # glibc: x = O_EXCL, an existing path fails
comptime PNG_FORMAT_RGBA = 3
comptime PNG_FORMAT_RGBA_COLORMAP = 11  # RGBA | COLORMAP (8): the buffer holds palette indices


def _write(
    path: String,
    buf: Int,
    w: Int,
    h: Int,
    format: Int,
    colormap: Int,
    entries: Int,
) raises:
    var png: OwnedDLHandle
    try:
        png = OwnedDLHandle(LIB)
    except e:
        raise Error("pngw: cannot load " + LIB + ": " + String(e))
    var c = path.copy()
    var f = external_call["fopen", OpaquePointer[MutAnyOrigin]](
        c.as_c_string_span().ptr(), MODE.as_c_string_span().ptr()
    )
    if Int(f) == 0:
        raise Error("pngw: cannot create " + path + " (exists or not writable)")
    var img = List[UInt8](length=104, fill=0)
    var p = img.unsafe_ptr()
    p.unsafe_bitcast[UInt32]()[unsafe_offset=2] = 1  # PNG_IMAGE_VERSION
    p.unsafe_bitcast[UInt32]()[unsafe_offset=3] = UInt32(w)
    p.unsafe_bitcast[UInt32]()[unsafe_offset=4] = UInt32(h)
    p.unsafe_bitcast[UInt32]()[unsafe_offset=5] = UInt32(format)
    p.unsafe_bitcast[UInt32]()[unsafe_offset=7] = UInt32(entries)
    var ok = png.get_function[Int32]("png_image_write_to_stdio")(
        p, f, Int32(0), buf, Int32(0), colormap
    )
    var closed = external_call["fclose", Int32](Int(f))
    if not ok or closed != 0:
        _ = external_call["unlink", Int32](
            c.as_c_string_span().ptr()
        )  # our own partial file
        raise Error("pngw: write failed for " + path)


def _check(px: Span[UInt8, _], w: Int, h: Int) raises:
    if w <= 0 or h <= 0 or len(px) < w * h * 4:
        raise Error(
            "pngw: "
            + String(len(px))
            + " bytes do not hold "
            + String(w)
            + "x"
            + String(h)
            + " RGBA"
        )


def png_write_rgba(path: String, px: Span[UInt8, _], w: Int, h: Int) raises:
    _check(px, w, h)
    _write(path, Int(px.unsafe_ptr()), w, h, PNG_FORMAT_RGBA, 0, 0)


def png_write_mask(path: String, px: Span[UInt8, _], w: Int, h: Int) raises:
    """Write RGBA mask pixels (transparent where alpha is 0, else red 255,0,0,255) as a 2-colour indexed PNG.
    Reads back as RGBA equal to the raster `mask_raster` makes."""
    _check(px, w, h)
    var idx = List[UInt8](length=w * h, fill=0)
    for i in range(w * h):
        if px[i * 4 + 3] != 0:
            idx[i] = 1
    var palette: List[UInt8] = [0, 0, 0, 0, 255, 0, 0, 255]
    _write(
        path,
        Int(idx.unsafe_ptr()),
        w,
        h,
        PNG_FORMAT_RGBA_COLORMAP,
        Int(palette.unsafe_ptr()),
        2,
    )
    _ = (idx[0], palette[0])  # both live until libpng has read them
