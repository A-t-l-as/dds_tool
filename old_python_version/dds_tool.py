#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dds_tool.py - DDS texture converter (DDS <-> PNG/EXR) with full metadata stored in JSON.

  Export:   python dds_tool.py export texture.dds         -> texture.png + texture.json
  Import:   python dds_tool.py import texture.json        -> texture.dds (same format as the
                                                             original, mipmaps regenerated)
  Create:   python dds_tool.py create image.png -f BC7_UNORM_SRGB
  Info:     python dds_tool.py info texture.dds

  You can also simply drag & drop .dds / .json / .png files onto the script (no subcommand).

Required: numpy, imagecodecs      (pip install numpy imagecodecs)
Optional: etcpak                  (pip install etcpak) - better BC7 compression quality
"""

import sys
import os
import json
import struct
import argparse
import math

import numpy as np

try:
    import imagecodecs
except ImportError:  # pragma: no cover
    imagecodecs = None

try:
    import etcpak
except ImportError:  # pragma: no cover
    etcpak = None

TOOL_VERSION = 3


class DDSError(Exception):
    pass


# =============================================================================
#  DDS constants
# =============================================================================
DDSD_CAPS, DDSD_HEIGHT, DDSD_WIDTH, DDSD_PITCH = 0x1, 0x2, 0x4, 0x8
DDSD_PIXELFORMAT, DDSD_MIPMAPCOUNT, DDSD_LINEARSIZE, DDSD_DEPTH = 0x1000, 0x20000, 0x80000, 0x800000
DDSCAPS_COMPLEX, DDSCAPS_TEXTURE, DDSCAPS_MIPMAP = 0x8, 0x1000, 0x400000
DDSCAPS2_CUBEMAP, DDSCAPS2_VOLUME = 0x200, 0x200000
CUBE_FACE_FLAGS = [0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000]
CUBE_FACE_NAMES = ["posx", "negx", "posy", "negy", "posz", "negz"]
DDPF_ALPHAPIXELS, DDPF_ALPHA, DDPF_FOURCC, DDPF_PAL8 = 0x1, 0x2, 0x4, 0x20
DDPF_RGB, DDPF_YUV, DDPF_LUMINANCE = 0x40, 0x200, 0x20000
DDPF_BUMPLUMINANCE, DDPF_BUMPDUDV = 0x40000, 0x80000
DIM_1D, DIM_2D, DIM_3D = 2, 3, 4
MISC_TEXTURECUBE = 0x4


# =============================================================================
#  Format descriptors
# =============================================================================
class Fmt:
    """kind == 'raw': uncompressed, channels = [(name, bit_offset, bits, type)]
       kind == 'bc' : block compressed BC1..BC7"""

    def __init__(self, name, kind, esize=0, channels=(), srgb=False, bc=0,
                 signed=False, special=None):
        self.name = name
        self.kind = kind
        self.esize = esize
        self.channels = list(channels)
        self.srgb = srgb
        self.bc = bc
        self.signed = signed
        self.special = special

    @property
    def block_bytes(self):
        return 8 if self.bc in (1, 4) else 16

    def surface_size(self, w, h):
        if self.kind == 'bc':
            return max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * self.block_bytes
        return w * h * self.esize

    def row_pitch(self, w):
        if self.kind == 'bc':
            return max(1, (w + 3) // 4) * self.block_bytes
        return w * self.esize

    def value_channels(self):
        """Logical channels (without padding) as [(name, bits, type)]."""
        if self.kind == 'bc':
            if self.bc in (1, 2, 3, 7):
                return [(c, 8, 'unorm') for c in 'rgba']
            t = 'snorm' if self.signed else 'unorm'
            if self.bc == 4:
                return [('r', 8, t)]
            if self.bc == 5:
                return [('r', 8, t), ('g', 8, t)]
            return [(c, 16, 'float') for c in 'rgb']  # BC6H
        if self.special == 'rgb9e5':
            return [(c, 9, 'float') for c in 'rgb']
        return [(n, b, t) for (n, o, b, t) in self.channels if t != 'x']


def _seq(bits, ctype, names):
    return [(n, i * bits, bits, ctype) for i, n in enumerate(names)]


def _build_dxgi():
    T = {}

    def raw(num, name, esize, chans, srgb=False, special=None):
        T[num] = Fmt(name, 'raw', esize, chans, srgb=srgb, special=special)

    def bc(num, name, n, srgb=False, signed=False):
        T[num] = Fmt(name, 'bc', bc=n, srgb=srgb, signed=signed)

    # 128 bit
    for n, t in ((1, 'uint'), (2, 'float'), (3, 'uint'), (4, 'sint')):
        sfx = {1: 'TYPELESS', 2: 'FLOAT', 3: 'UINT', 4: 'SINT'}[n]
        raw(n, 'R32G32B32A32_' + sfx, 16, _seq(32, t, 'rgba'))
    for n, t in ((5, 'uint'), (6, 'float'), (7, 'uint'), (8, 'sint')):
        sfx = {5: 'TYPELESS', 6: 'FLOAT', 7: 'UINT', 8: 'SINT'}[n]
        raw(n, 'R32G32B32_' + sfx, 12, _seq(32, t, 'rgb'))
    for n, t, s in ((9, 'unorm', 'TYPELESS'), (10, 'float', 'FLOAT'), (11, 'unorm', 'UNORM'),
                    (12, 'uint', 'UINT'), (13, 'snorm', 'SNORM'), (14, 'sint', 'SINT')):
        raw(n, 'R16G16B16A16_' + s, 8, _seq(16, t, 'rgba'))
    for n, t, s in ((15, 'uint', 'TYPELESS'), (16, 'float', 'FLOAT'), (17, 'uint', 'UINT'), (18, 'sint', 'SINT')):
        raw(n, 'R32G32_' + s, 8, _seq(32, t, 'rg'))
    raw(19, 'R32G8X24_TYPELESS', 8, [('r', 0, 32, 'float'), ('g', 32, 8, 'uint')])
    raw(20, 'D32_FLOAT_S8X24_UINT', 8, [('r', 0, 32, 'float'), ('g', 32, 8, 'uint')])
    raw(21, 'R32_FLOAT_X8X24_TYPELESS', 8, [('r', 0, 32, 'float')])
    raw(22, 'X32_TYPELESS_G8X24_UINT', 8, [('g', 32, 8, 'uint')])
    rgb10a2 = lambda t: [('r', 0, 10, t), ('g', 10, 10, t), ('b', 20, 10, t), ('a', 30, 2, t)]
    raw(23, 'R10G10B10A2_TYPELESS', 4, rgb10a2('unorm'))
    raw(24, 'R10G10B10A2_UNORM', 4, rgb10a2('unorm'))
    raw(25, 'R10G10B10A2_UINT', 4, rgb10a2('uint'))
    raw(26, 'R11G11B10_FLOAT', 4, [('r', 0, 11, 'uf11'), ('g', 11, 11, 'uf11'), ('b', 22, 10, 'uf10')])
    for n, t, s, srgb in ((27, 'unorm', 'TYPELESS', False), (28, 'unorm', 'UNORM', False),
                          (29, 'unorm', 'UNORM_SRGB', True), (30, 'uint', 'UINT', False),
                          (31, 'snorm', 'SNORM', False), (32, 'sint', 'SINT', False)):
        raw(n, 'R8G8B8A8_' + s, 4, _seq(8, t, 'rgba'), srgb=srgb)
    for n, t, s in ((33, 'unorm', 'TYPELESS'), (34, 'float', 'FLOAT'), (35, 'unorm', 'UNORM'),
                    (36, 'uint', 'UINT'), (37, 'snorm', 'SNORM'), (38, 'sint', 'SINT')):
        raw(n, 'R16G16_' + s, 4, _seq(16, t, 'rg'))
    for n, t, s in ((39, 'uint', 'R32_TYPELESS'), (40, 'float', 'D32_FLOAT'), (41, 'float', 'R32_FLOAT'),
                    (42, 'uint', 'R32_UINT'), (43, 'sint', 'R32_SINT')):
        raw(n, s, 4, [('r', 0, 32, t)])
    raw(44, 'R24G8_TYPELESS', 4, [('r', 0, 24, 'unorm'), ('g', 24, 8, 'uint')])
    raw(45, 'D24_UNORM_S8_UINT', 4, [('r', 0, 24, 'unorm'), ('g', 24, 8, 'uint')])
    raw(46, 'R24_UNORM_X8_TYPELESS', 4, [('r', 0, 24, 'unorm')])
    raw(47, 'X24_TYPELESS_G8_UINT', 4, [('g', 24, 8, 'uint')])
    for n, t, s in ((48, 'unorm', 'TYPELESS'), (49, 'unorm', 'UNORM'), (50, 'uint', 'UINT'),
                    (51, 'snorm', 'SNORM'), (52, 'sint', 'SINT')):
        raw(n, 'R8G8_' + s, 2, _seq(8, t, 'rg'))
    for n, t, s in ((53, 'unorm', 'R16_TYPELESS'), (54, 'float', 'R16_FLOAT'), (55, 'unorm', 'D16_UNORM'),
                    (56, 'unorm', 'R16_UNORM'), (57, 'uint', 'R16_UINT'), (58, 'snorm', 'R16_SNORM'),
                    (59, 'sint', 'R16_SINT')):
        raw(n, s, 2, [('r', 0, 16, t)])
    for n, t, s in ((60, 'unorm', 'TYPELESS'), (61, 'unorm', 'UNORM'), (62, 'uint', 'UINT'),
                    (63, 'snorm', 'SNORM'), (64, 'sint', 'SINT')):
        raw(n, 'R8_' + s, 1, [('r', 0, 8, t)])
    raw(65, 'A8_UNORM', 1, [('a', 0, 8, 'unorm')])
    raw(67, 'R9G9B9E5_SHAREDEXP', 4, [], special='rgb9e5')
    bc(70, 'BC1_TYPELESS', 1); bc(71, 'BC1_UNORM', 1); bc(72, 'BC1_UNORM_SRGB', 1, srgb=True)
    bc(73, 'BC2_TYPELESS', 2); bc(74, 'BC2_UNORM', 2); bc(75, 'BC2_UNORM_SRGB', 2, srgb=True)
    bc(76, 'BC3_TYPELESS', 3); bc(77, 'BC3_UNORM', 3); bc(78, 'BC3_UNORM_SRGB', 3, srgb=True)
    bc(79, 'BC4_TYPELESS', 4); bc(80, 'BC4_UNORM', 4); bc(81, 'BC4_SNORM', 4, signed=True)
    bc(82, 'BC5_TYPELESS', 5); bc(83, 'BC5_UNORM', 5); bc(84, 'BC5_SNORM', 5, signed=True)
    raw(85, 'B5G6R5_UNORM', 2, [('b', 0, 5, 'unorm'), ('g', 5, 6, 'unorm'), ('r', 11, 5, 'unorm')])
    raw(86, 'B5G5R5A1_UNORM', 2, [('b', 0, 5, 'unorm'), ('g', 5, 5, 'unorm'), ('r', 10, 5, 'unorm'),
                                   ('a', 15, 1, 'unorm')])
    bgra = [('b', 0, 8, 'unorm'), ('g', 8, 8, 'unorm'), ('r', 16, 8, 'unorm'), ('a', 24, 8, 'unorm')]
    bgrx = [('b', 0, 8, 'unorm'), ('g', 8, 8, 'unorm'), ('r', 16, 8, 'unorm'), ('x', 24, 8, 'x')]
    raw(87, 'B8G8R8A8_UNORM', 4, bgra)
    raw(88, 'B8G8R8X8_UNORM', 4, bgrx)
    raw(89, 'R10G10B10_XR_BIAS_A2_UNORM', 4, rgb10a2('unorm'))
    raw(90, 'B8G8R8A8_TYPELESS', 4, bgra)
    raw(91, 'B8G8R8A8_UNORM_SRGB', 4, bgra, srgb=True)
    raw(92, 'B8G8R8X8_TYPELESS', 4, bgrx)
    raw(93, 'B8G8R8X8_UNORM_SRGB', 4, bgrx, srgb=True)
    bc(94, 'BC6H_TYPELESS', 6); bc(95, 'BC6H_UF16', 6); bc(96, 'BC6H_SF16', 6, signed=True)
    bc(97, 'BC7_TYPELESS', 7); bc(98, 'BC7_UNORM', 7); bc(99, 'BC7_UNORM_SRGB', 7, srgb=True)
    raw(115, 'B4G4R4A4_UNORM', 2, [('b', 0, 4, 'unorm'), ('g', 4, 4, 'unorm'), ('r', 8, 4, 'unorm'),
                                   ('a', 12, 4, 'unorm')])
    raw(191, 'A4B4G4R4_UNORM', 2, [('a', 0, 4, 'unorm'), ('b', 4, 4, 'unorm'), ('g', 8, 4, 'unorm'),
                                    ('r', 12, 4, 'unorm')])
    return T


DXGI = _build_dxgi()

UNSUPPORTED_DXGI = {
    66: 'R1_UNORM', 68: 'R8G8_B8G8_UNORM', 69: 'G8R8_G8B8_UNORM', 100: 'AYUV', 101: 'Y410',
    102: 'Y416', 103: 'NV12', 104: 'P010', 105: 'P016', 106: '420_OPAQUE', 107: 'YUY2',
    108: 'Y210', 109: 'Y216', 110: 'NV11', 111: 'AI44', 112: 'IA44', 113: 'P8', 114: 'A8P8',
}

FOURCC_TO_DXGI = {
    b'DXT1': 71, b'DXT2': 74, b'DXT3': 74, b'DXT4': 77, b'DXT5': 77, b'RXGB': 77,
    b'ATI1': 80, b'BC4U': 80, b'BC4S': 81, b'ATI2': 83, b'BC5U': 83, b'BC5S': 84,
    b'BC6H': 95, b'BC7L': 98, b'BC7\0': 98,
}
D3DFMT_TO_DXGI = {36: 11, 110: 13, 111: 54, 112: 34, 113: 10, 114: 41, 115: 16, 116: 2}


def _mask_info(m):
    if not m:
        return None
    off = (m & -m).bit_length() - 1
    bits = bin(m).count('1')
    if (m >> off) != (1 << bits) - 1:
        raise DDSError('Non-contiguous bit mask 0x%08X is not supported' % m)
    return off, bits


def fmt_from_legacy(pf):
    flags, fourcc = pf['flags'], pf['fourcc']
    if flags & DDPF_FOURCC:
        fcc = struct.pack('<I', fourcc)
        if fcc in FOURCC_TO_DXGI:
            f = DXGI[FOURCC_TO_DXGI[fcc]]
            return f, fcc.decode('latin1').strip('\0')
        if fourcc in D3DFMT_TO_DXGI:
            f = DXGI[D3DFMT_TO_DXGI[fourcc]]
            return f, 'D3DFMT_%d (%s)' % (fourcc, f.name)
        raise DDSError('Unsupported FourCC: %r (%d)' % (fcc, fourcc))
    if flags & (DDPF_YUV | DDPF_PAL8):
        raise DDSError('YUV / palettized formats are not supported')
    bpp = pf['bitcount']
    if bpp not in (8, 16, 24, 32):
        raise DDSError('Unsupported bits per pixel: %d' % bpp)
    masks = (pf['rmask'], pf['gmask'], pf['bmask'], pf['amask'])
    if flags & DDPF_BUMPDUDV:
        types = ['snorm', 'snorm', 'snorm', 'snorm' if bpp == 32 and _mask_info(masks[3]) and
                 _mask_info(masks[3])[1] == 8 else 'unorm']
        names = ['r', 'g', 'b', 'a']
    elif flags & DDPF_BUMPLUMINANCE:
        types, names = ['snorm', 'snorm', 'unorm', 'unorm'], ['r', 'g', 'b', 'a']
    elif flags & DDPF_LUMINANCE:
        types, names = ['unorm'] * 4, ['l', None, None, 'a']
    elif (flags & DDPF_ALPHA) and not (flags & DDPF_RGB):
        types, names = ['unorm'] * 4, [None, None, None, 'a']
    else:
        types, names = ['unorm'] * 4, ['r', 'g', 'b', 'a']
    if not (flags & (DDPF_ALPHAPIXELS | DDPF_ALPHA)) and not (flags & DDPF_BUMPDUDV):
        names[3] = None
    full = (1 << bpp) - 1
    # some tools write masks shifted outside the pixel bit range
    masks = tuple((m >> (32 - bpp)) if (m & ~full) and bpp < 32 else m for m in masks)
    chans = []
    used = 0
    for m, n, t in zip(masks, names, types):
        if n is None or not m:
            continue
        off, bits = _mask_info(m)
        chans.append((n, off, bits, t))
        used |= m
    if not chans:
        raise DDSError('Pixel format has no channels')
    # unused bits in the alpha position (e.g. X8R8G8B8) are filled with ones
    free = full & ~used
    if free and not any(c[0] == 'a' for c in chans):
        off = (free & -free).bit_length() - 1
        bits = bin(free).count('1')
        if (free >> off) == (1 << bits) - 1:
            chans.append(('x', off, bits, 'x'))
    desc = 'legacy %dbpp' % bpp
    desc += ' ' + ''.join('%s%d' % (c[0].upper(), c[2]) for c in sorted(chans, key=lambda c: -c[1]))
    return Fmt(desc, 'raw', bpp // 8, chans), desc


# =============================================================================
#  DDS header
# =============================================================================
def read_dds(path):
    with open(path, 'rb') as f:
        data = f.read()
    if len(data) < 128 or data[:4] != b'DDS ':
        raise DDSError('Not a DDS file: %s' % path)
    h = struct.unpack_from('<7I11I', data, 4)
    size, flags, height, width, pitch, depth, mips = h[:7]
    reserved1 = list(h[7:18])
    pf = dict(zip(['size', 'flags', 'fourcc', 'bitcount', 'rmask', 'gmask', 'bmask', 'amask'],
                  struct.unpack_from('<8I', data, 76)))
    caps, caps2, caps3, caps4, reserved2 = struct.unpack_from('<5I', data, 108)
    hdr = dict(flags=flags, pitch_or_linear_size=pitch, reserved1=reserved1, pixel_format=pf,
               caps=caps, caps2=caps2, caps3=caps3, caps4=caps4, reserved2=reserved2,
               mip_map_count=mips, depth_field=depth)
    offset = 128
    dx10 = None
    if (pf['flags'] & DDPF_FOURCC) and pf['fourcc'] == struct.unpack('<I', b'DX10')[0]:
        dxgi, dim, misc, arr, misc2 = struct.unpack_from('<5I', data, 128)
        dx10 = dict(dxgi_format=dxgi, resource_dimension=dim, misc_flag=misc, array_size=arr,
                    misc_flags2=misc2)
        offset = 148
        if dxgi in UNSUPPORTED_DXGI:
            raise DDSError('DXGI format %s is not supported' % UNSUPPORTED_DXGI[dxgi])
        if dxgi not in DXGI:
            raise DDSError('Unknown DXGI format: %d' % dxgi)
        fmt = DXGI[dxgi]
        fmt_desc = fmt.name
    else:
        fmt, fmt_desc = fmt_from_legacy(pf)

    width = max(1, width)
    height = max(1, height)
    mip_levels = max(1, mips)
    array_size = 1
    faces = [0]
    volume = False
    dim = DIM_2D
    if dx10:
        dim = dx10['resource_dimension']
        array_size = max(1, dx10['array_size'])
        if dim == DIM_3D:
            volume = True
            depth = max(1, depth)
            array_size = 1
        else:
            depth = 1
            if dim == DIM_1D:
                height = 1
            if dx10['misc_flag'] & MISC_TEXTURECUBE:
                faces = list(range(6))
    else:
        if caps2 & DDSCAPS2_VOLUME:
            volume = True
            dim = DIM_3D
            depth = max(1, depth)
        else:
            depth = 1
            if caps2 & DDSCAPS2_CUBEMAP:
                faces = [i for i, fl in enumerate(CUBE_FACE_FLAGS) if caps2 & fl] or list(range(6))
    tex = dict(width=width, height=height, depth=depth, mip_levels=mip_levels,
               array_size=array_size, faces=faces, volume=volume, dimension=dim,
               cubemap=len(faces) > 1 or (dx10 is not None and bool(dx10['misc_flag'] & MISC_TEXTURECUBE))
               or bool(caps2 & DDSCAPS2_CUBEMAP))
    return dict(header=hdr, dx10=dx10, fmt=fmt, fmt_desc=fmt_desc, tex=tex,
                data=memoryview(data)[offset:])


def mip_dims(w, h, d, level):
    return max(1, w >> level), max(1, h >> level), max(1, d >> level)


def full_mip_count(w, h, d=1):
    return int(math.floor(math.log2(max(w, h, d)))) + 1


def iterate_surfaces(dds):
    """Returns a list of (array_idx, face, mip, slice, offset, w, h) for all surfaces."""
    fmt, t = dds['fmt'], dds['tex']
    out = []
    off = 0
    if t['volume']:
        for m in range(t['mip_levels']):
            w, h, d = mip_dims(t['width'], t['height'], t['depth'], m)
            sz = fmt.surface_size(w, h)
            for z in range(d):
                out.append((0, 0, m, z, off, w, h))
                off += sz
    else:
        for a in range(t['array_size']):
            for face in t['faces']:
                for m in range(t['mip_levels']):
                    w, h, _ = mip_dims(t['width'], t['height'], 1, m)
                    out.append((a, face, m, 0, off, w, h))
                    off += fmt.surface_size(w, h)
    return out, off


# =============================================================================
#  Uncompressed formats: unpack / pack
# =============================================================================
def _sign_extend(raw, bits):
    raw = raw.astype(np.int64)
    return raw - ((raw >> (bits - 1)) & 1) * (1 << bits)


def _aligned(off, bits):
    return off % 8 == 0 and bits in (8, 16, 32) and (off // 8) % (bits // 8) == 0


def _extract(a, off, bits):
    """a: (N, esize) uint8 -> raw channel values (uint64)."""
    if _aligned(off, bits):
        nb = bits // 8
        sub = np.ascontiguousarray(a[:, off // 8: off // 8 + nb])
        return sub.view('<u%d' % nb)[:, 0].astype(np.uint64)
    es = a.shape[1]
    if es > 8:
        raise DDSError('Unsupported bit layout')
    word = np.zeros(a.shape[0], np.uint64)
    for k in range(es):
        word |= a[:, k].astype(np.uint64) << np.uint64(8 * k)
    return (word >> np.uint64(off)) & np.uint64((1 << bits) - 1)


def _float16_from_bits(bits16):
    return bits16.astype(np.uint16).view(np.float16).astype(np.float32)


def _raw_to_value(raw, bits, ct):
    if ct == 'unorm':
        return (raw.astype(np.float64) / ((1 << bits) - 1)).astype(np.float32)
    if ct == 'snorm':
        s = _sign_extend(raw, bits).astype(np.float64)
        return np.maximum(s / ((1 << (bits - 1)) - 1), -1.0).astype(np.float32)
    if ct == 'uint':
        return raw.astype(np.float64 if bits > 24 else np.float32)
    if ct == 'sint':
        return _sign_extend(raw, bits).astype(np.float64 if bits > 24 else np.float32)
    if ct == 'float':
        if bits == 16:
            return _float16_from_bits(raw)
        if bits == 32:
            return raw.astype(np.uint32).view(np.float32).copy()
    if ct == 'uf11':
        return _float16_from_bits(raw << np.uint64(4))
    if ct == 'uf10':
        return _float16_from_bits(raw << np.uint64(5))
    raise DDSError('Unknown channel type %s' % ct)


def _value_to_raw(v, bits, ct):
    mask = (1 << bits) - 1
    if ct == 'unorm':
        return np.rint(np.clip(v, 0, 1) * mask).astype(np.uint64)
    if ct == 'snorm':
        s = np.rint(np.clip(v, -1, 1) * ((1 << (bits - 1)) - 1)).astype(np.int64)
        return (s & mask).astype(np.uint64)
    if ct == 'uint':
        return np.clip(np.rint(v), 0, mask).astype(np.uint64)
    if ct == 'sint':
        s = np.clip(np.rint(v), -(1 << (bits - 1)), (1 << (bits - 1)) - 1).astype(np.int64)
        return (s & mask).astype(np.uint64)
    if ct == 'float':
        if bits == 16:
            return np.asarray(v, np.float32).astype(np.float16).view(np.uint16).astype(np.uint64)
        return np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    if ct in ('uf11', 'uf10'):
        sh, mx, top = (4, 0x7BF, 65024.0) if ct == 'uf11' else (5, 0x3DF, 64512.0)
        v = np.nan_to_num(np.asarray(v, np.float32), nan=0.0, posinf=top, neginf=0.0)
        h = np.clip(v, 0, top).astype(np.float16).view(np.uint16).astype(np.uint64)
        r = (h + np.uint64(1 << (sh - 1))) >> np.uint64(sh)
        return np.minimum(r, np.uint64(mx))
    if ct == 'x':
        return np.full(np.shape(v), mask, np.uint64)
    raise DDSError('Unknown channel type %s' % ct)


def _rgb9e5_decode(word):
    word = word.astype(np.uint64)
    e = ((word >> np.uint64(27)) & np.uint64(31)).astype(np.float64)
    scale = np.exp2(e - 15 - 9)
    out = {}
    for i, c in enumerate('rgb'):
        m = ((word >> np.uint64(9 * i)) & np.uint64(511)).astype(np.float64)
        out[c] = (m * scale).astype(np.float32)
    return out


def _rgb9e5_encode(r, g, b):
    MAXV = 511.0 / 512.0 * 2.0 ** 16
    rr, gg, bb = [np.clip(np.nan_to_num(np.asarray(x, np.float64)), 0, MAXV) for x in (r, g, b)]
    mx = np.maximum(np.maximum(rr, gg), bb)
    exp_p = np.maximum(-16, np.floor(np.log2(np.maximum(mx, 1e-30)))) + 1 + 15
    denom = np.exp2(exp_p - 15 - 9)
    mm = np.floor(mx / denom + 0.5)
    fix = mm == 512
    denom = np.where(fix, denom * 2, denom)
    exp_p = np.where(fix, exp_p + 1, exp_p)
    word = (exp_p.astype(np.uint64) & np.uint64(31)) << np.uint64(27)
    for i, x in enumerate((rr, gg, bb)):
        m = np.clip(np.floor(x / denom + 0.5), 0, 511).astype(np.uint64)
        word |= m << np.uint64(9 * i)
    return word


def raw_decode(buf, fmt, w, h):
    n = w * h
    a = np.frombuffer(buf, np.uint8, count=n * fmt.esize).reshape(n, fmt.esize)
    if fmt.special == 'rgb9e5':
        d = _rgb9e5_decode(_extract(a, 0, 32))
        return {k: v.reshape(h, w) for k, v in d.items()}
    out = {}
    for (name, off, bits, ct) in fmt.channels:
        if ct == 'x':
            continue
        out[name] = _raw_to_value(_extract(a, off, bits), bits, ct).reshape(h, w)
    return out


def raw_encode(vals, fmt, w, h):
    """vals: dict name -> (h, w) values -> surface bytes."""
    n = w * h
    if fmt.special == 'rgb9e5':
        word = _rgb9e5_encode(*(vals[c].reshape(n) for c in 'rgb'))
        return word.astype('<u4').tobytes()
    es = fmt.esize
    out = np.zeros((n, es), np.uint8)
    word = np.zeros(n, np.uint64) if es <= 8 else None
    for (name, off, bits, ct) in fmt.channels:
        v = vals[name].reshape(n) if ct != 'x' else np.zeros(n, np.float32)
        raw = _value_to_raw(v, bits, ct)
        if word is not None:
            word |= (raw & np.uint64((1 << bits) - 1)) << np.uint64(off)
        else:  # elements > 8 bytes always have 32-bit aligned channels
            nb = bits // 8
            out[:, off // 8: off // 8 + nb] = raw.astype('<u%d' % nb).view(np.uint8).reshape(n, nb)
    if word is not None:
        for k in range(es):
            out[:, k] = ((word >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    return out.tobytes()


# =============================================================================
#  4x4 blocks
# =============================================================================
def to_blocks(arr):
    """(H, W, C) -> (N, 16, C), edge-padded to a multiple of 4."""
    H, W, C = arr.shape
    bh, bw = (H + 3) // 4, (W + 3) // 4
    if bh * 4 != H or bw * 4 != W:
        arr = np.pad(arr, ((0, bh * 4 - H), (0, bw * 4 - W), (0, 0)), mode='edge')
    return arr.reshape(bh, 4, bw, 4, C).transpose(0, 2, 1, 3, 4).reshape(bh * bw, 16, C)


def from_blocks(b, H, W):
    bh, bw = (H + 3) // 4, (W + 3) // 4
    C = b.shape[-1]
    arr = b.reshape(bh, bw, 4, 4, C).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, C)
    return arr[:H, :W]


def _expand565(c):
    c = c.astype(np.int64)
    r = (c >> 11) & 31
    g = (c >> 5) & 63
    b = c & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1)


def _bc1_palette(c0, c1, four_only):
    """c0, c1: (N,) uint16 -> paleta (N, 4, 4) RGBA 0..255 (float)."""
    e0 = _expand565(c0).astype(np.float32)
    e1 = _expand565(c1).astype(np.float32)
    four = (c0 > c1) | four_only
    p2 = np.where(four[:, None], (2 * e0 + e1) / 3, (e0 + e1) / 2)
    p3 = np.where(four[:, None], (e0 + 2 * e1) / 3, 0)
    pal = np.stack([e0, e1, p2, p3], 1)
    alpha = np.full((len(c0), 4), 255.0, np.float32)
    alpha[:, 3] = np.where(four, 255.0, 0.0)
    return np.concatenate([np.rint(pal), alpha[..., None]], -1)


def _bc1_decode_blocks(blk8, four_only=False):
    u = blk8.view('<u2')
    c0, c1 = u[:, 0], u[:, 1]
    idx_word = blk8.view('<u4')[:, 1].astype(np.int64)
    idx = (idx_word[:, None] >> (2 * np.arange(16))) & 3
    pal = _bc1_palette(c0, c1, four_only)
    return np.take_along_axis(pal, idx[:, :, None], 1)  # (N,16,4)


def _bc4_palette(r0, r1, signed):
    r0 = r0.astype(np.float32)
    r1 = r1.astype(np.float32)
    lo, hi = (-127.0, 127.0) if signed else (0.0, 255.0)
    eight = r0 > r1
    pal = np.zeros((len(r0), 8), np.float32)
    pal[:, 0] = r0
    pal[:, 1] = r1
    for i in range(2, 8):
        p8 = ((8 - i) * r0 + (i - 1) * r1) / 7
        if i <= 5:
            p6 = ((6 - i) * r0 + (i - 1) * r1) / 5
        else:
            p6 = np.full_like(r0, lo if i == 6 else hi)
        pal[:, i] = np.where(eight, p8, p6)
    return pal


def _bc4_decode_blocks(blk8, signed):
    if signed:
        r = blk8[:, :2].view(np.int8).astype(np.int64)
        r = np.maximum(r, -127)
    else:
        r = blk8[:, :2].astype(np.int64)
    w = np.zeros(len(blk8), np.int64)
    for k in range(2, 8):
        w |= blk8[:, k].astype(np.int64) << (8 * (k - 2))
    idx = (w[:, None] >> (3 * np.arange(16))) & 7
    pal = _bc4_palette(r[:, 0], r[:, 1], signed)
    v = np.take_along_axis(pal, idx, 1)
    return v / (127.0 if signed else 255.0)  # (N,16)


def bc_decode(buf, fmt, w, h):
    """Returns dict channel -> (h, w) values, normalized as in value_channels()."""
    bh, bw = max(1, (h + 3) // 4), max(1, (w + 3) // 4)
    nb = bh * bw
    bs = fmt.block_bytes
    raw = np.frombuffer(buf, np.uint8, count=nb * bs).reshape(nb, bs)
    n = fmt.bc
    if n in (6, 7):
        if imagecodecs is None:
            raise DDSError('The imagecodecs package is required to decode BC6H/BC7')
        code = (-6 if fmt.signed else 6) if n == 6 else 7
        ch = 3 if n == 6 else 4
        img = imagecodecs.bcn_decode(raw.tobytes(), code, shape=(bh * 4, bw * 4, ch))
        img = img[:h, :w].astype(np.float32)
        if n == 7:
            img /= 255.0
        return {c: img[..., i] for i, c in enumerate('rgba'[:ch])}
    if n == 1:
        px = _bc1_decode_blocks(raw) / 255.0
    elif n == 2:
        px = _bc1_decode_blocks(np.ascontiguousarray(raw[:, 8:]), True) / 255.0
        a = raw[:, :8].view('<u8')[:, 0]
        px[:, :, 3] = ((a[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & np.uint64(15)) / 15.0
    elif n == 3:
        px = _bc1_decode_blocks(np.ascontiguousarray(raw[:, 8:]), True) / 255.0
        px[:, :, 3] = _bc4_decode_blocks(np.ascontiguousarray(raw[:, :8]), False)
    elif n == 4:
        px = _bc4_decode_blocks(raw, fmt.signed)[:, :, None]
    elif n == 5:
        px = np.stack([_bc4_decode_blocks(np.ascontiguousarray(raw[:, :8]), fmt.signed),
                       _bc4_decode_blocks(np.ascontiguousarray(raw[:, 8:]), fmt.signed)], -1)
    img = from_blocks(px.astype(np.float32), bh * 4, bw * 4)[:h, :w]
    return {c: img[..., i] for i, c in enumerate('rgba'[:img.shape[-1]])}


# =============================================================================
#  BC encoders (vectorized, numpy)
# =============================================================================
CHUNK = 32768  # blocks per chunk (limits memory usage)


def _masked_stats(px, mask):
    w = mask.astype(np.float32)
    cnt = np.maximum(w.sum(1, keepdims=True), 1)
    mean = (px * w[..., None]).sum(1) / cnt
    return w, mean


def _pca_endpoints(px, mask):
    """Endpoints along the principal axis (PCA) of the masked pixels."""
    w, mean = _masked_stats(px, mask)
    d = (px - mean[:, None, :]) * w[..., None]
    cov = np.einsum('npi,npj->nij', d, d)
    k = np.argmax(np.einsum('nii->ni', cov), 1)
    axis = cov[np.arange(len(px)), k]
    for _ in range(6):
        axis = np.einsum('nij,nj->ni', cov, axis)
        nrm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = axis / np.maximum(nrm, 1e-12)
    t = np.einsum('npc,nc->np', px - mean[:, None, :], axis)
    tmin = np.where(mask, t, np.inf).min(1)
    tmax = np.where(mask, t, -np.inf).max(1)
    tmin = np.where(np.isfinite(tmin), tmin, 0)
    tmax = np.where(np.isfinite(tmax), tmax, 0)
    return mean + tmin[:, None] * axis, mean + tmax[:, None] * axis


def _ls_endpoints(px, t, mask, e0, e1):
    """Least-squares endpoint fit for known interpolation weights t."""
    w = mask.astype(np.float32)
    s = 1 - t
    a = (s * s * w).sum(1)
    b = (s * t * w).sum(1)
    c = (t * t * w).sum(1)
    r0 = ((s * w)[..., None] * px).sum(1)
    r1 = ((t * w)[..., None] * px).sum(1)
    det = a * c - b * b
    ok = np.abs(det) > 1e-8
    sd = np.where(ok, det, 1)[:, None]
    n0 = (c[:, None] * r0 - b[:, None] * r1) / sd
    n1 = (a[:, None] * r1 - b[:, None] * r0) / sd
    return np.where(ok[:, None], n0, e0), np.where(ok[:, None], n1, e1)


def _assign(px, pal, mask=None):
    d = ((px[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = np.argmin(d, 2)
    err = np.take_along_axis(d, idx[..., None], 2)[..., 0]
    if mask is not None:
        err = err * mask
    return idx, err.sum(1)


def _pick(best, cand, better):
    for k in best:
        b = better.reshape((-1,) + (1,) * (best[k].ndim - 1))
        best[k] = np.where(b, cand[k], best[k])


def _put(lo, hi, pos, n, val):
    val = val.astype(np.uint64) & np.uint64((1 << n) - 1)
    if pos + n <= 64:
        lo |= val << np.uint64(pos)
    elif pos >= 64:
        hi |= val << np.uint64(pos - 64)
    else:
        k = 64 - pos
        lo |= (val & np.uint64((1 << k) - 1)) << np.uint64(pos)
        hi |= val >> np.uint64(k)


def _blocks128(lo, hi):
    out = np.empty((len(lo), 2), '<u8')
    out[:, 0] = lo
    out[:, 1] = hi
    return out.view(np.uint8).reshape(-1, 16)


# ---------------------------------------------------------------- BC1 color
def _q565(e):
    e = np.clip(e, 0, 255)
    r = np.rint(e[:, 0] * 31 / 255).astype(np.int64)
    g = np.rint(e[:, 1] * 63 / 255).astype(np.int64)
    b = np.rint(e[:, 2] * 31 / 255).astype(np.int64)
    return (r << 11) | (g << 5) | b


def _bc1_color(px, transparent=None):
    """px: (N,16,3) 0..255. transparent: (N,16) bool or None.
       Returns (N, 8) bytes of BC1 color blocks."""
    N = len(px)
    mask = np.ones((N, 16), bool) if transparent is None else ~transparent
    three = np.zeros(N, bool) if transparent is None else transparent.any(1)
    T4 = np.array([0, 1, 1 / 3, 2 / 3], np.float32)
    T3 = np.array([0, 1, 0.5, 0], np.float32)

    def evaluate(e0, e1):
        c0, c1 = _q565(e0), _q565(e1)
        pal4 = _bc1_palette(np.maximum(c0, c1), np.minimum(c0, c1), True)[:, :, :3]
        pal3 = _bc1_palette(np.minimum(c0, c1), np.maximum(c0, c1), False)[:, :3, :3]
        i4, er4 = _assign(px, pal4, mask)
        if three.any():
            i3, er3 = _assign(px, pal3, mask)
        else:
            i3, er3 = i4, er4
        idx = np.where(three[:, None], i3, i4)
        err = np.where(three, er3, er4)
        a = np.where(three, np.minimum(c0, c1), np.maximum(c0, c1))
        b = np.where(three, np.maximum(c0, c1), np.minimum(c0, c1))
        return dict(c0=a, c1=b, idx=idx, err=err)

    e0, e1 = _pca_endpoints(px, mask)
    best = evaluate(e0, e1)
    for _ in range(2):
        t = np.where(three[:, None], T3[best['idx']], T4[best['idx']])
        e0, e1 = _ls_endpoints(px, t, mask, _expand565(best['c0']).astype(np.float32),
                               _expand565(best['c1']).astype(np.float32))
        cand = evaluate(e0, e1)
        _pick(best, cand, cand['err'] < best['err'])
    idx = best['idx']
    if transparent is not None:
        idx = np.where(transparent, 3, idx)
    # c0 == c1 in 4-color mode -> all indices 0
    same = (best['c0'] == best['c1']) & ~three
    idx = np.where(same[:, None], 0, idx)
    word = np.zeros(N, np.uint64)
    for i in range(16):
        word |= idx[:, i].astype(np.uint64) << np.uint64(2 * i)
    out = np.empty((N, 4), '<u2')
    out[:, 0] = best['c0']
    out[:, 1] = best['c1']
    out.view('<u4')[:, 1] = word
    return out.view(np.uint8).reshape(N, 8)


# ---------------------------------------------------------------- BC4
def _bc4(px, signed):
    """px: (N,16) in 0..255 (unorm) or -127..127 (snorm) -> (N, 8) bytes."""
    N = len(px)
    lo, hi = (-127, 127) if signed else (0, 255)
    px = np.clip(px, lo, hi).astype(np.float32)
    TT = np.array([0, 1] + [k / 7 for k in range(1, 7)], np.float32)

    def evaluate(a, b, six=False):
        a = np.clip(np.rint(a), lo, hi).astype(np.int64)
        b = np.clip(np.rint(b), lo, hi).astype(np.int64)
        if six:
            r0, r1 = np.minimum(a, b), np.maximum(a, b)
        else:
            r0, r1 = np.maximum(a, b), np.minimum(a, b)
        pal = _bc4_palette(r0, r1, signed)
        d = (px[:, :, None] - pal[:, None, :]) ** 2
        idx = np.argmin(d, 2)
        err = np.take_along_axis(d, idx[..., None], 2)[..., 0].sum(1)
        return dict(r0=r0, r1=r1, idx=idx, err=err)

    mn, mx = px.min(1), px.max(1)
    best = evaluate(mx, mn)
    t = TT[best['idx']]
    e0, e1 = _ls_endpoints(px[..., None], t, np.ones_like(px, bool),
                           best['r0'][:, None].astype(np.float32), best['r1'][:, None].astype(np.float32))
    cand = evaluate(e0[:, 0], e1[:, 0])
    _pick(best, cand, cand['err'] < best['err'])
    # 6-value mode (with exact extremes) - useful for alpha masks
    inner = (px > lo) & (px < hi)
    if inner.any():
        imn = np.where(inner, px, hi).min(1)
        imx = np.where(inner, px, lo).max(1)
        imn = np.where(inner.any(1), imn, mn)
        imx = np.where(inner.any(1), imx, mx)
        cand = evaluate(imn, imx, six=True)
        _pick(best, cand, cand['err'] < best['err'])
    word = np.zeros(N, np.uint64)
    for i in range(16):
        word |= best['idx'][:, i].astype(np.uint64) << np.uint64(3 * i)
    word = (word << np.uint64(16)) | ((best['r1'].astype(np.uint64) & np.uint64(255)) << np.uint64(8)) \
        | (best['r0'].astype(np.uint64) & np.uint64(255))
    return word.astype('<u8').view(np.uint8).reshape(N, 8)


# ---------------------------------------------------------------- BC6H
W16 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], np.int64)


def _bc6_unq(x, bits, signed):
    if not signed:
        return np.where(x == 0, 0, np.where(x == (1 << bits) - 1, 0xFFFF, ((x << 16) + 0x8000) >> bits))
    s = x < 0
    m = np.abs(x)
    u = np.where(m == 0, 0, np.where(m >= (1 << (bits - 1)) - 1, 0x7FFF, ((m << 15) + 0x4000) >> (bits - 1)))
    return np.where(s, -u, u)


def _bc6_finish(v, signed):
    if not signed:
        return (v * 31) >> 6
    return np.where(v < 0, -(((-v) * 31) >> 5), (v * 31) >> 5)


def _bc6_quant(f, bits, signed):
    """Value in the 'half bits' domain -> quantized endpoint."""
    if not signed:
        u = np.clip(f, 0, None) * 64 / 31
        x = np.rint((u - 2.0 ** (15 - bits)) / 2.0 ** (16 - bits))
        return np.clip(x, 0, (1 << bits) - 1).astype(np.int64)
    u = np.abs(f) * 32 / 31
    b = bits - 1
    x = np.clip(np.rint((u - 2.0 ** (14 - b)) / 2.0 ** (15 - b)), 0, (1 << b) - 1)
    return (np.sign(f) * x).astype(np.int64)


def _bc6_palette(x0, x1, bits, signed):
    u0 = _bc6_unq(x0, bits, signed)[:, None, :]
    u1 = _bc6_unq(x1, bits, signed)[:, None, :]
    w = W16[None, :, None]
    return _bc6_finish((u0 * (64 - w) + u1 * w + 32) >> 6, signed).astype(np.float64)


def _bc6h(rgb, signed):
    """rgb: (N,16,3) float -> (N,16) bytes (modes 11 and 12, single region)."""
    N = len(rgb)
    v = np.nan_to_num(rgb.astype(np.float32), nan=0.0, posinf=65504.0, neginf=-65504.0)
    v = np.clip(v, -65504, 65504) if signed else np.clip(v, 0, 65504)
    h = v.astype(np.float16).view(np.uint16).astype(np.int64)
    f = np.where(h & 0x8000, -(h & 0x7FFF), h).astype(np.float64)
    mask = np.ones((N, 16), bool)
    T = W16 / 64.0

    def evaluate(e0, e1, mode):
        if mode == 11:
            x0, x1 = _bc6_quant(e0, 10, signed), _bc6_quant(e1, 10, signed)
            pal = _bc6_palette(x0, x1, 10, signed)
        else:
            x0, x1 = _bc6_quant(e0, 11, signed), _bc6_quant(e1, 11, signed)
            lim = (1 << 10) - 1 if signed else (1 << 11) - 1
            x1 = np.clip(x0 + np.clip(x1 - x0, -255, 255), -lim if signed else 0, lim)
            pal = _bc6_palette(x0, x1, 11, signed)
        idx, err = _assign(f, pal)
        return dict(x0=x0, x1=x1, idx=idx, err=err, mode=np.full(N, mode))

    e0, e1 = _pca_endpoints(f, mask)
    best = None
    for mode in (11, 12):
        cur = evaluate(e0, e1, mode)
        for _ in range(2):
            a, b = _ls_endpoints(f, T[cur['idx']], mask, e0, e1)
            cand = evaluate(a, b, mode)
            _pick(cur, cand, cand['err'] < cur['err'])
        if best is None:
            best = cur
        else:
            _pick(best, cur, cur['err'] < best['err'])
    # pixel 0 (anchor) must have an index < 8
    swap = best['idx'][:, 0] >= 8
    x0 = np.where(swap[:, None], best['x1'], best['x0'])
    x1 = np.where(swap[:, None], best['x0'], best['x1'])
    idx = np.where(swap[:, None], 15 - best['idx'], best['idx'])
    lo = np.zeros(N, np.uint64)
    hi = np.zeros(N, np.uint64)
    m11 = best['mode'] == 11
    lo11, hi11 = lo.copy(), hi.copy()
    _put(lo11, hi11, 0, 5, np.full(N, 0x03))
    for c in range(3):
        _put(lo11, hi11, 5 + 10 * c, 10, x0[:, c] & 0x3FF)
        _put(lo11, hi11, 35 + 10 * c, 10, x1[:, c] & 0x3FF)
    lo12, hi12 = lo.copy(), hi.copy()
    _put(lo12, hi12, 0, 5, np.full(N, 0x07))
    d = (x1 - x0) & 0x1FF
    w = x0 & 0x7FF
    for c in range(3):
        _put(lo12, hi12, 5 + 10 * c, 10, w[:, c] & 0x3FF)
        _put(lo12, hi12, 35 + 10 * c, 9, d[:, c])
        _put(lo12, hi12, 44 + 10 * c, 1, w[:, c] >> 10)
    lo = np.where(m11, lo11, lo12)
    hi = np.where(m11, hi11, hi12)
    pos = 65
    for i in range(16):
        n = 3 if i == 0 else 4
        _put(lo, hi, pos, n, idx[:, i])
        pos += n
    return _blocks128(lo, hi)


# ---------------------------------------------------------------- BC7 (mode 6)
def _bc7_q(e):
    """Quantize an RGBA endpoint to 7 bits + shared p-bit."""
    best_c, best_p, best_err = None, None, None
    for p in (0, 1):
        c = np.clip(np.rint((e - p) / 2), 0, 127).astype(np.int64)
        err = ((c * 2 + p - e) ** 2).sum(1)
        if best_err is None:
            best_c, best_p, best_err = c, np.zeros(len(e), np.int64), err
        else:
            better = err < best_err
            best_c = np.where(better[:, None], c, best_c)
            best_p = np.where(better, 1, best_p)
            best_err = np.minimum(err, best_err)
    return best_c, best_p


def _bc7_mode6(px):
    """px: (N,16,4) 0..255 -> (N,16) bytes."""
    N = len(px)
    px = px.astype(np.float32)
    mask = np.ones((N, 16), bool)
    T = W16 / 64.0

    def evaluate(e0, e1):
        c0, p0 = _bc7_q(e0)
        c1, p1 = _bc7_q(e1)
        v0 = (c0 * 2 + p0[:, None])[:, None, :]
        v1 = (c1 * 2 + p1[:, None])[:, None, :]
        w = W16[None, :, None]
        pal = ((v0 * (64 - w) + v1 * w + 32) >> 6).astype(np.float32)
        idx, err = _assign(px, pal)
        return dict(c0=c0, c1=c1, p0=p0, p1=p1, idx=idx, err=err)

    e0, e1 = _pca_endpoints(px, mask)
    best = evaluate(e0, e1)
    for _ in range(2):
        a, b = _ls_endpoints(px, T[best['idx']], mask, e0, e1)
        cand = evaluate(a, b)
        _pick(best, cand, cand['err'] < best['err'])
    swap = best['idx'][:, 0] >= 8
    c0 = np.where(swap[:, None], best['c1'], best['c0'])
    c1 = np.where(swap[:, None], best['c0'], best['c1'])
    p0 = np.where(swap, best['p1'], best['p0'])
    p1 = np.where(swap, best['p0'], best['p1'])
    idx = np.where(swap[:, None], 15 - best['idx'], best['idx'])
    lo = np.zeros(N, np.uint64)
    hi = np.zeros(N, np.uint64)
    _put(lo, hi, 0, 7, np.full(N, 1 << 6))
    pos = 7
    for c in range(4):
        _put(lo, hi, pos, 7, c0[:, c]); pos += 7
        _put(lo, hi, pos, 7, c1[:, c]); pos += 7
    _put(lo, hi, 63, 1, p0)
    _put(lo, hi, 64, 1, p1)
    pos = 65
    for i in range(16):
        n = 3 if i == 0 else 4
        _put(lo, hi, pos, n, idx[:, i])
        pos += n
    return _blocks128(lo, hi)


# ---------------------------------------------------------------- dispatcher
def bc_encode(vals, fmt, w, h, opts):
    """vals: dict channel -> (h, w) values (as in value_channels) -> bytes."""
    n = fmt.bc
    if n in (1, 2, 3, 7):
        img = np.stack([vals.get(c, np.ones((h, w), np.float32) if c == 'a' else np.zeros((h, w), np.float32))
                        for c in 'rgba'], -1)
        img = np.clip(img, 0, 1) * 255
    elif n in (4, 5):
        sc = 127.0 if fmt.signed else 255.0
        img = np.stack([vals[c] for c in 'rg'[:n - 3]], -1) * sc
    else:
        img = np.stack([vals[c] for c in 'rgb'], -1)

    if n == 7 and etcpak is not None and not opts.get('bc7_builtin'):
        bh, bw = (h + 3) // 4, (w + 3) // 4
        full = from_blocks(to_blocks(img), bh * 4, bw * 4)
        rgba = np.ascontiguousarray(np.rint(full).astype(np.uint8))
        params = etcpak.BC7CompressBlockParams()
        lvl = opts.get('bc7_level')
        params.m_uber_level = int(2 if lvl is None else max(0, min(4, lvl)))
        return bytes(etcpak.compress_bc7(rgba.tobytes(), bw * 4, bh * 4, params))

    blocks = to_blocks(img.astype(np.float32))
    out = []
    for s in range(0, len(blocks), CHUNK):
        b = blocks[s:s + CHUNK]
        if n == 1:
            tr = (b[:, :, 3] < 128) if opts.get('bc1_alpha') else None
            out.append(_bc1_color(b[:, :, :3], tr))
        elif n == 2:
            a4 = np.clip(np.rint(b[:, :, 3] * 15 / 255), 0, 15).astype(np.uint64)
            word = np.zeros(len(b), np.uint64)
            for i in range(16):
                word |= a4[:, i] << np.uint64(4 * i)
            out.append(np.concatenate([word.astype('<u8').view(np.uint8).reshape(-1, 8),
                                       _bc1_color(b[:, :, :3])], 1))
        elif n == 3:
            out.append(np.concatenate([_bc4(b[:, :, 3], False), _bc1_color(b[:, :, :3])], 1))
        elif n == 4:
            out.append(_bc4(b[:, :, 0], fmt.signed))
        elif n == 5:
            out.append(np.concatenate([_bc4(b[:, :, 0], fmt.signed), _bc4(b[:, :, 1], fmt.signed)], 1))
        elif n == 6:
            out.append(_bc6h(b, fmt.signed))
        elif n == 7:
            out.append(_bc7_mode6(b))
    return np.concatenate(out, 0).tobytes()


# =============================================================================
#  Image <-> channel values
# =============================================================================
def image_plan(fmt, bc1_alpha=True):
    """Decides the image file kind and the channel mapping."""
    vch = fmt.value_channels()
    if fmt.kind == 'bc' and fmt.bc == 1 and not bc1_alpha:
        vch = vch[:3]
    names = [c[0] for c in vch]
    if len(names) == 1:
        layout = [names[0]]
    elif set(names) == {'l', 'a'}:
        layout = ['l', 'a']
    else:
        layout = ['r', 'g', 'b'] + (['a'] if 'a' in names else [])
        layout = [c if c in names else None for c in layout]
    floaty = any(t in ('float', 'uf11', 'uf10') or (t in ('unorm', 'snorm') and b > 16) for _, b, t in vch)
    if fmt.special == 'rgb9e5' or (fmt.kind == 'bc' and fmt.bc == 6):
        floaty = True
    if floaty:
        kind = 'exr_float'
    elif any(b > 16 for _, b, _ in vch):
        kind = 'exr_uint'
    elif max(b for _, b, _ in vch) > 8:
        kind = 'png16'
    else:
        kind = 'png8'
    return kind, layout


def _chan_info(fmt):
    return {n: (b, t) for n, b, t in fmt.value_channels()}


def values_to_image(vals, fmt, kind, layout, h, w):
    info = _chan_info(fmt)
    planes = []
    for name in layout:
        if name is None or name not in vals:
            planes.append(np.zeros((h, w), np.float64))
            continue
        v = vals[name].astype(np.float64)
        b, t = info[name]
        if kind in ('png8', 'png16'):
            if t == 'unorm':
                p = v
            elif t == 'snorm':
                s = np.rint(np.clip(v, -1, 1) * ((1 << (b - 1)) - 1))
                p = (s + (1 << (b - 1))) / ((1 << b) - 1)
            elif t == 'uint':
                p = v / ((1 << b) - 1)
            else:  # sint
                p = (v + (1 << (b - 1))) / ((1 << b) - 1)
            planes.append(p)
        elif kind == 'exr_uint':
            planes.append(np.where(v < 0, v + 2.0 ** 32, v))
        else:
            planes.append(v)
    arr = np.stack(planes, -1)
    if kind == 'png8':
        return np.rint(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if kind == 'png16':
        return np.rint(np.clip(arr, 0, 1) * 65535).astype(np.uint16)
    if kind == 'exr_uint':
        return arr.astype(np.uint32)
    return arr.astype(np.float32)


def image_to_values(arr, fmt, kind, layout):
    info = _chan_info(fmt)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    C = arr.shape[2]
    if np.issubdtype(arr.dtype, np.integer) and kind != 'exr_uint':
        p_all = arr.astype(np.float64) / np.iinfo(arr.dtype).max
    else:
        p_all = arr.astype(np.float64)
    # adapt the image channel count to the expected layout
    want = len(layout)
    if C == want:
        planes = [p_all[..., i] for i in range(want)]
    elif want == 1:
        planes = [p_all[..., 0]]
    elif want == 2:  # gray + alpha
        planes = [p_all[..., 0], p_all[..., -1] if C in (2, 4) else np.ones(arr.shape[:2])]
    else:
        if C <= 2:
            gray = p_all[..., 0]
            planes = [gray, gray, gray] + ([p_all[..., 1] if C == 2 else np.ones(arr.shape[:2])]
                                           if want == 4 else [])
        else:
            planes = [p_all[..., 0], p_all[..., 1], p_all[..., 2]]
            if want == 4:
                planes.append(p_all[..., 3] if C >= 4 else np.ones(arr.shape[:2]))
    vals = {}
    for name, p in zip(layout, planes):
        if name is None:
            continue
        b, t = info[name]
        if kind in ('png8', 'png16'):
            p = np.clip(p, 0, 1)
            if t == 'unorm':
                v = p
            elif t == 'snorm':
                s = np.rint(p * ((1 << b) - 1)) - (1 << (b - 1))
                v = np.maximum(s / ((1 << (b - 1)) - 1), -1)
            elif t == 'uint':
                v = np.rint(p * ((1 << b) - 1))
            else:
                v = np.rint(p * ((1 << b) - 1)) - (1 << (b - 1))
        elif kind == 'exr_uint':
            v = p
            if t == 'sint':
                v = np.where(v >= 2.0 ** (b - 1), v - 2.0 ** b, v)
        else:
            v = p
        vals[name] = v
    return vals


def save_image(path, arr, kind):
    if imagecodecs is None:
        raise DDSError('The imagecodecs package is missing (pip install imagecodecs)')
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if kind.startswith('png'):
        data = imagecodecs.png_encode(arr)
    else:
        data = imagecodecs.exr_encode(arr)
    with open(path, 'wb') as f:
        f.write(data)


def load_image(path):
    if imagecodecs is None:
        raise DDSError('The imagecodecs package is missing (pip install imagecodecs)')
    with open(path, 'rb') as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext == '.png':
        arr = imagecodecs.png_decode(data)
    elif ext == '.exr':
        arr = imagecodecs.exr_decode(data)
    elif ext in ('.tif', '.tiff'):
        arr = imagecodecs.tiff_decode(data)
    else:
        arr = imagecodecs.imread(data)
    arr = np.asarray(arr)
    while arr.ndim > 3:
        arr = arr[0]
    return arr


# =============================================================================
#  Mipmaps
# =============================================================================
def _srgb_to_lin(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _down_axis(a, axis):
    n = a.shape[axis]
    if n == 1:
        return a
    m = n // 2
    if n % 2 == 0:
        a0 = np.take(a, np.arange(0, n, 2), axis)
        a1 = np.take(a, np.arange(1, n, 2), axis)
        return (a0 + a1) * 0.5
    # odd size: area-weighted box filter
    s = n / m
    Wt = np.zeros((m, n), a.dtype)
    for j in range(m):
        x0, x1 = j * s, (j + 1) * s
        for i in range(int(math.floor(x0)), min(n, int(math.ceil(x1)))):
            Wt[j, i] = (min(x1, i + 1) - max(x0, i)) / s
    r = np.tensordot(Wt, a, axes=([1], [axis]))
    return np.moveaxis(r, 0, axis)


def build_mips(base, levels, srgb_idx, volume):
    """base: (D,H,W,K) float. srgb_idx: indices of sRGB channels. Returns a list of levels."""
    out = [base]
    cur = base
    if srgb_idx:
        cur = cur.copy()
        cur[..., srgb_idx] = _srgb_to_lin(cur[..., srgb_idx])
    for _ in range(1, levels):
        axes = (0, 1, 2) if volume else (1, 2)
        for ax in axes:
            cur = _down_axis(cur, ax)
        lvl = cur
        if srgb_idx:
            lvl = cur.copy()
            lvl[..., srgb_idx] = _lin_to_srgb(lvl[..., srgb_idx])
        out.append(lvl)
    return out


# =============================================================================
#  DDS writer
# =============================================================================
def write_dds(path, meta, fmt, surfaces, w, h, d, mips, array_size, faces):
    hd = meta['header']
    pf = hd['pixel_format']
    flags = hd['flags'] | DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    caps = hd['caps'] | DDSCAPS_TEXTURE
    if mips > 1:
        flags |= DDSD_MIPMAPCOUNT
        caps |= DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
    else:
        caps &= ~DDSCAPS_MIPMAP
    if meta['volume']:
        flags |= DDSD_DEPTH
    if flags & DDSD_LINEARSIZE:
        pitch = fmt.surface_size(w, h)
    elif flags & DDSD_PITCH:
        pitch = fmt.row_pitch(w)
    else:
        pitch = hd['pitch_or_linear_size']
    mip_field = mips if not (mips == 1 and hd.get('mip_map_count', 1) == 0) else 0
    depth_field = d if meta['volume'] else hd.get('depth_field', 0)
    out = bytearray(b'DDS ')
    out += struct.pack('<7I', 124, flags, h, w, pitch, depth_field, mip_field)
    out += struct.pack('<11I', *hd['reserved1'])
    out += struct.pack('<8I', 32, pf['flags'], pf['fourcc'], pf['bitcount'],
                       pf['rmask'], pf['gmask'], pf['bmask'], pf['amask'])
    out += struct.pack('<5I', caps, hd['caps2'], hd['caps3'], hd['caps4'], hd['reserved2'])
    if meta.get('dx10'):
        x = meta['dx10']
        out += struct.pack('<5I', x['dxgi_format'], x['resource_dimension'], x['misc_flag'],
                           x['array_size'] if (array_size == 1 and x['array_size'] == 0) or meta['volume']
                           else array_size, x['misc_flags2'])
    for s in surfaces:
        out += s
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        f.write(out)
    os.replace(tmp, path)


# =============================================================================
#  JSON file: editable / info / technical sections
# =============================================================================
SEC_EDIT, SEC_INFO, SEC_TECH = 'editable', 'info', 'do_not_edit'
# section/key names used by tool_version 2 (still accepted on import)
V2_SECTIONS = {'edytowalne': SEC_EDIT, 'informacje': SEC_INFO, 'nie_zmieniaj': SEC_TECH}
V2_KEYS = {'mipmapy': 'mipmaps'}
FULL_WORDS = ('full', 'max', '', 'pelny', 'pe\u0142ny')
README = [
    "Section 'editable' - safe to change.",
    "Section 'info' - for reference only, ignored on import (dimensions are taken from the images).",
    "Section 'do_not_edit' - technical data needed to reproduce the exact format. "
    "To change the format use: dds_tool.py create image.png -f FORMAT",
]
INFO_KEYS = ('source', 'format', 'container', 'width', 'height', 'depth', 'array_size', 'srgb')
TECH_KEYS = ('dimension', 'volume', 'cubemap', 'cube_faces', 'image_kind', 'image_layout', 'header', 'dx10')


def structure_meta(flat):
    """Flat metadata dict -> readable JSON split into sections."""
    edit = {}
    help_ = {}
    edit['mipmaps'] = 'full' if flat.get('full_mip_chain') else int(flat.get('mip_levels', 1))
    help_['mipmaps'] = ('"full" = full chain computed from the image size, '
                        'a number = exactly that many levels (1 = no mipmaps)')
    if flat.get('bc1_alpha') is not None:
        edit['bc1_alpha'] = bool(flat['bc1_alpha'])
        help_['bc1_alpha'] = 'BC1/DXT1 only: true = 1-bit transparency (alpha < 50% -> transparent)'
    edit['images'] = flat['images']
    help_['images'] = ('image files; "file" may be renamed. The order of entries matters '
                       '(cubemap: +X -X +Y -Y +Z -Z), for 3D textures the "slice" field')
    info = {k: flat.get(k) for k in INFO_KEYS}
    info['original_mip_levels'] = flat.get('mip_levels')
    tech = {k: flat.get(k) for k in TECH_KEYS}
    return {'_readme': README, 'tool': 'dds_tool', 'tool_version': TOOL_VERSION,
            SEC_EDIT: dict(edit, _help=help_), SEC_INFO: info, SEC_TECH: tech}


def flatten_meta(m):
    """Readable JSON (or an old flat one) -> flat dict used by import."""
    m = {V2_SECTIONS.get(k, k): v for k, v in m.items()}
    if SEC_TECH not in m:
        return m  # old flat file format (tool_version 1)
    flat = dict(m.get(SEC_INFO) or {})
    flat.update(m[SEC_TECH])
    e = {V2_KEYS.get(k, k): v for k, v in (m.get(SEC_EDIT) or {}).items()}
    mips = e.get('mipmaps', 'full')
    if isinstance(mips, str) and mips.strip().lower() in FULL_WORDS:
        flat['full_mip_chain'], flat['mip_levels'] = True, 1
    else:
        try:
            n = int(mips)
        except (TypeError, ValueError):
            raise DDSError('Invalid "mipmaps" value: %r (use "full" or a number)' % (mips,))
        flat['full_mip_chain'], flat['mip_levels'] = n <= 0, max(1, n)
    flat['bc1_alpha'] = e.get('bc1_alpha')
    flat['images'] = e.get('images', [])
    flat['tool'] = m.get('tool')
    flat['tool_version'] = m.get('tool_version')
    return flat


def load_meta(path):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            m = json.load(f)
        except json.JSONDecodeError as e:
            raise DDSError('Syntax error in %s (line %d): %s' % (os.path.basename(path), e.lineno, e.msg))
    if not isinstance(m, dict) or m.get('tool') != 'dds_tool':
        raise DDSError('%s is not a dds_tool metadata file' % path)
    return flatten_meta(m)


def save_meta(path, flat):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(structure_meta(flat), f, indent=2, ensure_ascii=False)


# =============================================================================
#  Export
# =============================================================================
DIM_NAMES = {DIM_1D: '1D', DIM_2D: '2D', DIM_3D: '3D'}


def _decode_surface(dds, off, w, h):
    fmt = dds['fmt']
    sz = fmt.surface_size(w, h)
    buf = dds['data'][off:off + sz]
    if len(buf) < sz:
        raise DDSError('The DDS file is truncated (texture data missing)')
    if fmt.kind == 'bc':
        return bc_decode(buf, fmt, w, h)
    return raw_decode(buf, fmt, w, h)


def export_dds(path, outdir=None, log=print):
    dds = read_dds(path)
    fmt, t = dds['fmt'], dds['tex']
    surfaces, total = iterate_surfaces(dds)
    if len(dds['data']) < total:
        log('  warning: the file has less data than the header implies (%d < %d B) - using mip 0 only'
            % (len(dds['data']), total))
    top = [s for s in surfaces if s[2] == 0]
    decoded = [(s, _decode_surface(dds, s[4], s[5], s[6])) for s in top]

    bc1_alpha = None
    if fmt.kind == 'bc' and fmt.bc == 1:
        bc1_alpha = any(bool((v['a'] < 0.5).any()) for _, v in decoded)
    kind, layout = image_plan(fmt, bc1_alpha if bc1_alpha is not None else True)
    ext = '.png' if kind.startswith('png') else '.exr'

    stem = os.path.splitext(os.path.basename(path))[0]
    outdir = outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    images = []
    for (a, face, m, z, off, w, h), vals in decoded:
        name = stem
        if t['array_size'] > 1:
            name += '_a%d' % a
        if t['cubemap']:
            name += '_' + CUBE_FACE_NAMES[face]
        if t['volume'] and t['depth'] > 1:
            name += '_z%03d' % z
        name += ext
        save_image(os.path.join(outdir, name), values_to_image(vals, fmt, kind, layout, h, w), kind)
        images.append(dict(file=name, array_index=a,
                           face=CUBE_FACE_NAMES[face] if t['cubemap'] else None,
                           slice=z if t['volume'] else None))

    hdr = dict(dds['header'])
    meta = dict(
        tool='dds_tool', tool_version=TOOL_VERSION,
        source=os.path.basename(path),
        format=dds['fmt_desc'],
        container='DX10' if dds['dx10'] else 'legacy',
        width=t['width'], height=t['height'], depth=t['depth'],
        mip_levels=t['mip_levels'],
        full_mip_chain=t['mip_levels'] == full_mip_count(t['width'], t['height'], t['depth']),
        dimension=DIM_NAMES.get(t['dimension'], '2D'),
        array_size=t['array_size'],
        cubemap=t['cubemap'],
        cube_faces=[CUBE_FACE_NAMES[f] for f in t['faces']] if t['cubemap'] else None,
        volume=t['volume'],
        srgb=fmt.srgb,
        bc1_alpha=bc1_alpha,
        image_kind=kind,
        image_layout=layout,
        images=images,
        header=hdr,
        dx10=dds['dx10'],
    )
    jpath = os.path.join(outdir, stem + '.json')
    save_meta(jpath, meta)
    log('  %s -> %s (%s, %dx%d, mip: %d) + %s' % (
        os.path.basename(path), ', '.join(i['file'] for i in images[:3]) + (' ...' if len(images) > 3 else ''),
        dds['fmt_desc'], t['width'], t['height'], t['mip_levels'], os.path.basename(jpath)))
    return jpath


# =============================================================================
#  Import (compile)
# =============================================================================
def fmt_from_meta(meta):
    if meta.get('dx10'):
        n = meta['dx10']['dxgi_format']
        if n not in DXGI:
            raise DDSError('Unsupported DXGI format %d' % n)
        return DXGI[n]
    return fmt_from_legacy(meta['header']['pixel_format'])[0]


def import_json(jpath, out=None, mips=None, opts=None, log=print):
    opts = dict(opts or {})
    meta = load_meta(jpath)
    base = os.path.dirname(os.path.abspath(jpath))
    dds_name = os.path.splitext(os.path.basename(jpath))[0] + '.dds'
    return import_meta(meta, base, dds_name, out, mips, opts, log, os.path.basename(jpath))


def import_meta(meta, base, dds_name, out=None, mips=None, opts=None, log=print, label=''):
    opts = dict(opts or {})
    fmt = fmt_from_meta(meta)
    kind, layout = meta['image_kind'], meta['image_layout']
    if fmt.kind == 'bc' and fmt.bc == 1:
        opts['bc1_alpha'] = bool(meta.get('bc1_alpha'))

    chans = [c[0] for c in fmt.value_channels()]
    need64 = any(b > 24 and t in ('uint', 'sint') for _, b, t in fmt.value_channels())
    dt = np.float64 if need64 else np.float32

    def load_vals(img):
        p = os.path.join(base, img['file'])
        if not os.path.exists(p):
            raise DDSError('Image file not found: %s' % p)
        arr = load_image(p)
        vals = image_to_values(arr, fmt, kind, layout)
        h, w = arr.shape[:2]
        planes = []
        for c in chans:
            if c in vals:
                planes.append(vals[c].astype(dt))
            else:
                planes.append(np.full((h, w), 1.0 if c == 'a' else 0.0, dt))
        return np.stack(planes, -1)

    images = meta['images']
    if meta['volume']:
        imgs = sorted(images, key=lambda i: i['slice'] or 0)
        groups = [np.stack([load_vals(i) for i in imgs], 0)]
    else:
        groups = [load_vals(i)[None] for i in images]
    d, h, w = groups[0].shape[:3]
    for g in groups:
        if g.shape[1:3] != (h, w):
            raise DDSError('All images (faces/layers) must have the same size')
    if meta['dimension'] == '1D' and h != 1:
        raise DDSError('A 1D texture must be 1 pixel high')

    maxm = full_mip_count(w, h, d if meta['volume'] else 1)
    if mips is not None:
        levels = maxm if mips <= 0 else min(mips, maxm)
    elif meta.get('full_mip_chain'):
        levels = maxm
    else:
        levels = min(meta['mip_levels'], maxm)

    srgb_idx = [k for k, c in enumerate(chans) if c in 'rgb'] if fmt.srgb else []
    encode = bc_encode if fmt.kind == 'bc' else (lambda v, f, ww, hh, o: raw_encode(v, f, ww, hh))

    def enc(level_arr):
        out = []
        for z in range(level_arr.shape[0]):
            sl = level_arr[z]
            vals = {c: sl[..., k] for k, c in enumerate(chans)}
            out.append(encode(vals, fmt, sl.shape[1], sl.shape[0], opts))
        return out

    surfaces = []
    if meta['volume']:
        for lvl in build_mips(groups[0], levels, srgb_idx, True):
            surfaces.extend(enc(lvl))
    else:
        for g in groups:
            for lvl in build_mips(g, levels, srgb_idx, False):
                surfaces.extend(enc(lvl))

    nfaces = len(meta['cube_faces'] or []) if meta['cubemap'] else 1
    array_size = max(1, len(groups) // max(1, nfaces)) if not meta['volume'] else 1
    if out is None:
        out = os.path.join(base, dds_name)
    elif os.path.isdir(out) or not out.lower().endswith('.dds'):
        os.makedirs(out, exist_ok=True)
        out = os.path.join(out, dds_name)
    write_dds(out, meta, fmt, surfaces, w, h, d if meta['volume'] else 1, levels, array_size,
              meta.get('cube_faces'))
    log('  %s -> %s (%s, %dx%d, mip: %d)' % (label, os.path.basename(out),
                                           meta['format'], w, h, levels))
    return out


# =============================================================================
#  Creating a new texture
# =============================================================================
LEGACY_ALIASES = {'DXT1': 71, 'DXT2': 74, 'DXT3': 74, 'DXT4': 77, 'DXT5': 77,
                  'ATI1': 80, 'ATI2': 83, 'BC4U': 80, 'BC4S': 81, 'BC5U': 83, 'BC5S': 84}


def find_format(name):
    """DXGI name (e.g. BC7_UNORM_SRGB, R8G8B8A8_UNORM) or a legacy FourCC (DXT1, DXT5, ATI2)."""
    key = name.strip().upper().replace('DXGI_FORMAT_', '')
    if key in LEGACY_ALIASES:
        return key, DXGI[LEGACY_ALIASES[key]]
    for num, f in DXGI.items():
        if f.name == key:
            return num, f
    raise DDSError('Unknown format "%s". List: python dds_tool.py formats' % name)


def _new_meta(fmt_name):
    key, fmt = find_format(fmt_name)
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | \
        (DDSD_LINEARSIZE if fmt.kind == 'bc' else DDSD_PITCH)
    if isinstance(key, str):  # legacy FourCC header (best compatibility with older games)
        pf = dict(size=32, flags=DDPF_FOURCC, fourcc=struct.unpack('<I', key.encode())[0],
                  bitcount=0, rmask=0, gmask=0, bmask=0, amask=0)
        dx10 = None
        desc = key
    else:
        pf = dict(size=32, flags=DDPF_FOURCC, fourcc=struct.unpack('<I', b'DX10')[0],
                  bitcount=0, rmask=0, gmask=0, bmask=0, amask=0)
        dx10 = dict(dxgi_format=key, resource_dimension=DIM_2D, misc_flag=0, array_size=1, misc_flags2=0)
        desc = fmt.name
    hdr = dict(flags=flags, pitch_or_linear_size=0, reserved1=[0] * 11, pixel_format=pf,
               caps=DDSCAPS_TEXTURE, caps2=0, caps3=0, caps4=0, reserved2=0, mip_map_count=1,
               depth_field=0)
    return fmt, dict(tool='dds_tool', tool_version=TOOL_VERSION, source=None, format=desc,
                     container='DX10' if dx10 else 'legacy', dimension='2D', volume=False,
                     cubemap=False, cube_faces=None, array_size=1, full_mip_chain=True, mip_levels=1,
                     srgb=fmt.srgb, bc1_alpha=None, header=hdr, dx10=dx10)


def create_dds(image_paths, fmt_name=None, like=None, out=None, mips=None, cube=False,
               write_json=False, opts=None, log=print):
    paths = [os.path.abspath(p) for p in image_paths]
    for p in paths:
        if not os.path.exists(p):
            raise DDSError('File not found: %s' % p)
    if like:
        if like.lower().endswith('.dds'):
            src = read_dds(like)
            fmt, meta = src['fmt'], dict(tool='dds_tool', tool_version=TOOL_VERSION,
                                         format=src['fmt_desc'], header=src['header'],
                                         dx10=dict(src['dx10']) if src['dx10'] else None,
                                         mip_levels=src['tex']['mip_levels'],
                                         full_mip_chain=src['tex']['mip_levels'] == full_mip_count(
                                             src['tex']['width'], src['tex']['height'], src['tex']['depth']))
        else:
            meta = load_meta(like)
            fmt = fmt_from_meta(meta)
        meta = dict(meta, dimension='2D', volume=False, cubemap=False, cube_faces=None, array_size=1,
                    srgb=fmt.srgb)
        meta['header'] = dict(meta['header'], caps2=0)
        if meta.get('dx10'):
            meta['dx10'] = dict(meta['dx10'], resource_dimension=DIM_2D, misc_flag=0, array_size=1)
    elif fmt_name:
        fmt, meta = _new_meta(fmt_name)
    else:
        raise DDSError('Specify a format (--format) or a template (--like)')
    if cube:
        if len(paths) % 6:
            raise DDSError('A cubemap needs 6 images (or a multiple of 6) in the order +X -X +Y -Y +Z -Z')
        meta['cubemap'] = True
        meta['cube_faces'] = list(CUBE_FACE_NAMES)
        meta['header']['caps2'] = DDSCAPS2_CUBEMAP | sum(CUBE_FACE_FLAGS)
        meta['header']['caps'] |= DDSCAPS_COMPLEX
        if meta.get('dx10'):
            meta['dx10']['misc_flag'] |= MISC_TEXTURECUBE
        elif len(paths) > 6:
            raise DDSError('Cubemap arrays require a DXGI format name (e.g. BC1_UNORM instead of DXT1, BC3_UNORM instead of DXT5)')
    elif len(paths) > 1 and not meta.get('dx10'):
        raise DDSError('Texture arrays require a DXGI format name (e.g. BC1_UNORM instead of DXT1, BC3_UNORM instead of DXT5)')

    bc1_alpha = None
    if fmt.kind == 'bc' and fmt.bc == 1:
        bc1_alpha = False
        for p in paths:
            a = load_image(p)
            if a.ndim == 3 and a.shape[2] in (2, 4):
                al = a[..., -1].astype(np.float64)
                if np.issubdtype(a.dtype, np.integer):
                    al /= np.iinfo(a.dtype).max
                bc1_alpha = bc1_alpha or bool((al < 0.5).any())
    kind, layout = image_plan(fmt, bc1_alpha if bc1_alpha is not None else True)
    meta.update(bc1_alpha=bc1_alpha, image_kind=kind, image_layout=layout)
    faces = CUBE_FACE_NAMES if cube else [None]
    meta['images'] = [dict(file=p, array_index=i // len(faces), face=faces[i % len(faces)], slice=None)
                      for i, p in enumerate(paths)]
    if mips is None and not meta.get('full_mip_chain', True):
        mips = meta.get('mip_levels', 1)
    stem = os.path.splitext(os.path.basename(paths[0]))[0]
    base = os.path.dirname(paths[0])
    res = import_meta(meta, base, stem + '.dds', out, mips, opts, log,
                      os.path.basename(paths[0]) + (' (+%d)' % (len(paths) - 1) if len(paths) > 1 else ''))
    if write_json:
        jm = dict(meta, source=os.path.basename(res),
                  images=[dict(i, file=os.path.relpath(i['file'], os.path.dirname(res)))
                          for i in meta['images']])
        d = read_dds(res)
        jm.update(width=d['tex']['width'], height=d['tex']['height'], depth=1,
                  mip_levels=d['tex']['mip_levels'], array_size=d['tex']['array_size'],
                  header=d['header'], dx10=d['dx10'])
        save_meta(os.path.splitext(res)[0] + '.json', jm)
    return res


def list_formats(log=print):
    log('Legacy headers (FourCC): ' + ', '.join(LEGACY_ALIASES))
    log('DXGI formats:')
    names = [f.name for _, f in sorted(DXGI.items()) if 'TYPELESS' not in f.name]
    for i in range(0, len(names), 3):
        log('  ' + ''.join('%-30s' % n for n in names[i:i + 3]))
    log('Typical: color -> BC7_UNORM_SRGB or BC1_UNORM_SRGB / BC3_UNORM_SRGB, normal map -> BC5_UNORM,')
    log('         single-channel mask -> BC4_UNORM, HDR -> BC6H_UF16, uncompressed -> R8G8B8A8_UNORM(_SRGB)')


# =============================================================================
#  Info
# =============================================================================
def info_dds(path, log=print):
    dds = read_dds(path)
    t = dds['tex']
    _, total = iterate_surfaces(dds)
    log('%s' % path)
    log('  format:     %s (%s)' % (dds['fmt_desc'], 'DX10' if dds['dx10'] else 'legacy'))
    log('  size:       %d x %d%s' % (t['width'], t['height'], (' x %d' % t['depth']) if t['volume'] else ''))
    log('  type:       %s%s%s' % (DIM_NAMES.get(t['dimension'], '2D'), ', cubemap' if t['cubemap'] else '',
                                  (', array x%d' % t['array_size']) if t['array_size'] > 1 else ''))
    log('  mipmaps:    %d%s' % (t['mip_levels'], ' (full chain)' if t['mip_levels'] == full_mip_count(
        t['width'], t['height'], t['depth']) else ''))
    log('  sRGB:       %s' % ('yes' if dds['fmt'].srgb else 'no'))
    log('  data:       %d B (expected %d B)' % (len(dds['data']), total))


# =============================================================================
#  CLI
# =============================================================================
IMAGE_EXTS = ('.png', '.exr', '.tif', '.tiff')


def find_json_for_image(path):
    d = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    cand = os.path.join(d, os.path.splitext(name)[0] + '.json')
    if os.path.exists(cand):
        return cand
    for fn in os.listdir(d):
        if fn.lower().endswith('.json'):
            try:
                m = load_meta(os.path.join(d, fn))
                if any(i['file'] == name for i in m.get('images', [])):
                    return os.path.join(d, fn)
            except Exception:
                pass
    raise DDSError('No .json file found for %s' % path)


def collect(paths, exts, recursive):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(exts)]
                if not recursive:
                    break
        else:
            out.append(p)
    return out


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass
    if imagecodecs is None:
        print('The imagecodecs package is missing. Install it with: pip install numpy imagecodecs')
        return 1
    argv = list(sys.argv[1:] if argv is None else argv)
    auto = bool(argv) and argv[0] not in ('export', 'import', 'info', 'create', 'formats', '-h', '--help')
    if auto:
        argv = ['auto'] + argv
    ap = argparse.ArgumentParser(prog='dds_tool', description='DDS <-> PNG/EXR + JSON converter')
    sub = ap.add_subparsers(dest='cmd')
    pe = sub.add_parser('export', help='DDS -> image(s) + JSON (largest mipmap only)')
    pe.add_argument('files', nargs='+')
    pe.add_argument('-o', '--outdir', help='output directory')
    pe.add_argument('-r', '--recursive', action='store_true', help='search subdirectories')
    pi = sub.add_parser('import', help='JSON (+ image) -> DDS with generated mipmaps')
    pi.add_argument('files', nargs='+', help='.json files (or images that have a .json)')
    pi.add_argument('-o', '--out', help='output file or directory')
    pi.add_argument('-r', '--recursive', action='store_true')
    pi.add_argument('--mips', type=int, help='number of mipmaps (0 = full chain, 1 = no mipmaps)')
    pi.add_argument('--bc7-builtin', action='store_true', help='use the built-in BC7 encoder instead of etcpak')
    pi.add_argument('--bc7-level', type=int, help='etcpak BC7 quality (0-4, default 2; higher = slower)')
    pc = sub.add_parser('create', help='new DDS texture from any image (PNG/EXR/TIFF...)')
    pc.add_argument('images', nargs='+', help='image(s); several = texture array, with --cube = cubemap')
    pc.add_argument('-f', '--format', help='format, e.g. BC7_UNORM_SRGB, BC5_UNORM, DXT5 (list: formats)')
    pc.add_argument('--like', help='copy the format and header from an existing .dds or .json file')
    pc.add_argument('-o', '--out', help='output file or directory')
    pc.add_argument('--mips', type=int, help='number of mipmaps (default: full chain)')
    pc.add_argument('--cube', action='store_true', help='6 images: +X -X +Y -Y +Z -Z')
    pc.add_argument('--json', action='store_true', help='also write a .json for later editing')
    pc.add_argument('--bc7-builtin', action='store_true', help='use the built-in BC7 encoder instead of etcpak')
    pc.add_argument('--bc7-level', type=int, help='etcpak BC7 quality (0-4, default 2; higher = slower)')
    sub.add_parser('formats', help='list supported formats')
    pn = sub.add_parser('info', help='show information about a DDS file')
    pn.add_argument('files', nargs='+')
    pa = sub.add_parser('auto')
    pa.add_argument('files', nargs='+')
    args = ap.parse_args(argv)
    if not args.cmd:
        ap.print_help()
        return 1

    errors = 0

    def run(fn, *a, **k):
        nonlocal errors
        try:
            fn(*a, **k)
        except DDSError as e:
            errors += 1
            print('  ERROR: %s' % e)
        except Exception as e:  # pragma: no cover
            errors += 1
            print('  ERROR (%s): %s' % (type(e).__name__, e))

    if args.cmd == 'export':
        for p in collect(args.files, ('.dds',), args.recursive):
            run(export_dds, p, args.outdir)
    elif args.cmd == 'import':
        opts = dict(bc7_builtin=args.bc7_builtin, bc7_level=args.bc7_level)
        for p in collect(args.files, ('.json',), args.recursive):
            if p.lower().endswith(IMAGE_EXTS):
                try:
                    p = find_json_for_image(p)
                except DDSError as e:
                    errors += 1
                    print('  ERROR: %s' % e)
                    continue
            run(import_json, p, args.out, args.mips, opts)
    elif args.cmd == 'create':
        opts = dict(bc7_builtin=args.bc7_builtin, bc7_level=args.bc7_level)
        run(create_dds, args.images, args.format, args.like, args.out, args.mips, args.cube, args.json, opts)
    elif args.cmd == 'formats':
        list_formats()
    elif args.cmd == 'info':
        for p in collect(args.files, ('.dds',), False):
            run(info_dds, p)
    else:  # auto: drag & dropped files
        done = set()
        for p in collect(args.files, ('.dds', '.json'), False):
            low = p.lower()
            if low.endswith('.dds'):
                run(export_dds, p)
            else:
                try:
                    j = find_json_for_image(p) if low.endswith(IMAGE_EXTS) else p
                except DDSError as e:
                    errors += 1
                    print('  ERROR: %s' % e)
                    continue
                if j not in done:
                    done.add(j)
                    run(import_json, j)
        if os.name == 'nt' and sys.stdin and sys.stdin.isatty():
            input('\nDone. Press Enter...')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
