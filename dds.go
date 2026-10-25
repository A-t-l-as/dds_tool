package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	DDSD_CAPS        = 0x1
	DDSD_HEIGHT      = 0x2
	DDSD_WIDTH       = 0x4
	DDSD_PITCH       = 0x8
	DDSD_PIXELFORMAT = 0x1000
	DDSD_MIPMAPCOUNT = 0x20000
	DDSD_LINEARSIZE  = 0x80000
	DDSD_DEPTH       = 0x800000

	DDSCAPS_COMPLEX = 0x8
	DDSCAPS_TEXTURE = 0x1000
	DDSCAPS_MIPMAP  = 0x400000

	DDSCAPS2_CUBEMAP = 0x200
	DDSCAPS2_VOLUME  = 0x200000

	DDPF_ALPHAPIXELS   = 0x1
	DDPF_ALPHA         = 0x2
	DDPF_FOURCC        = 0x4
	DDPF_PAL8          = 0x20
	DDPF_RGB           = 0x40
	DDPF_YUV           = 0x200
	DDPF_LUMINANCE     = 0x20000
	DDPF_BUMPLUMINANCE = 0x40000
	DDPF_BUMPDUDV      = 0x80000

	DIM_1D = 2
	DIM_2D = 3
	DIM_3D = 4

	MISC_TEXTURECUBE = 0x4
)

var CubeFaceFlags = []uint32{0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000}
var CubeFaceNames = []string{"posx", "negx", "posy", "negy", "posz", "negz"}
var DimNames = map[uint32]string{DIM_1D: "1D", DIM_2D: "2D", DIM_3D: "3D"}

var fourccDX10 = binary.LittleEndian.Uint32([]byte("DX10"))

type PixelFormat struct {
	Size     uint32 `json:"size"`
	Flags    uint32 `json:"flags"`
	FourCC   uint32 `json:"fourcc"`
	BitCount uint32 `json:"bitcount"`
	RMask    uint32 `json:"rmask"`
	GMask    uint32 `json:"gmask"`
	BMask    uint32 `json:"bmask"`
	AMask    uint32 `json:"amask"`
}

type Header struct {
	Flags       uint32      `json:"flags"`
	Pitch       uint32      `json:"pitch_or_linear_size"`
	Reserved1   []uint32    `json:"reserved1"`
	PixelFormat PixelFormat `json:"pixel_format"`
	Caps        uint32      `json:"caps"`
	Caps2       uint32      `json:"caps2"`
	Caps3       uint32      `json:"caps3"`
	Caps4       uint32      `json:"caps4"`
	Reserved2   uint32      `json:"reserved2"`
	MipMapCount uint32      `json:"mip_map_count"`
	DepthField  uint32      `json:"depth_field"`
}

type DX10 struct {
	DXGIFormat        uint32 `json:"dxgi_format"`
	ResourceDimension uint32 `json:"resource_dimension"`
	MiscFlag          uint32 `json:"misc_flag"`
	ArraySize         uint32 `json:"array_size"`
	MiscFlags2        uint32 `json:"misc_flags2"`
}

type TexInfo struct {
	Width, Height, Depth int
	MipLevels            int
	ArraySize            int
	Faces                []int
	Volume               bool
	Dim                  uint32
	Cubemap              bool
}

type DDSFile struct {
	Header  Header
	DX10    *DX10
	Fmt     *Fmt
	FmtDesc string
	Tex     TexInfo
	Data    []byte
}

type Surface struct {
	Array, Face, Mip, Slice int
	Offset                  int
	W, H                    int
}

func readDDS(path string) (*DDSFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 128 || string(data[:4]) != "DDS " {
		return nil, fmt.Errorf("Not a DDS file: %s", path)
	}
	u := func(off int) uint32 { return binary.LittleEndian.Uint32(data[off:]) }
	d := &DDSFile{}
	h := &d.Header
	h.Flags = u(8)
	height, width := u(12), u(16)
	h.Pitch = u(20)
	depth := u(24)
	h.DepthField = depth
	h.MipMapCount = u(28)
	h.Reserved1 = make([]uint32, 11)
	for i := range h.Reserved1 {
		h.Reserved1[i] = u(32 + 4*i)
	}
	h.PixelFormat = PixelFormat{u(76), u(80), u(84), u(88), u(92), u(96), u(100), u(104)}
	h.Caps, h.Caps2, h.Caps3, h.Caps4, h.Reserved2 = u(108), u(112), u(116), u(120), u(124)
	offset := 128
	pf := h.PixelFormat
	if pf.Flags&DDPF_FOURCC != 0 && pf.FourCC == fourccDX10 {
		if len(data) < 148 {
			return nil, fmt.Errorf("Truncated DX10 header: %s", path)
		}
		d.DX10 = &DX10{u(128), u(132), u(136), u(140), u(144)}
		offset = 148
		n := int(d.DX10.DXGIFormat)
		if name, ok := UnsupportedDXGI[n]; ok {
			return nil, fmt.Errorf("DXGI format %s is not supported", name)
		}
		f, ok := DXGI[n]
		if !ok {
			return nil, fmt.Errorf("Unknown DXGI format: %d", n)
		}
		d.Fmt, d.FmtDesc = f, f.Name
	} else {
		f, desc, err := fmtFromLegacy(pf)
		if err != nil {
			return nil, err
		}
		d.Fmt, d.FmtDesc = f, desc
	}
	t := &d.Tex
	t.Width = maxi(1, int(width))
	t.Height = maxi(1, int(height))
	t.MipLevels = maxi(1, int(h.MipMapCount))
	t.ArraySize = 1
	t.Faces = []int{0}
	t.Dim = DIM_2D
	t.Depth = 1
	if d.DX10 != nil {
		t.Dim = d.DX10.ResourceDimension
		t.ArraySize = maxi(1, int(d.DX10.ArraySize))
		if t.Dim == DIM_3D {
			t.Volume = true
			t.Depth = maxi(1, int(depth))
			t.ArraySize = 1
		} else {
			if t.Dim == DIM_1D {
				t.Height = 1
			}
			if d.DX10.MiscFlag&MISC_TEXTURECUBE != 0 {
				t.Faces = []int{0, 1, 2, 3, 4, 5}
			}
		}
	} else {
		if h.Caps2&DDSCAPS2_VOLUME != 0 {
			t.Volume = true
			t.Dim = DIM_3D
			t.Depth = maxi(1, int(depth))
		} else if h.Caps2&DDSCAPS2_CUBEMAP != 0 {
			t.Faces = nil
			for i, fl := range CubeFaceFlags {
				if h.Caps2&fl != 0 {
					t.Faces = append(t.Faces, i)
				}
			}
			if len(t.Faces) == 0 {
				t.Faces = []int{0, 1, 2, 3, 4, 5}
			}
		}
	}
	if t.Width > 1<<16 || t.Height > 1<<16 || t.Depth > 1<<16 || t.ArraySize > 1<<16 || t.MipLevels > 32 {
		return nil, fmt.Errorf("Invalid DDS header (size %dx%dx%d, array %d, mipmaps %d)",
			t.Width, t.Height, t.Depth, t.ArraySize, t.MipLevels)
	}
	t.Cubemap = len(t.Faces) > 1 || (d.DX10 != nil && d.DX10.MiscFlag&MISC_TEXTURECUBE != 0) ||
		h.Caps2&DDSCAPS2_CUBEMAP != 0
	d.Data = data[offset:]
	return d, nil
}

func mipDims(w, h, d, level int) (int, int, int) {
	return maxi(1, w>>uint(level)), maxi(1, h>>uint(level)), maxi(1, d>>uint(level))
}

func fullMipCount(w, h, d int) int {
	m := maxi(w, maxi(h, d))
	return int(math.Floor(math.Log2(float64(m)))) + 1
}

// iterateSurfaces lists all surfaces in file order and returns the expected data size.
func iterateSurfaces(d *DDSFile) ([]Surface, int) {
	f, t := d.Fmt, d.Tex
	var out []Surface
	off := 0
	if t.Volume {
		for m := 0; m < t.MipLevels; m++ {
			w, h, dd := mipDims(t.Width, t.Height, t.Depth, m)
			sz := f.SurfaceSize(w, h)
			for z := 0; z < dd; z++ {
				out = append(out, Surface{0, 0, m, z, off, w, h})
				off += sz
			}
		}
	} else {
		for a := 0; a < t.ArraySize; a++ {
			for _, face := range t.Faces {
				for m := 0; m < t.MipLevels; m++ {
					w, h, _ := mipDims(t.Width, t.Height, 1, m)
					out = append(out, Surface{a, face, m, 0, off, w, h})
					off += f.SurfaceSize(w, h)
				}
			}
		}
	}
	return out, off
}

func writeDDS(path string, meta *Meta, f *Fmt, surfaces [][]byte, w, h, d, mips, arraySize int) error {
	hd := meta.Header
	pf := hd.PixelFormat
	flags := hd.Flags | DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
	caps := hd.Caps | DDSCAPS_TEXTURE
	if mips > 1 {
		flags |= DDSD_MIPMAPCOUNT
		caps |= DDSCAPS_MIPMAP | DDSCAPS_COMPLEX
	} else {
		caps &^= DDSCAPS_MIPMAP
	}
	if meta.Volume {
		flags |= DDSD_DEPTH
	}
	var pitch uint32
	switch {
	case flags&DDSD_LINEARSIZE != 0:
		pitch = uint32(f.SurfaceSize(w, h))
	case flags&DDSD_PITCH != 0:
		pitch = uint32(f.RowPitch(w))
	default:
		pitch = hd.Pitch
	}
	mipField := uint32(mips)
	if mips == 1 && hd.MipMapCount == 0 {
		mipField = 0
	}
	depthField := hd.DepthField
	if meta.Volume {
		depthField = uint32(d)
	}
	var buf bytes.Buffer
	wr := func(vals ...uint32) {
		for _, v := range vals {
			binary.Write(&buf, binary.LittleEndian, v)
		}
	}
	buf.WriteString("DDS ")
	wr(124, flags, uint32(h), uint32(w), pitch, depthField, mipField)
	for i := 0; i < 11; i++ {
		var v uint32
		if i < len(hd.Reserved1) {
			v = hd.Reserved1[i]
		}
		wr(v)
	}
	wr(32, pf.Flags, pf.FourCC, pf.BitCount, pf.RMask, pf.GMask, pf.BMask, pf.AMask)
	wr(caps, hd.Caps2, hd.Caps3, hd.Caps4, hd.Reserved2)
	if meta.DX10 != nil {
		x := meta.DX10
		as := uint32(arraySize)
		if (arraySize == 1 && x.ArraySize == 0) || meta.Volume {
			as = x.ArraySize
		}
		wr(x.DXGIFormat, x.ResourceDimension, x.MiscFlag, as, x.MiscFlags2)
	}
	for _, s := range surfaces {
		buf.Write(s)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return err
	}
	return nil
}
