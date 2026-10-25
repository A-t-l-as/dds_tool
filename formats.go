package main

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"sort"
	"strings"
)

// Chan describes one channel of an uncompressed pixel.
// Type: unorm, snorm, uint, sint, float, uf11, uf10, x (padding).
type Chan struct {
	Name string
	Off  int
	Bits int
	Type string
}

// VChan is a logical (value) channel without padding.
type VChan struct {
	Name string
	Bits int
	Type string
}

// Fmt describes a pixel format. Kind is "raw" (uncompressed) or "bc" (block compressed).
type Fmt struct {
	Name     string
	Kind     string
	ESize    int
	Channels []Chan
	SRGB     bool
	BC       int
	Signed   bool
	Special  string
}

func (f *Fmt) BlockBytes() int {
	if f.BC == 1 || f.BC == 4 {
		return 8
	}
	return 16
}

func (f *Fmt) SurfaceSize(w, h int) int {
	if f.Kind == "bc" {
		return maxi(1, (w+3)/4) * maxi(1, (h+3)/4) * f.BlockBytes()
	}
	return w * h * f.ESize
}

func (f *Fmt) RowPitch(w int) int {
	if f.Kind == "bc" {
		return maxi(1, (w+3)/4) * f.BlockBytes()
	}
	return w * f.ESize
}

func (f *Fmt) ValueChannels() []VChan {
	if f.Kind == "bc" {
		t := "unorm"
		if f.Signed {
			t = "snorm"
		}
		switch f.BC {
		case 1, 2, 3, 7:
			return []VChan{{"r", 8, "unorm"}, {"g", 8, "unorm"}, {"b", 8, "unorm"}, {"a", 8, "unorm"}}
		case 4:
			return []VChan{{"r", 8, t}}
		case 5:
			return []VChan{{"r", 8, t}, {"g", 8, t}}
		default:
			return []VChan{{"r", 16, "float"}, {"g", 16, "float"}, {"b", 16, "float"}}
		}
	}
	if f.Special == "rgb9e5" {
		return []VChan{{"r", 9, "float"}, {"g", 9, "float"}, {"b", 9, "float"}}
	}
	var out []VChan
	for _, c := range f.Channels {
		if c.Type != "x" {
			out = append(out, VChan{c.Name, c.Bits, c.Type})
		}
	}
	return out
}

// Need64 reports whether channel values need float64 precision (32-bit integers, >16-bit norms).
func (f *Fmt) Need64() bool {
	for _, c := range f.ValueChannels() {
		if (c.Type == "uint" || c.Type == "sint") && c.Bits > 24 {
			return true
		}
		if (c.Type == "unorm" || c.Type == "snorm") && c.Bits > 16 {
			return true
		}
	}
	return false
}

func seq(b int, t string, names string) []Chan {
	var out []Chan
	for i, n := range names {
		out = append(out, Chan{string(n), i * b, b, t})
	}
	return out
}

var DXGI = map[int]*Fmt{}

// UnsupportedDXGI lists known DXGI formats this tool cannot handle.
var UnsupportedDXGI = map[int]string{
	66: "R1_UNORM", 68: "R8G8_B8G8_UNORM", 69: "G8R8_G8B8_UNORM", 100: "AYUV", 101: "Y410",
	102: "Y416", 103: "NV12", 104: "P010", 105: "P016", 106: "420_OPAQUE", 107: "YUY2",
	108: "Y210", 109: "Y216", 110: "NV11", 111: "AI44", 112: "IA44", 113: "P8", 114: "A8P8",
}

var FourCCToDXGI = map[string]int{
	"DXT1": 71, "DXT2": 74, "DXT3": 74, "DXT4": 77, "DXT5": 77, "RXGB": 77,
	"ATI1": 80, "BC4U": 80, "BC4S": 81, "ATI2": 83, "BC5U": 83, "BC5S": 84,
	"BC6H": 95, "BC7L": 98, "BC7\x00": 98,
}

var D3DFmtToDXGI = map[uint32]int{36: 11, 110: 13, 111: 54, 112: 34, 113: 10, 114: 41, 115: 16, 116: 2}

func init() {
	raw := func(num int, name string, esize int, ch []Chan, srgb bool, special string) {
		DXGI[num] = &Fmt{Name: name, Kind: "raw", ESize: esize, Channels: ch, SRGB: srgb, Special: special}
	}
	bc := func(num int, name string, n int, srgb, signed bool) {
		DXGI[num] = &Fmt{Name: name, Kind: "bc", BC: n, SRGB: srgb, Signed: signed}
	}
	type nts struct {
		n    int
		t, s string
	}
	for _, e := range []nts{{1, "uint", "TYPELESS"}, {2, "float", "FLOAT"}, {3, "uint", "UINT"}, {4, "sint", "SINT"}} {
		raw(e.n, "R32G32B32A32_"+e.s, 16, seq(32, e.t, "rgba"), false, "")
	}
	for _, e := range []nts{{5, "uint", "TYPELESS"}, {6, "float", "FLOAT"}, {7, "uint", "UINT"}, {8, "sint", "SINT"}} {
		raw(e.n, "R32G32B32_"+e.s, 12, seq(32, e.t, "rgb"), false, "")
	}
	for _, e := range []nts{{9, "unorm", "TYPELESS"}, {10, "float", "FLOAT"}, {11, "unorm", "UNORM"},
		{12, "uint", "UINT"}, {13, "snorm", "SNORM"}, {14, "sint", "SINT"}} {
		raw(e.n, "R16G16B16A16_"+e.s, 8, seq(16, e.t, "rgba"), false, "")
	}
	for _, e := range []nts{{15, "uint", "TYPELESS"}, {16, "float", "FLOAT"}, {17, "uint", "UINT"}, {18, "sint", "SINT"}} {
		raw(e.n, "R32G32_"+e.s, 8, seq(32, e.t, "rg"), false, "")
	}
	raw(19, "R32G8X24_TYPELESS", 8, []Chan{{"r", 0, 32, "float"}, {"g", 32, 8, "uint"}}, false, "")
	raw(20, "D32_FLOAT_S8X24_UINT", 8, []Chan{{"r", 0, 32, "float"}, {"g", 32, 8, "uint"}}, false, "")
	raw(21, "R32_FLOAT_X8X24_TYPELESS", 8, []Chan{{"r", 0, 32, "float"}}, false, "")
	raw(22, "X32_TYPELESS_G8X24_UINT", 8, []Chan{{"g", 32, 8, "uint"}}, false, "")
	rgb10a2 := func(t string) []Chan {
		return []Chan{{"r", 0, 10, t}, {"g", 10, 10, t}, {"b", 20, 10, t}, {"a", 30, 2, t}}
	}
	raw(23, "R10G10B10A2_TYPELESS", 4, rgb10a2("unorm"), false, "")
	raw(24, "R10G10B10A2_UNORM", 4, rgb10a2("unorm"), false, "")
	raw(25, "R10G10B10A2_UINT", 4, rgb10a2("uint"), false, "")
	raw(26, "R11G11B10_FLOAT", 4, []Chan{{"r", 0, 11, "uf11"}, {"g", 11, 11, "uf11"}, {"b", 22, 10, "uf10"}}, false, "")
	type ntss struct {
		n    int
		t, s string
		srgb bool
	}
	for _, e := range []ntss{{27, "unorm", "TYPELESS", false}, {28, "unorm", "UNORM", false},
		{29, "unorm", "UNORM_SRGB", true}, {30, "uint", "UINT", false},
		{31, "snorm", "SNORM", false}, {32, "sint", "SINT", false}} {
		raw(e.n, "R8G8B8A8_"+e.s, 4, seq(8, e.t, "rgba"), e.srgb, "")
	}
	for _, e := range []nts{{33, "unorm", "TYPELESS"}, {34, "float", "FLOAT"}, {35, "unorm", "UNORM"},
		{36, "uint", "UINT"}, {37, "snorm", "SNORM"}, {38, "sint", "SINT"}} {
		raw(e.n, "R16G16_"+e.s, 4, seq(16, e.t, "rg"), false, "")
	}
	for _, e := range []nts{{39, "uint", "R32_TYPELESS"}, {40, "float", "D32_FLOAT"}, {41, "float", "R32_FLOAT"},
		{42, "uint", "R32_UINT"}, {43, "sint", "R32_SINT"}} {
		raw(e.n, e.s, 4, []Chan{{"r", 0, 32, e.t}}, false, "")
	}
	raw(44, "R24G8_TYPELESS", 4, []Chan{{"r", 0, 24, "unorm"}, {"g", 24, 8, "uint"}}, false, "")
	raw(45, "D24_UNORM_S8_UINT", 4, []Chan{{"r", 0, 24, "unorm"}, {"g", 24, 8, "uint"}}, false, "")
	raw(46, "R24_UNORM_X8_TYPELESS", 4, []Chan{{"r", 0, 24, "unorm"}}, false, "")
	raw(47, "X24_TYPELESS_G8_UINT", 4, []Chan{{"g", 24, 8, "uint"}}, false, "")
	for _, e := range []nts{{48, "unorm", "TYPELESS"}, {49, "unorm", "UNORM"}, {50, "uint", "UINT"},
		{51, "snorm", "SNORM"}, {52, "sint", "SINT"}} {
		raw(e.n, "R8G8_"+e.s, 2, seq(8, e.t, "rg"), false, "")
	}
	for _, e := range []nts{{53, "unorm", "R16_TYPELESS"}, {54, "float", "R16_FLOAT"}, {55, "unorm", "D16_UNORM"},
		{56, "unorm", "R16_UNORM"}, {57, "uint", "R16_UINT"}, {58, "snorm", "R16_SNORM"}, {59, "sint", "R16_SINT"}} {
		raw(e.n, e.s, 2, []Chan{{"r", 0, 16, e.t}}, false, "")
	}
	for _, e := range []nts{{60, "unorm", "TYPELESS"}, {61, "unorm", "UNORM"}, {62, "uint", "UINT"},
		{63, "snorm", "SNORM"}, {64, "sint", "SINT"}} {
		raw(e.n, "R8_"+e.s, 1, []Chan{{"r", 0, 8, e.t}}, false, "")
	}
	raw(65, "A8_UNORM", 1, []Chan{{"a", 0, 8, "unorm"}}, false, "")
	raw(67, "R9G9B9E5_SHAREDEXP", 4, nil, false, "rgb9e5")
	bc(70, "BC1_TYPELESS", 1, false, false)
	bc(71, "BC1_UNORM", 1, false, false)
	bc(72, "BC1_UNORM_SRGB", 1, true, false)
	bc(73, "BC2_TYPELESS", 2, false, false)
	bc(74, "BC2_UNORM", 2, false, false)
	bc(75, "BC2_UNORM_SRGB", 2, true, false)
	bc(76, "BC3_TYPELESS", 3, false, false)
	bc(77, "BC3_UNORM", 3, false, false)
	bc(78, "BC3_UNORM_SRGB", 3, true, false)
	bc(79, "BC4_TYPELESS", 4, false, false)
	bc(80, "BC4_UNORM", 4, false, false)
	bc(81, "BC4_SNORM", 4, false, true)
	bc(82, "BC5_TYPELESS", 5, false, false)
	bc(83, "BC5_UNORM", 5, false, false)
	bc(84, "BC5_SNORM", 5, false, true)
	raw(85, "B5G6R5_UNORM", 2, []Chan{{"b", 0, 5, "unorm"}, {"g", 5, 6, "unorm"}, {"r", 11, 5, "unorm"}}, false, "")
	raw(86, "B5G5R5A1_UNORM", 2, []Chan{{"b", 0, 5, "unorm"}, {"g", 5, 5, "unorm"}, {"r", 10, 5, "unorm"},
		{"a", 15, 1, "unorm"}}, false, "")
	bgra := []Chan{{"b", 0, 8, "unorm"}, {"g", 8, 8, "unorm"}, {"r", 16, 8, "unorm"}, {"a", 24, 8, "unorm"}}
	bgrx := []Chan{{"b", 0, 8, "unorm"}, {"g", 8, 8, "unorm"}, {"r", 16, 8, "unorm"}, {"x", 24, 8, "x"}}
	raw(87, "B8G8R8A8_UNORM", 4, bgra, false, "")
	raw(88, "B8G8R8X8_UNORM", 4, bgrx, false, "")
	raw(89, "R10G10B10_XR_BIAS_A2_UNORM", 4, rgb10a2("unorm"), false, "")
	raw(90, "B8G8R8A8_TYPELESS", 4, bgra, false, "")
	raw(91, "B8G8R8A8_UNORM_SRGB", 4, bgra, true, "")
	raw(92, "B8G8R8X8_TYPELESS", 4, bgrx, false, "")
	raw(93, "B8G8R8X8_UNORM_SRGB", 4, bgrx, true, "")
	bc(94, "BC6H_TYPELESS", 6, false, false)
	bc(95, "BC6H_UF16", 6, false, false)
	bc(96, "BC6H_SF16", 6, false, true)
	bc(97, "BC7_TYPELESS", 7, false, false)
	bc(98, "BC7_UNORM", 7, false, false)
	bc(99, "BC7_UNORM_SRGB", 7, true, false)
	raw(115, "B4G4R4A4_UNORM", 2, []Chan{{"b", 0, 4, "unorm"}, {"g", 4, 4, "unorm"}, {"r", 8, 4, "unorm"},
		{"a", 12, 4, "unorm"}}, false, "")
	raw(191, "A4B4G4R4_UNORM", 2, []Chan{{"a", 0, 4, "unorm"}, {"b", 4, 4, "unorm"}, {"g", 8, 4, "unorm"},
		{"r", 12, 4, "unorm"}}, false, "")
}

func sortedDXGI() []int {
	var keys []int
	for k := range DXGI {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	return keys
}

func maskInfo(m uint32) (off, nbits int, err error) {
	if m == 0 {
		return 0, 0, nil
	}
	off = bits.TrailingZeros32(m)
	nbits = bits.OnesCount32(m)
	if uint64(m>>uint(off)) != (uint64(1)<<uint(nbits))-1 {
		return 0, 0, fmt.Errorf("Non-contiguous bit mask 0x%08X is not supported", m)
	}
	return off, nbits, nil
}

func fourccString(v uint32) string {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	return string(b[:])
}

// fmtFromLegacy builds a format from a legacy (non-DX10) DDS pixel format.
func fmtFromLegacy(pf PixelFormat) (*Fmt, string, error) {
	flags := pf.Flags
	if flags&DDPF_FOURCC != 0 {
		fcc := fourccString(pf.FourCC)
		if n, ok := FourCCToDXGI[fcc]; ok {
			return DXGI[n], strings.TrimRight(fcc, "\x00"), nil
		}
		if n, ok := D3DFmtToDXGI[pf.FourCC]; ok {
			f := DXGI[n]
			return f, fmt.Sprintf("D3DFMT_%d (%s)", pf.FourCC, f.Name), nil
		}
		return nil, "", fmt.Errorf("Unsupported FourCC: %q (%d)", fcc, pf.FourCC)
	}
	if flags&(DDPF_YUV|DDPF_PAL8) != 0 {
		return nil, "", fmt.Errorf("YUV / palettized formats are not supported")
	}
	bpp := int(pf.BitCount)
	if bpp != 8 && bpp != 16 && bpp != 24 && bpp != 32 {
		return nil, "", fmt.Errorf("Unsupported bits per pixel: %d", bpp)
	}
	masks := [4]uint32{pf.RMask, pf.GMask, pf.BMask, pf.AMask}
	var types [4]string
	var names [4]string
	switch {
	case flags&DDPF_BUMPDUDV != 0:
		at := "unorm"
		if _, ab, _ := maskInfo(masks[3]); bpp == 32 && ab == 8 {
			at = "snorm"
		}
		types = [4]string{"snorm", "snorm", "snorm", at}
		names = [4]string{"r", "g", "b", "a"}
	case flags&DDPF_BUMPLUMINANCE != 0:
		types = [4]string{"snorm", "snorm", "unorm", "unorm"}
		names = [4]string{"r", "g", "b", "a"}
	case flags&DDPF_LUMINANCE != 0:
		types = [4]string{"unorm", "unorm", "unorm", "unorm"}
		names = [4]string{"l", "", "", "a"}
	case flags&DDPF_ALPHA != 0 && flags&DDPF_RGB == 0:
		types = [4]string{"unorm", "unorm", "unorm", "unorm"}
		names = [4]string{"", "", "", "a"}
	default:
		types = [4]string{"unorm", "unorm", "unorm", "unorm"}
		names = [4]string{"r", "g", "b", "a"}
	}
	if flags&(DDPF_ALPHAPIXELS|DDPF_ALPHA) == 0 && flags&DDPF_BUMPDUDV == 0 {
		names[3] = ""
	}
	full := uint32((uint64(1) << uint(bpp)) - 1)
	// some tools write masks shifted outside the pixel bit range
	for i, m := range masks {
		if m&^full != 0 && bpp < 32 {
			masks[i] = m >> uint(32-bpp)
		}
	}
	var chans []Chan
	var used uint32
	for i := 0; i < 4; i++ {
		if names[i] == "" || masks[i] == 0 {
			continue
		}
		off, nb, err := maskInfo(masks[i])
		if err != nil {
			return nil, "", err
		}
		chans = append(chans, Chan{names[i], off, nb, types[i]})
		used |= masks[i]
	}
	if len(chans) == 0 {
		return nil, "", fmt.Errorf("Pixel format has no channels")
	}
	// unused bits in the alpha position (e.g. X8R8G8B8) are filled with ones
	free := full &^ used
	hasA := false
	for _, c := range chans {
		if c.Name == "a" {
			hasA = true
		}
	}
	if free != 0 && !hasA {
		off := bits.TrailingZeros32(free)
		nb := bits.OnesCount32(free)
		if uint64(free>>uint(off)) == (uint64(1)<<uint(nb))-1 {
			chans = append(chans, Chan{"x", off, nb, "x"})
		}
	}
	sorted := append([]Chan(nil), chans...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Off > sorted[j].Off })
	desc := fmt.Sprintf("legacy %dbpp ", bpp)
	for _, c := range sorted {
		desc += fmt.Sprintf("%s%d", strings.ToUpper(c.Name), c.Bits)
	}
	return &Fmt{Name: desc, Kind: "raw", ESize: bpp / 8, Channels: chans}, desc, nil
}

func maxi(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func mini(a, b int) int {
	if a < b {
		return a
	}
	return b
}
