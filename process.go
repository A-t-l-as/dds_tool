package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strings"
)

type Logger func(format string, a ...interface{})

// ---------------------------------------------------------------- image <-> channel values

// imagePlan decides the image file kind and the channel mapping.
func imagePlan(f *Fmt, bc1Alpha bool) (string, []*string) {
	vch := f.ValueChannels()
	if f.Kind == "bc" && f.BC == 1 && !bc1Alpha {
		vch = vch[:3]
	}
	has := map[string]bool{}
	for _, c := range vch {
		has[c.Name] = true
	}
	var layout []*string
	switch {
	case len(vch) == 1:
		layout = []*string{strp(vch[0].Name)}
	case len(vch) == 2 && has["l"] && has["a"]:
		layout = []*string{strp("l"), strp("a")}
	default:
		names := []string{"r", "g", "b"}
		if has["a"] {
			names = append(names, "a")
		}
		for _, n := range names {
			if has[n] {
				layout = append(layout, strp(n))
			} else {
				layout = append(layout, nil)
			}
		}
	}
	floaty, maxBits, over16 := false, 0, false
	for _, c := range vch {
		if c.Type == "float" || c.Type == "uf11" || c.Type == "uf10" ||
			((c.Type == "unorm" || c.Type == "snorm") && c.Bits > 16) {
			floaty = true
		}
		maxBits = maxi(maxBits, c.Bits)
		over16 = over16 || c.Bits > 16
	}
	if f.Special == "rgb9e5" || (f.Kind == "bc" && f.BC == 6) {
		floaty = true
	}
	switch {
	case floaty:
		return "exr_float", layout
	case over16:
		return "exr_uint", layout
	case maxBits > 8:
		return "png16", layout
	}
	return "png8", layout
}

func chanIndex(f *Fmt) map[string]int {
	m := map[string]int{}
	for k, c := range f.ValueChannels() {
		m[c.Name] = k
	}
	return m
}

// valuesToImage maps decoded channel values to the image written to disk.
func valuesToImage(vals *Img, f *Fmt, kind string, layout []*string) *Img {
	vch := f.ValueChannels()
	idx := chanIndex(f)
	if strings.HasPrefix(kind, "png") && len(layout) == vals.C && len(vch) == vals.C {
		identity := true
		for i, name := range layout {
			if name == nil || *name != vch[i].Name || vch[i].Type != "unorm" {
				identity = false
			}
		}
		if identity {
			return vals // nothing to convert (e.g. BC1-3/BC7, RGBA8)
		}
	}
	out := NewImg(vals.W, vals.H, len(layout), kind == "exr_uint")
	n := vals.W * vals.H
	for li, name := range layout {
		k, ok := -1, false
		if name != nil {
			k, ok = idx[*name]
		}
		for i := 0; i < n; i++ {
			if !ok {
				out.Set(i*out.C+li, 0)
				continue
			}
			v := vals.Get(i*vals.C + k)
			b, t := vch[k].Bits, vch[k].Type
			var p float64
			switch kind {
			case "png8", "png16":
				full := float64((int64(1) << uint(b)) - 1)
				half := float64(int64(1) << uint(b-1))
				switch t {
				case "unorm":
					p = v
				case "snorm":
					s := rint(clamp(v, -1, 1) * (half - 1))
					p = (s + half) / full
				case "uint":
					p = v / full
				default: // sint
					p = (v + half) / full
				}
			case "exr_uint":
				p = v
				if v < 0 {
					p = v + 4294967296
				}
			default:
				p = v
			}
			out.Set(i*out.C+li, p)
		}
	}
	return out
}

// imageToValues maps a loaded image to channel values in f.ValueChannels() order.
func imageToValues(li *LImage, f *Fmt, kind string, layout []*string) *Img {
	vch := f.ValueChannels()
	idx := chanIndex(f)
	src := li.Pix
	w, h := src.W, src.H
	out := NewImg(w, h, len(vch), f.Need64())
	// defaults for channels not present in the image: alpha = 1, others = 0
	for k, c := range vch {
		if c.Name == "a" {
			for i := 0; i < w*h; i++ {
				out.Set(i*out.C+k, 1)
			}
		}
	}
	var planes []int
	switch len(layout) {
	case 1:
		planes = []int{0}
	case 2:
		planes = []int{0, 3}
	case 3:
		planes = []int{0, 1, 2}
	default:
		planes = []int{0, 1, 2, 3}
	}
	normalize := li.Scale > 1 && kind != "exr_uint"
	for pi, name := range layout {
		if name == nil {
			continue
		}
		k, ok := idx[*name]
		if !ok {
			continue
		}
		b, t := vch[k].Bits, vch[k].Type
		full := float64((int64(1) << uint(b)) - 1)
		half := float64(int64(1) << uint(b-1))
		sc := planes[pi]
		for i := 0; i < w*h; i++ {
			p := src.Get(i*4 + sc)
			if normalize {
				p /= li.Scale
			}
			var v float64
			switch kind {
			case "png8", "png16":
				p = clamp(p, 0, 1)
				switch t {
				case "unorm":
					v = p
				case "snorm":
					s := rint(p*full) - half
					v = math.Max(s/(half-1), -1)
				case "uint":
					v = rint(p * full)
				default:
					v = rint(p*full) - half
				}
			case "exr_uint":
				v = p
				if t == "sint" && v >= half {
					v -= 2 * half
				}
			default:
				v = p
			}
			out.Set(i*out.C+k, v)
		}
	}
	return out
}

// ---------------------------------------------------------------- mipmaps

func srgbToLin(c float64) float64 {
	c = clamp(c, 0, 1)
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

func linToSrgb(c float64) float64 {
	c = clamp(c, 0, 1)
	if c <= 0.0031308 {
		return c * 12.92
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

type tap struct {
	i int
	w float64
}

// downWeights returns, for each output sample, the input samples and weights (box filter).
func downWeights(n int) [][]tap {
	m := n / 2
	out := make([][]tap, m)
	if n%2 == 0 {
		for j := 0; j < m; j++ {
			out[j] = []tap{{2 * j, 0.5}, {2*j + 1, 0.5}}
		}
		return out
	}
	s := float64(n) / float64(m)
	for j := 0; j < m; j++ {
		x0, x1 := float64(j)*s, float64(j+1)*s
		for i := int(math.Floor(x0)); i < mini(n, int(math.Ceil(x1))); i++ {
			wt := (math.Min(x1, float64(i+1)) - math.Max(x0, float64(i))) / s
			if wt > 0 {
				out[j] = append(out[j], tap{i, wt})
			}
		}
	}
	return out
}

func downX(m *Img) *Img {
	if m.W == 1 {
		return m
	}
	wts := downWeights(m.W)
	out := NewImg(len(wts), m.H, m.C, m.Wide())
	C := m.C
	parallelRows(m.H, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for j, taps := range wts {
				for c := 0; c < C; c++ {
					s := 0.0
					for _, t := range taps {
						s += m.Get((y*m.W+t.i)*C+c) * t.w
					}
					out.Set((y*out.W+j)*C+c, s)
				}
			}
		}
	})
	return out
}

func downY(m *Img) *Img {
	if m.H == 1 {
		return m
	}
	wts := downWeights(m.H)
	out := NewImg(m.W, len(wts), m.C, m.Wide())
	rowLen := m.W * m.C
	parallelRows(len(wts), func(j0, j1 int) {
		for j := j0; j < j1; j++ {
			for x := 0; x < rowLen; x++ {
				s := 0.0
				for _, t := range wts[j] {
					s += m.Get(t.i*rowLen+x) * t.w
				}
				out.Set(j*rowLen+x, s)
			}
		}
	})
	return out
}

func downZ(sl []*Img) []*Img {
	if len(sl) == 1 {
		return sl
	}
	wts := downWeights(len(sl))
	out := make([]*Img, len(wts))
	for j, taps := range wts {
		o := NewImg(sl[0].W, sl[0].H, sl[0].C, sl[0].Wide())
		for i := 0; i < o.Len(); i++ {
			s := 0.0
			for _, t := range taps {
				s += sl[t.i].Get(i) * t.w
			}
			o.Set(i, s)
		}
		out[j] = o
	}
	return out
}

func applySRGB(sl []*Img, idx []int, fn func(float64) float64) []*Img {
	out := make([]*Img, len(sl))
	for k, m := range sl {
		c := m.Clone()
		n := c.W * c.H
		for i := 0; i < n; i++ {
			for _, ch := range idx {
				c.Set(i*c.C+ch, fn(c.Get(i*c.C+ch)))
			}
		}
		out[k] = c
	}
	return out
}

// buildMips calls emit for every level, starting with the base level (slices > 1 only for volumes).
func buildMips(base []*Img, levels int, srgbIdx []int, volume bool, emit func(level int, slices []*Img) error) error {
	if err := emit(0, base); err != nil {
		return err
	}
	if levels <= 1 {
		return nil
	}
	cur := base
	if len(srgbIdx) > 0 {
		// level 0 is already encoded, so the base can be linearized in place
		for _, m := range cur {
			n := m.W * m.H
			for i := 0; i < n; i++ {
				for _, ch := range srgbIdx {
					m.Set(i*m.C+ch, srgbToLin(m.Get(i*m.C+ch)))
				}
			}
		}
	}
	for l := 1; l < levels; l++ {
		next := make([]*Img, len(cur))
		for k, m := range cur {
			next[k] = downX(downY(m))
		}
		if volume {
			next = downZ(next)
		}
		cur = next
		lvl := cur
		if len(srgbIdx) > 0 {
			lvl = applySRGB(cur, srgbIdx, linToSrgb)
		}
		if err := emit(l, lvl); err != nil {
			return err
		}
	}
	return nil
}

// ---------------------------------------------------------------- export

func decodeSurface(d *DDSFile, s Surface) (*Img, error) {
	f := d.Fmt
	sz := f.SurfaceSize(s.W, s.H)
	if s.Offset+sz > len(d.Data) {
		return nil, fmt.Errorf("The DDS file is truncated (texture data missing)")
	}
	buf := d.Data[s.Offset : s.Offset+sz]
	if f.Kind == "bc" {
		return bcDecode(buf, f, s.W, s.H)
	}
	return rawDecode(buf, f, s.W, s.H, f.Need64())
}

func exportDDS(path, outdir string, log Logger) (string, error) {
	d, err := readDDS(path)
	if err != nil {
		return "", err
	}
	f, t := d.Fmt, d.Tex
	surfaces, total := iterateSurfaces(d)
	if len(d.Data) < total {
		log("  warning: the file has less data than the header implies (%d < %d B) - using mip 0 only",
			len(d.Data), total)
	}
	var top []Surface
	for _, s := range surfaces {
		if s.Mip == 0 {
			top = append(top, s)
		}
	}
	var bc1Alpha *bool
	if f.Kind == "bc" && f.BC == 1 {
		found := false
		for _, s := range top {
			img, err := decodeSurface(d, s)
			if err != nil {
				return "", err
			}
			for i := 3; i < len(img.F32); i += 4 {
				if img.F32[i] < 0.5 {
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		bc1Alpha = &found
	}
	kind, layout := imagePlan(f, bc1Alpha == nil || *bc1Alpha)
	ext := ".png"
	if strings.HasPrefix(kind, "exr") {
		ext = ".exr"
	}
	stem := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	if outdir == "" {
		abs, _ := filepath.Abs(path)
		outdir = filepath.Dir(abs)
	}
	if err := os.MkdirAll(outdir, 0755); err != nil {
		return "", err
	}
	var images []ImageEntry
	for _, s := range top {
		name := stem
		if t.ArraySize > 1 {
			name += fmt.Sprintf("_a%d", s.Array)
		}
		if t.Cubemap {
			name += "_" + CubeFaceNames[s.Face]
		}
		if t.Volume && t.Depth > 1 {
			name += fmt.Sprintf("_z%03d", s.Slice)
		}
		name += ext
		img, err := decodeSurface(d, s)
		if err != nil {
			return "", err
		}
		if err := saveImage(filepath.Join(outdir, name), valuesToImage(img, f, kind, layout), kind); err != nil {
			return "", err
		}
		e := ImageEntry{File: name, ArrayIndex: s.Array}
		if t.Cubemap {
			e.Face = strp(CubeFaceNames[s.Face])
		}
		if t.Volume {
			e.Slice = intp(s.Slice)
		}
		images = append(images, e)
	}
	container := "legacy"
	if d.DX10 != nil {
		container = "DX10"
	}
	dim, ok := DimNames[t.Dim]
	if !ok {
		dim = "2D"
	}
	var faces []string
	if t.Cubemap {
		for _, fi := range t.Faces {
			faces = append(faces, CubeFaceNames[fi])
		}
	}
	m := &Meta{
		Tool: "dds_tool", ToolVersion: ToolVersion, Source: filepath.Base(path), Format: d.FmtDesc,
		Container: container, Width: t.Width, Height: t.Height, Depth: t.Depth, MipLevels: t.MipLevels,
		FullMipChain: t.MipLevels == fullMipCount(t.Width, t.Height, t.Depth), Dimension: dim,
		ArraySize: t.ArraySize, Cubemap: t.Cubemap, CubeFaces: faces, Volume: t.Volume, SRGB: f.SRGB,
		BC1Alpha: bc1Alpha, ImageKind: kind, ImageLayout: layout, Images: images, Header: d.Header, DX10: d.DX10,
	}
	jpath := filepath.Join(outdir, stem+".json")
	if err := saveMeta(jpath, m); err != nil {
		return "", err
	}
	var names []string
	for i, e := range images {
		if i == 3 {
			names = append(names, "...")
			break
		}
		names = append(names, e.File)
	}
	log("  %s -> %s (%s, %dx%d, mip: %d) + %s", filepath.Base(path), strings.Join(names, ", "),
		d.FmtDesc, t.Width, t.Height, t.MipLevels, filepath.Base(jpath))
	return jpath, nil
}

// ---------------------------------------------------------------- import

func fmtFromMeta(m *Meta) (*Fmt, error) {
	if m.DX10 != nil {
		f, ok := DXGI[int(m.DX10.DXGIFormat)]
		if !ok {
			return nil, fmt.Errorf("Unsupported DXGI format %d", m.DX10.DXGIFormat)
		}
		return f, nil
	}
	f, _, err := fmtFromLegacy(m.Header.PixelFormat)
	return f, err
}

func resolveImagePath(base, file string) string {
	p := filepath.FromSlash(strings.ReplaceAll(file, "\\", "/"))
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(base, p)
}

func importJSON(jpath, out string, mips *int, opts EncOpts, log Logger) (string, error) {
	m, err := loadMeta(jpath)
	if err != nil {
		return "", err
	}
	abs, _ := filepath.Abs(jpath)
	stem := strings.TrimSuffix(filepath.Base(jpath), filepath.Ext(jpath))
	return importMeta(m, filepath.Dir(abs), stem+".dds", out, mips, opts, log, filepath.Base(jpath))
}

func importMeta(m *Meta, base, ddsName, out string, mips *int, opts EncOpts, log Logger, label string) (string, error) {
	f, err := fmtFromMeta(m)
	if err != nil {
		return "", err
	}
	kind, layout := m.ImageKind, m.ImageLayout
	switch kind {
	case "png8", "png16", "exr_float", "exr_uint":
	default:
		return "", fmt.Errorf("Unknown image_kind %q", kind)
	}
	if f.Kind == "bc" && f.BC == 1 {
		opts.BC1Alpha = m.BC1Alpha != nil && *m.BC1Alpha
	}
	if len(m.Images) == 0 {
		return "", fmt.Errorf("No images are listed in the metadata")
	}
	vch := f.ValueChannels()
	var srgbIdx []int
	if f.SRGB {
		for k, c := range vch {
			if c.Name == "r" || c.Name == "g" || c.Name == "b" {
				srgbIdx = append(srgbIdx, k)
			}
		}
	}
	loadVals := func(e ImageEntry) (*Img, error) {
		p := resolveImagePath(base, e.File)
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("Image file not found: %s", p)
		}
		li, err := loadImage(p)
		if err != nil {
			return nil, err
		}
		return imageToValues(li, f, kind, layout), nil
	}
	var groups [][]ImageEntry
	if m.Volume {
		imgs := append([]ImageEntry(nil), m.Images...)
		sort.SliceStable(imgs, func(i, j int) bool {
			a, b := 0, 0
			if imgs[i].Slice != nil {
				a = *imgs[i].Slice
			}
			if imgs[j].Slice != nil {
				b = *imgs[j].Slice
			}
			return a < b
		})
		groups = [][]ImageEntry{imgs}
	} else {
		for _, e := range m.Images {
			groups = append(groups, []ImageEntry{e})
		}
	}
	encode := func(img *Img) []byte {
		if f.Kind == "bc" {
			return bcEncode(img, f, opts)
		}
		return rawEncode(img, f)
	}
	var surfaces [][]byte
	w, h, d, levels := 0, 0, 0, 0
	for gi, g := range groups {
		var slices []*Img
		for _, e := range g {
			img, err := loadVals(e)
			if err != nil {
				return "", err
			}
			debug.FreeOSMemory() // release the decoded source image before building mipmaps
			if len(slices) > 0 && (img.W != slices[0].W || img.H != slices[0].H) {
				return "", fmt.Errorf("All images (faces/layers/slices) must have the same size")
			}
			slices = append(slices, img)
		}
		if gi == 0 {
			w, h, d = slices[0].W, slices[0].H, len(slices)
			if w > 1<<16 || h > 1<<16 {
				return "", fmt.Errorf("Image is too large (%dx%d)", w, h)
			}
			if m.Dimension == "1D" && h != 1 {
				return "", fmt.Errorf("A 1D texture must be 1 pixel high")
			}
			dd := 1
			if m.Volume {
				dd = d
			}
			maxm := fullMipCount(w, h, dd)
			switch {
			case mips != nil && *mips <= 0:
				levels = maxm
			case mips != nil:
				levels = mini(*mips, maxm)
			case m.FullMipChain:
				levels = maxm
			default:
				levels = mini(maxi(1, m.MipLevels), maxm)
			}
		} else if slices[0].W != w || slices[0].H != h {
			return "", fmt.Errorf("All images (faces/layers) must have the same size")
		}
		err := buildMips(slices, levels, srgbIdx, m.Volume, func(level int, sl []*Img) error {
			for _, s := range sl {
				surfaces = append(surfaces, encode(s))
			}
			return nil
		})
		if err != nil {
			return "", err
		}
	}
	arraySize := 1
	if !m.Volume {
		nfaces := 1
		if m.Cubemap {
			nfaces = maxi(1, len(m.CubeFaces))
		}
		arraySize = maxi(1, len(groups)/nfaces)
	}
	if out == "" {
		out = filepath.Join(base, ddsName)
	} else if st, err := os.Stat(out); (err == nil && st.IsDir()) || !strings.HasSuffix(strings.ToLower(out), ".dds") {
		if err := os.MkdirAll(out, 0755); err != nil {
			return "", err
		}
		out = filepath.Join(out, ddsName)
	} else if dir := filepath.Dir(out); dir != "" {
		os.MkdirAll(dir, 0755)
	}
	depth := 1
	if m.Volume {
		depth = d
	}
	if err := writeDDS(out, m, f, surfaces, w, h, depth, levels, arraySize); err != nil {
		return "", err
	}
	log("  %s -> %s (%s, %dx%d, mip: %d)", label, filepath.Base(out), m.Format, w, h, levels)
	return out, nil
}

// ---------------------------------------------------------------- create

var legacyAliases = map[string]int{"DXT1": 71, "DXT2": 74, "DXT3": 74, "DXT4": 77, "DXT5": 77,
	"ATI1": 80, "ATI2": 83, "BC4U": 80, "BC4S": 81, "BC5U": 83, "BC5S": 84}
var legacyAliasOrder = []string{"DXT1", "DXT2", "DXT3", "DXT4", "DXT5", "ATI1", "ATI2", "BC4U", "BC4S", "BC5U", "BC5S"}

func findFormat(name string) (string, int, *Fmt, error) {
	key := strings.TrimPrefix(strings.ToUpper(strings.TrimSpace(name)), "DXGI_FORMAT_")
	if n, ok := legacyAliases[key]; ok {
		return key, n, DXGI[n], nil
	}
	for _, n := range sortedDXGI() {
		if DXGI[n].Name == key {
			return "", n, DXGI[n], nil
		}
	}
	return "", 0, nil, fmt.Errorf("Unknown format %q. List: dds_tool formats", name)
}

func newMeta(fmtName string) (*Fmt, *Meta, error) {
	alias, num, f, err := findFormat(fmtName)
	if err != nil {
		return nil, nil, err
	}
	flags := uint32(DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT)
	if f.Kind == "bc" {
		flags |= DDSD_LINEARSIZE
	} else {
		flags |= DDSD_PITCH
	}
	m := &Meta{Tool: "dds_tool", ToolVersion: ToolVersion, Dimension: "2D", ArraySize: 1,
		FullMipChain: true, MipLevels: 1, SRGB: f.SRGB}
	pf := PixelFormat{Size: 32, Flags: DDPF_FOURCC}
	if alias != "" { // legacy FourCC header (best compatibility with older games)
		pf.FourCC = binary.LittleEndian.Uint32([]byte(alias))
		m.Format, m.Container = alias, "legacy"
	} else {
		pf.FourCC = fourccDX10
		m.DX10 = &DX10{DXGIFormat: uint32(num), ResourceDimension: DIM_2D, ArraySize: 1}
		m.Format, m.Container = f.Name, "DX10"
	}
	m.Header = Header{Flags: flags, Reserved1: make([]uint32, 11), PixelFormat: pf, Caps: DDSCAPS_TEXTURE, MipMapCount: 1}
	return f, m, nil
}

func createDDS(images []string, fmtName, like, out string, mips *int, cube, writeJSON bool, opts EncOpts, log Logger) (string, error) {
	var paths []string
	for _, p := range images {
		a, _ := filepath.Abs(p)
		if _, err := os.Stat(a); err != nil {
			return "", fmt.Errorf("File not found: %s", a)
		}
		paths = append(paths, a)
	}
	var f *Fmt
	var m *Meta
	var err error
	switch {
	case like != "":
		if strings.HasSuffix(strings.ToLower(like), ".dds") {
			src, err := readDDS(like)
			if err != nil {
				return "", err
			}
			f = src.Fmt
			t := src.Tex
			m = &Meta{Tool: "dds_tool", ToolVersion: ToolVersion, Format: src.FmtDesc, Header: src.Header,
				MipLevels: t.MipLevels, FullMipChain: t.MipLevels == fullMipCount(t.Width, t.Height, t.Depth)}
			m.Container = "legacy"
			if src.DX10 != nil {
				x := *src.DX10
				m.DX10, m.Container = &x, "DX10"
			}
		} else {
			if m, err = loadMeta(like); err != nil {
				return "", err
			}
			if f, err = fmtFromMeta(m); err != nil {
				return "", err
			}
		}
		m.Dimension, m.Volume, m.Cubemap, m.CubeFaces, m.ArraySize, m.SRGB = "2D", false, false, nil, 1, f.SRGB
		m.Header.Caps2 = 0
		if m.DX10 != nil {
			m.DX10.ResourceDimension, m.DX10.MiscFlag, m.DX10.ArraySize = DIM_2D, 0, 1
		}
	case fmtName != "":
		if f, m, err = newMeta(fmtName); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("Specify a format (-f) or a template (--like)")
	}
	if cube {
		if len(paths)%6 != 0 {
			return "", fmt.Errorf("A cubemap needs 6 images (or a multiple of 6) in the order +X -X +Y -Y +Z -Z")
		}
		if m.DX10 == nil && len(paths) > 6 {
			return "", fmt.Errorf("Cubemap arrays require a DXGI format name (e.g. BC1_UNORM instead of DXT1, BC3_UNORM instead of DXT5)")
		}
		m.Cubemap = true
		m.CubeFaces = append([]string(nil), CubeFaceNames...)
		m.Header.Caps2 = DDSCAPS2_CUBEMAP
		for _, fl := range CubeFaceFlags {
			m.Header.Caps2 |= fl
		}
		m.Header.Caps |= DDSCAPS_COMPLEX
		if m.DX10 != nil {
			m.DX10.MiscFlag |= MISC_TEXTURECUBE
		}
	} else if len(paths) > 1 && m.DX10 == nil {
		return "", fmt.Errorf("Texture arrays require a DXGI format name (e.g. BC1_UNORM instead of DXT1, BC3_UNORM instead of DXT5)")
	}
	m.BC1Alpha = nil
	if f.Kind == "bc" && f.BC == 1 {
		found := false
		for _, p := range paths {
			li, err := loadImage(p)
			if err != nil {
				return "", err
			}
			n := li.Pix.W * li.Pix.H
			for i := 0; i < n && !found; i++ {
				found = li.Pix.Get(4*i+3)/li.Scale < 0.5
			}
		}
		m.BC1Alpha = &found
	}
	m.ImageKind, m.ImageLayout = imagePlan(f, m.BC1Alpha == nil || *m.BC1Alpha)
	m.Images = nil
	for i, p := range paths {
		e := ImageEntry{File: p}
		if cube {
			e.ArrayIndex = i / 6
			e.Face = strp(CubeFaceNames[i%6])
		} else {
			e.ArrayIndex = i
		}
		m.Images = append(m.Images, e)
	}
	if mips == nil && !m.FullMipChain {
		mips = intp(maxi(1, m.MipLevels))
	}
	stem := strings.TrimSuffix(filepath.Base(paths[0]), filepath.Ext(paths[0]))
	label := filepath.Base(paths[0])
	if len(paths) > 1 {
		label += fmt.Sprintf(" (+%d)", len(paths)-1)
	}
	res, err := importMeta(m, filepath.Dir(paths[0]), stem+".dds", out, mips, opts, log, label)
	if err != nil {
		return "", err
	}
	if writeJSON {
		resAbs, _ := filepath.Abs(res)
		jm := *m
		jm.Source = filepath.Base(res)
		jm.Images = nil
		for _, e := range m.Images {
			rel, err := filepath.Rel(filepath.Dir(resAbs), e.File)
			if err != nil {
				rel = e.File
			}
			e.File = filepath.ToSlash(rel)
			jm.Images = append(jm.Images, e)
		}
		d, err := readDDS(res)
		if err != nil {
			return "", err
		}
		jm.Width, jm.Height, jm.Depth = d.Tex.Width, d.Tex.Height, 1
		jm.MipLevels, jm.ArraySize, jm.Header, jm.DX10 = d.Tex.MipLevels, d.Tex.ArraySize, d.Header, d.DX10
		if mips != nil {
			jm.FullMipChain = *mips <= 0 || d.Tex.MipLevels == fullMipCount(d.Tex.Width, d.Tex.Height, 1)
		}
		if err := saveMeta(strings.TrimSuffix(res, filepath.Ext(res))+".json", &jm); err != nil {
			return "", err
		}
	}
	return res, nil
}

// ---------------------------------------------------------------- info / formats

func listFormats(log Logger) {
	log("Legacy headers (FourCC): %s", strings.Join(legacyAliasOrder, ", "))
	log("DXGI formats:")
	var names []string
	for _, n := range sortedDXGI() {
		if !strings.Contains(DXGI[n].Name, "TYPELESS") {
			names = append(names, DXGI[n].Name)
		}
	}
	for i := 0; i < len(names); i += 3 {
		line := "  "
		for _, n := range names[i:mini(i+3, len(names))] {
			line += fmt.Sprintf("%-30s", n)
		}
		log("%s", strings.TrimRight(line, " "))
	}
	log("Typical: color -> BC7_UNORM_SRGB or BC1_UNORM_SRGB / BC3_UNORM_SRGB, normal map -> BC5_UNORM,")
	log("         single-channel mask -> BC4_UNORM, HDR -> BC6H_UF16, uncompressed -> R8G8B8A8_UNORM(_SRGB)")
}

func infoDDS(path string, log Logger) error {
	d, err := readDDS(path)
	if err != nil {
		return err
	}
	t := d.Tex
	_, total := iterateSurfaces(d)
	container := "legacy"
	if d.DX10 != nil {
		container = "DX10"
	}
	log("%s", path)
	log("  format:     %s (%s)", d.FmtDesc, container)
	size := fmt.Sprintf("%d x %d", t.Width, t.Height)
	if t.Volume {
		size += fmt.Sprintf(" x %d", t.Depth)
	}
	log("  size:       %s", size)
	dim, ok := DimNames[t.Dim]
	if !ok {
		dim = "2D"
	}
	if t.Cubemap {
		dim += ", cubemap"
	}
	if t.ArraySize > 1 {
		dim += fmt.Sprintf(", array x%d", t.ArraySize)
	}
	log("  type:       %s", dim)
	full := ""
	if t.MipLevels == fullMipCount(t.Width, t.Height, t.Depth) {
		full = " (full chain)"
	}
	log("  mipmaps:    %d%s", t.MipLevels, full)
	srgb := "no"
	if d.Fmt.SRGB {
		srgb = "yes"
	}
	log("  sRGB:       %s", srgb)
	log("  data:       %d B (expected %d B)", len(d.Data), total)
	return nil
}
