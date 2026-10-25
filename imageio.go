package main

import (
	"bufio"
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// LImage is a loaded image normalized to 4 channels (R, G, B, A).
// Integer formats keep their raw values; Scale is the maximum integer value (1 for float data).
type LImage struct {
	Pix   *Img
	Scale float64
}

var ImageExts = []string{".png", ".exr", ".jpg", ".jpeg", ".bmp", ".tga", ".gif"}

func isImageExt(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	for _, e := range ImageExts {
		if ext == e {
			return true
		}
	}
	return false
}

func loadImage(path string) (*LImage, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".exr":
		return loadEXRImage(path)
	case ".bmp":
		return loadBMP(path)
	case ".tga":
		return loadTGA(path)
	case ".tif", ".tiff":
		return nil, fmt.Errorf("TIFF images are not supported - save the image as PNG (or EXR for HDR)")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	im, _, err := image.Decode(bufio.NewReader(f))
	if err != nil {
		return nil, fmt.Errorf("Cannot read image %s: %v", filepath.Base(path), err)
	}
	return fromGoImage(im), nil
}

func fromGoImage(im image.Image) *LImage {
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	out := NewImg(w, h, 4, false)
	p := out.F32
	set := func(i int, r, g, bl, a float32) {
		p[4*i], p[4*i+1], p[4*i+2], p[4*i+3] = r, g, bl, a
	}
	scale := 255.0
	switch m := im.(type) {
	case *image.Gray:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := float32(m.Pix[y*m.Stride+x])
				set(y*w+x, v, v, v, 255)
			}
		}
	case *image.Gray16:
		scale = 65535
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				o := y*m.Stride + 2*x
				v := float32(uint16(m.Pix[o])<<8 | uint16(m.Pix[o+1]))
				set(y*w+x, v, v, v, 65535)
			}
		}
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				o := y*m.Stride + 4*x
				set(y*w+x, float32(m.Pix[o]), float32(m.Pix[o+1]), float32(m.Pix[o+2]), float32(m.Pix[o+3]))
			}
		}
	case *image.NRGBA64:
		scale = 65535
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				o := y*m.Stride + 8*x
				g := func(k int) float32 { return float32(uint16(m.Pix[o+2*k])<<8 | uint16(m.Pix[o+2*k+1])) }
				set(y*w+x, g(0), g(1), g(2), g(3))
			}
		}
	case *image.RGBA64:
		scale = 65535
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBA64Model.Convert(m.RGBA64At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA64)
				set(y*w+x, float32(c.R), float32(c.G), float32(c.B), float32(c.A))
			}
		}
	default:
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				c := color.NRGBAModel.Convert(im.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				set(y*w+x, float32(c.R), float32(c.G), float32(c.B), float32(c.A))
			}
		}
	}
	return &LImage{Pix: out, Scale: scale}
}

func loadEXRImage(path string) (*LImage, error) {
	e, err := readEXR(path)
	if err != nil {
		return nil, err
	}
	allUint := true
	for _, c := range e.chans {
		if c.ptype != exrUint {
			allUint = false
		}
	}
	out := NewImg(e.w, e.h, 4, allUint)
	roles := map[string]int{}
	var unnamed []int
	for k, c := range e.chans {
		r := exrChannelRole(c.name)
		if _, dup := roles[r]; r != "" && !dup {
			roles[r] = k
		} else {
			unnamed = append(unnamed, k)
		}
	}
	src := [4]int{-1, -1, -1, -1}
	colorCount := 0
	for i, r := range []string{"r", "g", "b"} {
		if k, ok := roles[r]; ok {
			src[i] = k
			colorCount++
		}
	}
	if k, ok := roles["a"]; ok {
		src[3] = k
	}
	if colorCount == 0 {
		gray := -1
		if k, ok := roles["y"]; ok {
			gray = k
		} else if len(unnamed) > 0 {
			gray = unnamed[0]
		}
		src[0], src[1], src[2] = gray, gray, gray
	} else if colorCount == 1 && src[0] >= 0 {
		src[1], src[2] = src[0], src[0] // single colour channel = grayscale
	}
	n := e.w * e.h
	scale := 1.0
	if allUint {
		scale = 4294967295
	}
	for c := 0; c < 4; c++ {
		for i := 0; i < n; i++ {
			var v float64
			if src[c] >= 0 {
				v = e.data[src[c]][i]
			} else if c == 3 {
				v = scale
			}
			out.Set(4*i+c, v)
		}
	}
	return &LImage{Pix: out, Scale: scale}, nil
}

// ---------------------------------------------------------------- BMP

func loadBMP(path string) (*LImage, error) {
	d, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	bad := fmt.Errorf("Unsupported or corrupt BMP file: %s", filepath.Base(path))
	if len(d) < 54 || d[0] != 'B' || d[1] != 'M' {
		return nil, bad
	}
	off := int(binary.LittleEndian.Uint32(d[10:]))
	hsz := int(binary.LittleEndian.Uint32(d[14:]))
	if hsz < 40 || 14+hsz > len(d) {
		return nil, bad
	}
	w := int(int32(binary.LittleEndian.Uint32(d[18:])))
	h := int(int32(binary.LittleEndian.Uint32(d[22:])))
	bpp := int(binary.LittleEndian.Uint16(d[28:]))
	comp := binary.LittleEndian.Uint32(d[30:])
	topDown := h < 0
	if topDown {
		h = -h
	}
	if w <= 0 || h <= 0 || w > 1<<16 || h > 1<<16 {
		return nil, bad
	}
	masks := [4]uint32{0xFF0000, 0xFF00, 0xFF, 0}
	if bpp == 32 {
		masks[3] = 0xFF000000
	}
	if bpp == 16 {
		masks = [4]uint32{0x7C00, 0x3E0, 0x1F, 0}
	}
	if comp == 3 || comp == 6 { // BI_BITFIELDS / BI_ALPHABITFIELDS
		mo := 54
		if hsz >= 52 {
			mo = 14 + 40
		}
		if mo+12 > len(d) {
			return nil, bad
		}
		for k := 0; k < 3; k++ {
			masks[k] = binary.LittleEndian.Uint32(d[mo+4*k:])
		}
		if hsz >= 56 || comp == 6 {
			masks[3] = binary.LittleEndian.Uint32(d[mo+12:])
		}
	} else if comp != 0 {
		return nil, fmt.Errorf("Compressed BMP files are not supported - save as PNG")
	}
	var pal [][4]byte
	if bpp <= 8 {
		n := int(binary.LittleEndian.Uint32(d[46:]))
		if n == 0 {
			n = 1 << uint(bpp)
		}
		po := 14 + hsz
		for i := 0; i < n && po+4*i+4 <= len(d); i++ {
			q := d[po+4*i:]
			pal = append(pal, [4]byte{q[2], q[1], q[0], 255})
		}
	}
	stride := ((w*bpp + 31) / 32) * 4
	if off+stride*h > len(d) {
		return nil, bad
	}
	out := NewImg(w, h, 4, false)
	// alpha in 32-bit BMPs is only trusted when it is not all zero
	useAlpha := masks[3] != 0
	if useAlpha && bpp == 32 {
		allZero := true
		for y := 0; y < h && allZero; y++ {
			for x := 0; x < w; x++ {
				if binary.LittleEndian.Uint32(d[off+y*stride+4*x:])&masks[3] != 0 {
					allZero = false
					break
				}
			}
		}
		useAlpha = !allZero
	}
	field := func(v, m uint32) float32 {
		if m == 0 {
			return 0
		}
		sh := 0
		for m&1 == 0 {
			m >>= 1
			sh++
		}
		return float32((v>>uint(sh))&m) * 255 / float32(m)
	}
	for y := 0; y < h; y++ {
		row := d[off+y*stride:]
		dy := h - 1 - y
		if topDown {
			dy = y
		}
		for x := 0; x < w; x++ {
			var r, g, b, a float32 = 0, 0, 0, 255
			switch bpp {
			case 1, 2, 4, 8:
				bitPos := x * bpp
				idx := int(row[bitPos/8]>>uint(8-bpp-bitPos%8)) & (1<<uint(bpp) - 1)
				if idx < len(pal) {
					r, g, b = float32(pal[idx][0]), float32(pal[idx][1]), float32(pal[idx][2])
				}
			case 16, 24, 32:
				var v uint32
				for k := 0; k < bpp/8; k++ {
					v |= uint32(row[x*bpp/8+k]) << uint(8*k)
				}
				r, g, b = field(v, masks[0]), field(v, masks[1]), field(v, masks[2])
				if useAlpha {
					a = field(v, masks[3])
				}
			default:
				return nil, bad
			}
			o := 4 * (dy*w + x)
			out.F32[o], out.F32[o+1], out.F32[o+2], out.F32[o+3] = float32(math.Round(float64(r))), float32(math.Round(float64(g))), float32(math.Round(float64(b))), float32(math.Round(float64(a)))
		}
	}
	return &LImage{Pix: out, Scale: 255}, nil
}

// ---------------------------------------------------------------- TGA

func loadTGA(path string) (*LImage, error) {
	d, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	bad := fmt.Errorf("Unsupported or corrupt TGA file: %s", filepath.Base(path))
	if len(d) < 18 {
		return nil, bad
	}
	idLen := int(d[0])
	cmapType := d[1]
	typ := d[2]
	cmapFirst := int(binary.LittleEndian.Uint16(d[3:]))
	cmapLen := int(binary.LittleEndian.Uint16(d[5:]))
	cmapBits := int(d[7])
	w := int(binary.LittleEndian.Uint16(d[12:]))
	h := int(binary.LittleEndian.Uint16(d[14:]))
	bpp := int(d[16])
	desc := d[17]
	if w == 0 || h == 0 {
		return nil, bad
	}
	pos := 18 + idLen
	var pal [][4]float32
	readColor := func(p []byte, bits int) [4]float32 {
		switch bits {
		case 8:
			v := float32(p[0])
			return [4]float32{v, v, v, 255}
		case 15, 16:
			v := uint16(p[0]) | uint16(p[1])<<8
			c := func(s uint) float32 { return float32(math.Round(float64((v>>s)&31) * 255 / 31)) }
			a := float32(255)
			if bits == 16 && desc&15 != 0 && v&0x8000 == 0 {
				a = 0
			}
			return [4]float32{c(10), c(5), c(0), a}
		case 24:
			return [4]float32{float32(p[2]), float32(p[1]), float32(p[0]), 255}
		case 32:
			a := float32(p[3])
			if desc&15 == 0 {
				a = 255
			}
			return [4]float32{float32(p[2]), float32(p[1]), float32(p[0]), a}
		}
		return [4]float32{}
	}
	if cmapType == 1 {
		eb := (cmapBits + 7) / 8
		if pos+eb*cmapLen > len(d) {
			return nil, bad
		}
		pal = make([][4]float32, cmapFirst+cmapLen)
		for i := 0; i < cmapLen; i++ {
			pal[cmapFirst+i] = readColor(d[pos+eb*i:], cmapBits)
		}
		pos += eb * cmapLen
	}
	base := typ & 7
	rle := typ&8 != 0
	if base < 1 || base > 3 {
		return nil, bad
	}
	pb := (bpp + 7) / 8
	n := w * h
	pixels := make([]byte, 0, n*pb)
	if rle {
		for len(pixels) < n*pb {
			if pos >= len(d) {
				return nil, bad
			}
			hd := d[pos]
			pos++
			cnt := int(hd&0x7F) + 1
			if hd&0x80 != 0 {
				if pos+pb > len(d) {
					return nil, bad
				}
				for k := 0; k < cnt; k++ {
					pixels = append(pixels, d[pos:pos+pb]...)
				}
				pos += pb
			} else {
				if pos+cnt*pb > len(d) {
					return nil, bad
				}
				pixels = append(pixels, d[pos:pos+cnt*pb]...)
				pos += cnt * pb
			}
		}
		pixels = pixels[:n*pb]
	} else {
		if pos+n*pb > len(d) {
			return nil, bad
		}
		pixels = d[pos : pos+n*pb]
	}
	out := NewImg(w, h, 4, false)
	topDown := desc&0x20 != 0
	rightLeft := desc&0x10 != 0
	for i := 0; i < n; i++ {
		p := pixels[i*pb:]
		var c [4]float32
		switch base {
		case 1:
			idx := int(p[0])
			if pb == 2 {
				idx |= int(p[1]) << 8
			}
			if idx >= len(pal) {
				return nil, bad
			}
			c = pal[idx]
		case 2:
			c = readColor(p, bpp)
		case 3:
			v := float32(p[0])
			a := float32(255)
			if pb == 2 {
				a = float32(p[1])
			}
			c = [4]float32{v, v, v, a}
		}
		x, y := i%w, i/w
		if !topDown {
			y = h - 1 - y
		}
		if rightLeft {
			x = w - 1 - x
		}
		o := 4 * (y*w + x)
		copy(out.F32[o:o+4], c[:])
	}
	return &LImage{Pix: out, Scale: 255}, nil
}

// ---------------------------------------------------------------- saving

// saveImage writes an image whose channels are already mapped for the file kind:
// png8/png16 hold normalized 0..1 values, exr_uint raw integers, exr_float floats.
func saveImage(path string, img *Img, kind string) error {
	w, h, C := img.W, img.H, img.C
	if strings.HasPrefix(kind, "exr") {
		names := map[int][]string{1: {"Y"}, 2: {"Y", "A"}, 3: {"R", "G", "B"}, 4: {"R", "G", "B", "A"}}[C]
		pt := exrFloat
		if kind == "exr_uint" {
			pt = exrUint
		}
		return writeEXR(path, img, names, pt)
	}
	sixteen := kind == "png16"
	if C == 2 {
		return writeGrayAlphaPNG(path, img, sixteen)
	}
	q := func(v float64) uint32 {
		if sixteen {
			return uint32(rint(clamp(v, 0, 1) * 65535))
		}
		return uint32(rint(clamp(v, 0, 1) * 255))
	}
	var out image.Image
	switch {
	case C == 1 && !sixteen:
		m := image.NewGray(image.Rect(0, 0, w, h))
		for i := 0; i < w*h; i++ {
			m.Pix[i] = uint8(q(img.Get(i)))
		}
		out = m
	case C == 1:
		m := image.NewGray16(image.Rect(0, 0, w, h))
		for i := 0; i < w*h; i++ {
			v := q(img.Get(i))
			m.Pix[2*i], m.Pix[2*i+1] = uint8(v>>8), uint8(v)
		}
		out = m
	default:
		get := func(i, c int) uint32 {
			switch C {
			case 2: // gray + alpha
				if c == 3 {
					return q(img.Get(i*2 + 1))
				}
				return q(img.Get(i * 2))
			case 3:
				if c == 3 {
					if sixteen {
						return 65535
					}
					return 255
				}
			}
			return q(img.Get(i*C + c))
		}
		if !sixteen {
			m := image.NewNRGBA(image.Rect(0, 0, w, h))
			for i := 0; i < w*h; i++ {
				for c := 0; c < 4; c++ {
					m.Pix[4*i+c] = uint8(get(i, c))
				}
			}
			out = m
		} else {
			m := image.NewNRGBA64(image.Rect(0, 0, w, h))
			for i := 0; i < w*h; i++ {
				for c := 0; c < 4; c++ {
					v := get(i, c)
					m.Pix[8*i+2*c], m.Pix[8*i+2*c+1] = uint8(v>>8), uint8(v)
				}
			}
			out = m
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(f)
	if err := (&png.Encoder{CompressionLevel: png.DefaultCompression}).Encode(bw, out); err != nil {
		f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// writeGrayAlphaPNG writes a 2-channel (gray + alpha) PNG, which image/png cannot produce.
func writeGrayAlphaPNG(path string, img *Img, sixteen bool) error {
	w, h := img.W, img.H
	bpc := 1
	if sixteen {
		bpc = 2
	}
	raw := make([]byte, 0, h*(1+w*2*bpc))
	for y := 0; y < h; y++ {
		raw = append(raw, 0) // filter: none
		for x := 0; x < w; x++ {
			for c := 0; c < 2; c++ {
				v := clamp(img.Get((y*w+x)*2+c), 0, 1)
				if sixteen {
					q := uint16(rint(v * 65535))
					raw = append(raw, byte(q>>8), byte(q))
				} else {
					raw = append(raw, byte(rint(v*255)))
				}
			}
		}
	}
	var z bytes.Buffer
	zw, _ := zlib.NewWriterLevel(&z, zlib.DefaultCompression)
	zw.Write(raw)
	zw.Close()
	var out bytes.Buffer
	out.Write([]byte("\x89PNG\r\n\x1a\n"))
	chunk := func(typ string, data []byte) {
		var l [4]byte
		binary.BigEndian.PutUint32(l[:], uint32(len(data)))
		out.Write(l[:])
		crc := crc32.NewIEEE()
		crc.Write([]byte(typ))
		crc.Write(data)
		out.WriteString(typ)
		out.Write(data)
		binary.BigEndian.PutUint32(l[:], crc.Sum32())
		out.Write(l[:])
	}
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr, uint32(w))
	binary.BigEndian.PutUint32(ihdr[4:], uint32(h))
	ihdr[8] = byte(8 * bpc)
	ihdr[9] = 4 // gray + alpha
	chunk("IHDR", ihdr)
	chunk("IDAT", z.Bytes())
	chunk("IEND", nil)
	return os.WriteFile(path, out.Bytes(), 0644)
}
