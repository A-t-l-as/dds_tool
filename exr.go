package main

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// Minimal OpenEXR support: scanline images, compression NONE/RLE/ZIPS/ZIP/PIZ/PXR24 (read),
// ZIP (write), pixel types HALF/FLOAT/UINT.

const (
	exrUint  = 0
	exrHalf  = 1
	exrFloat = 2
)

var exrCompressionNames = []string{"none", "RLE", "ZIPS", "ZIP", "PIZ", "PXR24", "B44", "B44A", "DWAA", "DWAB"}

type exrChan struct {
	name  string
	ptype int
	size  int // bytes per sample
}

func exrLinesPerChunk(comp int) int {
	switch comp {
	case 3, 5:
		return 16
	case 4, 6, 7:
		return 32
	}
	return 1
}

// ---------------------------------------------------------------- writer

func writeEXR(path string, img *Img, names []string, ptype int) error {
	w, h := img.W, img.H
	type ch struct {
		name string
		src  int
	}
	var chans []ch
	for i, n := range names {
		chans = append(chans, ch{n, i})
	}
	sort.Slice(chans, func(i, j int) bool { return chans[i].name < chans[j].name })
	var hdr bytes.Buffer
	le := func(v interface{}) { binary.Write(&hdr, binary.LittleEndian, v) }
	attr := func(name, typ string, data []byte) {
		hdr.WriteString(name + "\x00" + typ + "\x00")
		le(int32(len(data)))
		hdr.Write(data)
	}
	var cl bytes.Buffer
	for _, c := range chans {
		cl.WriteString(c.name + "\x00")
		binary.Write(&cl, binary.LittleEndian, []int32{int32(ptype), 0, 1, 1})
	}
	cl.WriteByte(0)
	box := func() []byte {
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, []int32{0, 0, int32(w - 1), int32(h - 1)})
		return b.Bytes()
	}
	f32 := func(vals ...float32) []byte {
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, vals)
		return b.Bytes()
	}
	hdr.Write([]byte{0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0})
	attr("channels", "chlist", cl.Bytes())
	attr("compression", "compression", []byte{3})
	attr("dataWindow", "box2i", box())
	attr("displayWindow", "box2i", box())
	attr("lineOrder", "lineOrder", []byte{0})
	attr("pixelAspectRatio", "float", f32(1))
	attr("screenWindowCenter", "v2f", f32(0, 0))
	attr("screenWindowWidth", "float", f32(1))
	hdr.WriteByte(0)

	lpc := 16
	nChunks := (h + lpc - 1) / lpc
	chunks := make([][]byte, nChunks)
	parallelRows(nChunks, func(c0, c1 int) {
		for ci := c0; ci < c1; ci++ {
			y0 := ci * lpc
			y1 := mini(h, y0+lpc)
			var raw bytes.Buffer
			for y := y0; y < y1; y++ {
				for _, c := range chans {
					for x := 0; x < w; x++ {
						v := img.Get((y*w+x)*img.C + c.src)
						switch ptype {
						case exrHalf:
							binary.Write(&raw, binary.LittleEndian, floatToHalf(float32(v)))
						case exrFloat:
							binary.Write(&raw, binary.LittleEndian, math.Float32bits(float32(v)))
						default:
							binary.Write(&raw, binary.LittleEndian, uint32(clamp(v, 0, 4294967295)))
						}
					}
				}
			}
			rb := raw.Bytes()
			packed := zipPack(rb)
			if len(packed) >= len(rb) {
				packed = rb
			}
			var cb bytes.Buffer
			binary.Write(&cb, binary.LittleEndian, int32(y0))
			binary.Write(&cb, binary.LittleEndian, int32(len(packed)))
			cb.Write(packed)
			chunks[ci] = cb.Bytes()
		}
	})
	var out bytes.Buffer
	out.Write(hdr.Bytes())
	off := uint64(hdr.Len() + 8*nChunks)
	for _, c := range chunks {
		binary.Write(&out, binary.LittleEndian, off)
		off += uint64(len(c))
	}
	for _, c := range chunks {
		out.Write(c)
	}
	return os.WriteFile(path, out.Bytes(), 0644)
}

func zipPack(raw []byte) []byte {
	n := len(raw)
	t := make([]byte, n)
	half := (n + 1) / 2
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			t[i/2] = raw[i]
		} else {
			t[half+i/2] = raw[i]
		}
	}
	p := 0
	if n > 0 {
		p = int(t[0])
	}
	for i := 1; i < n; i++ {
		d := int(t[i]) - p + 128 + 256
		p = int(t[i])
		t[i] = byte(d)
	}
	var b bytes.Buffer
	zw := zlib.NewWriter(&b)
	zw.Write(t)
	zw.Close()
	return b.Bytes()
}

// ---------------------------------------------------------------- reader

func undoPredictor(t []byte) []byte {
	n := len(t)
	for i := 1; i < n; i++ {
		t[i] = byte(int(t[i-1]) + int(t[i]) - 128)
	}
	out := make([]byte, n)
	half := (n + 1) / 2
	for i := 0; i < n; i++ {
		if i%2 == 0 {
			out[i] = t[i/2]
		} else {
			out[i] = t[half+i/2]
		}
	}
	return out
}

func zlibUnpack(src []byte, size int) ([]byte, error) {
	zr, err := zlib.NewReader(bytes.NewReader(src))
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	out := make([]byte, size)
	if _, err := io.ReadFull(zr, out); err != nil {
		return nil, err
	}
	return out, nil
}

func rleUnpack(src []byte, size int) ([]byte, error) {
	out := make([]byte, 0, size)
	for i := 0; i < len(src); {
		cnt := int(int8(src[i]))
		i++
		if cnt < 0 {
			if i-cnt > len(src) {
				return nil, errors.New("corrupt RLE data")
			}
			out = append(out, src[i:i-cnt]...)
			i -= cnt
		} else {
			if i >= len(src) {
				return nil, errors.New("corrupt RLE data")
			}
			for k := 0; k <= cnt; k++ {
				out = append(out, src[i])
			}
			i++
		}
	}
	if len(out) != size {
		return nil, errors.New("corrupt RLE data")
	}
	return out, nil
}

func pxr24Unpack(src []byte, chans []exrChan, w, lines, size int) ([]byte, error) {
	tmpSize := 0
	for _, c := range chans {
		n := 4
		if c.ptype == exrHalf {
			n = 2
		} else if c.ptype == exrFloat {
			n = 3
		}
		tmpSize += n * w * lines
	}
	tmp, err := zlibUnpack(src, tmpSize)
	if err != nil {
		return nil, err
	}
	out := make([]byte, 0, size)
	p := 0
	var b4 [4]byte
	for y := 0; y < lines; y++ {
		for _, c := range chans {
			var pixel uint32
			switch c.ptype {
			case exrUint:
				p0, p1, p2, p3 := p, p+w, p+2*w, p+3*w
				p += 4 * w
				for j := 0; j < w; j++ {
					pixel += uint32(tmp[p0+j])<<24 | uint32(tmp[p1+j])<<16 | uint32(tmp[p2+j])<<8 | uint32(tmp[p3+j])
					binary.LittleEndian.PutUint32(b4[:], pixel)
					out = append(out, b4[:]...)
				}
			case exrHalf:
				p0, p1 := p, p+w
				p += 2 * w
				for j := 0; j < w; j++ {
					pixel += uint32(tmp[p0+j])<<8 | uint32(tmp[p1+j])
					out = append(out, byte(pixel), byte(pixel>>8))
				}
			default:
				p0, p1, p2 := p, p+w, p+2*w
				p += 3 * w
				for j := 0; j < w; j++ {
					pixel += uint32(tmp[p0+j])<<24 | uint32(tmp[p1+j])<<16 | uint32(tmp[p2+j])<<8
					binary.LittleEndian.PutUint32(b4[:], pixel)
					out = append(out, b4[:]...)
				}
			}
		}
	}
	return out, nil
}

type exrImage struct {
	w, h  int
	chans []exrChan
	data  [][]float64 // per channel, w*h values
}

func readEXR(path string) (*exrImage, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data) < 8 || binary.LittleEndian.Uint32(data) != 20000630 {
		return nil, fmt.Errorf("Not an OpenEXR file: %s", path)
	}
	ver := binary.LittleEndian.Uint32(data[4:])
	if ver&0x200 != 0 {
		return nil, fmt.Errorf("Tiled EXR files are not supported (save as scanline EXR)")
	}
	if ver&0x1800 != 0 {
		return nil, fmt.Errorf("Multi-part / deep EXR files are not supported")
	}
	pos := 8
	readStr := func() (string, error) {
		i := bytes.IndexByte(data[pos:], 0)
		if i < 0 {
			return "", errors.New("corrupt EXR header")
		}
		s := string(data[pos : pos+i])
		pos += i + 1
		return s, nil
	}
	var chans []exrChan
	comp := 0
	var dw [4]int32
	haveDW := false
	for {
		name, err := readStr()
		if err != nil {
			return nil, err
		}
		if name == "" {
			break
		}
		typ, err := readStr()
		if err != nil {
			return nil, err
		}
		if pos+4 > len(data) {
			return nil, errors.New("corrupt EXR header")
		}
		sz := int(binary.LittleEndian.Uint32(data[pos:]))
		pos += 4
		if pos+sz > len(data) {
			return nil, errors.New("corrupt EXR header")
		}
		val := data[pos : pos+sz]
		pos += sz
		switch {
		case name == "channels" && typ == "chlist":
			q := 0
			for q < len(val) && val[q] != 0 {
				i := bytes.IndexByte(val[q:], 0)
				if i < 0 || q+i+17 > len(val) {
					return nil, errors.New("corrupt EXR channel list")
				}
				cn := string(val[q : q+i])
				q += i + 1
				pt := int(binary.LittleEndian.Uint32(val[q:]))
				xs := binary.LittleEndian.Uint32(val[q+8:])
				ys := binary.LittleEndian.Uint32(val[q+12:])
				q += 16
				if xs != 1 || ys != 1 {
					return nil, fmt.Errorf("EXR channel %q uses subsampling, which is not supported", cn)
				}
				size := 4
				if pt == exrHalf {
					size = 2
				} else if pt != exrUint && pt != exrFloat {
					return nil, fmt.Errorf("Unknown EXR pixel type %d", pt)
				}
				chans = append(chans, exrChan{cn, pt, size})
			}
		case name == "compression":
			comp = int(val[0])
		case name == "dataWindow" && sz == 16:
			for k := 0; k < 4; k++ {
				dw[k] = int32(binary.LittleEndian.Uint32(val[4*k:]))
			}
			haveDW = true
		}
	}
	if !haveDW || len(chans) == 0 {
		return nil, errors.New("EXR file has no data window or channels")
	}
	if comp > 5 {
		name := "unknown"
		if comp < len(exrCompressionNames) {
			name = exrCompressionNames[comp]
		}
		return nil, fmt.Errorf("EXR compression %s is not supported - save the EXR with ZIP, PIZ or no compression", name)
	}
	w := int(dw[2]-dw[0]) + 1
	h := int(dw[3]-dw[1]) + 1
	if w <= 0 || h <= 0 || w > 1<<16 || h > 1<<16 {
		return nil, errors.New("invalid EXR size")
	}
	lpc := exrLinesPerChunk(comp)
	nChunks := (h + lpc - 1) / lpc
	if pos+8*nChunks > len(data) {
		return nil, errors.New("truncated EXR file")
	}
	img := &exrImage{w: w, h: h, chans: chans, data: make([][]float64, len(chans))}
	for i := range chans {
		img.data[i] = make([]float64, w*h)
	}
	lineBytes := 0
	for _, c := range chans {
		lineBytes += c.size * w
	}
	var firstErr error
	errMu := make(chan struct{}, 1)
	errMu <- struct{}{}
	setErr := func(e error) {
		<-errMu
		if firstErr == nil {
			firstErr = e
		}
		errMu <- struct{}{}
	}
	parallelRows(nChunks, func(c0, c1 int) {
		for ci := c0; ci < c1; ci++ {
			off := int(binary.LittleEndian.Uint64(data[pos+8*ci:]))
			if off < 0 || off+8 > len(data) {
				setErr(errors.New("truncated EXR file"))
				return
			}
			y := int(int32(binary.LittleEndian.Uint32(data[off:]))) - int(dw[1])
			sz := int(binary.LittleEndian.Uint32(data[off+4:]))
			if off+8+sz > len(data) || y < 0 || y >= h {
				setErr(errors.New("truncated or corrupt EXR file"))
				return
			}
			src := data[off+8 : off+8+sz]
			lines := mini(lpc, h-y)
			want := lines * lineBytes
			var raw []byte
			var err error
			if sz == want || comp == 0 {
				raw = src
			} else {
				switch comp {
				case 1:
					raw, err = rleUnpack(src, want)
					if err == nil {
						raw = undoPredictor(raw)
					}
				case 2, 3:
					raw, err = zlibUnpack(src, want)
					if err == nil {
						raw = undoPredictor(raw)
					}
				case 4:
					raw, err = pizUnpack(src, chans, w, lines, want)
				case 5:
					raw, err = pxr24Unpack(src, chans, w, lines, want)
				}
			}
			if err == nil && len(raw) < want {
				err = errors.New("short EXR chunk")
			}
			if err != nil {
				setErr(fmt.Errorf("cannot decode EXR data: %v", err))
				return
			}
			q := 0
			for l := 0; l < lines; l++ {
				for k, c := range chans {
					dst := img.data[k][(y+l)*w:]
					for x := 0; x < w; x++ {
						switch c.ptype {
						case exrHalf:
							dst[x] = float64(halfToFloat(binary.LittleEndian.Uint16(raw[q:])))
						case exrFloat:
							dst[x] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[q:])))
						default:
							dst[x] = float64(binary.LittleEndian.Uint32(raw[q:]))
						}
						q += c.size
					}
				}
			}
		}
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return img, nil
}

func exrChannelRole(name string) string {
	n := name
	if i := strings.LastIndexByte(n, '.'); i >= 0 {
		n = n[i+1:]
	}
	switch strings.ToUpper(n) {
	case "R", "RED":
		return "r"
	case "G", "GREEN":
		return "g"
	case "B", "BLUE":
		return "b"
	case "A", "ALPHA":
		return "a"
	case "Y", "L", "GRAY", "GREY":
		return "y"
	}
	return ""
}

// ---------------------------------------------------------------- PIZ

const (
	hufEncBits = 16
	hufDecBits = 14
	hufEncSize = (1 << hufEncBits) + 1
	hufDecSize = 1 << hufDecBits
	hufDecMask = hufDecSize - 1
)

type hufDec struct {
	len int
	lit int
	p   []int
}

var errPiz = errors.New("corrupt PIZ data")

type bitReader struct {
	data []byte
	pos  int
	c    int64
	lc   int
}

func (b *bitReader) getBits(n int) (int64, error) {
	for b.lc < n {
		if b.pos >= len(b.data) {
			return 0, errPiz
		}
		b.c = b.c<<8 | int64(b.data[b.pos])
		b.pos++
		b.lc += 8
	}
	b.lc -= n
	return (b.c >> uint(b.lc)) & ((1 << uint(n)) - 1), nil
}

func hufCanonicalCodeTable(hcode []int64) {
	var n [59]int64
	for _, l := range hcode {
		n[l]++
	}
	c := int64(0)
	for i := 58; i > 0; i-- {
		nc := (c + n[i]) >> 1
		n[i] = c
		c = nc
	}
	for i, l := range hcode {
		if l > 0 {
			hcode[i] = l | (n[l] << 6)
			n[l]++
		}
	}
}

func hufUnpackEncTable(src []byte, im, iM int, hcode []int64) (int, error) {
	br := &bitReader{data: src}
	for ; im <= iM; im++ {
		l, err := br.getBits(6)
		if err != nil {
			return 0, err
		}
		hcode[im] = l
		if l == 63 {
			z, err := br.getBits(8)
			if err != nil {
				return 0, err
			}
			zerun := int(z) + 6
			if im+zerun > iM+1 {
				return 0, errPiz
			}
			for ; zerun > 0; zerun-- {
				hcode[im] = 0
				im++
			}
			im--
		} else if l >= 59 {
			zerun := int(l) - 59 + 2
			if im+zerun > iM+1 {
				return 0, errPiz
			}
			for ; zerun > 0; zerun-- {
				hcode[im] = 0
				im++
			}
			im--
		}
	}
	hufCanonicalCodeTable(hcode)
	return br.pos, nil
}

func hufBuildDecTable(hcode []int64, im, iM int, hdec []hufDec) error {
	for ; im <= iM; im++ {
		c := hcode[im] >> 6
		l := int(hcode[im] & 63)
		if c>>uint(l) != 0 {
			return errPiz
		}
		if l > hufDecBits {
			pl := &hdec[c>>uint(l-hufDecBits)]
			if pl.len != 0 {
				return errPiz
			}
			pl.lit++
			pl.p = append(pl.p, im)
		} else if l > 0 {
			base := c << uint(hufDecBits-l)
			for i := int64(0); i < int64(1)<<uint(hufDecBits-l); i++ {
				pl := &hdec[base+i]
				if pl.len != 0 || pl.p != nil {
					return errPiz
				}
				pl.len = l
				pl.lit = im
			}
		}
	}
	return nil
}

func hufDecode(hcode []int64, hdec []hufDec, in []byte, nBits int, rlc int, out []uint16) error {
	var c int64
	lc := 0
	o := 0
	no := len(out)
	ie := (nBits + 7) / 8
	if ie > len(in) {
		return errPiz
	}
	pos := 0
	getCode := func(po int) error {
		if po == rlc {
			if lc < 8 {
				if pos >= len(in) {
					return errPiz
				}
				c = c<<8 | int64(in[pos])
				pos++
				lc += 8
			}
			lc -= 8
			cs := int(byte(c >> uint(lc)))
			if o+cs > no || o < 1 {
				return errPiz
			}
			s := out[o-1]
			for ; cs > 0; cs-- {
				out[o] = s
				o++
			}
		} else if o < no {
			out[o] = uint16(po)
			o++
		} else {
			return errPiz
		}
		return nil
	}
	for pos < ie {
		c = c<<8 | int64(in[pos])
		pos++
		lc += 8
		for lc >= hufDecBits {
			pl := hdec[(c>>uint(lc-hufDecBits))&hufDecMask]
			if pl.len != 0 {
				lc -= pl.len
				if err := getCode(pl.lit); err != nil {
					return err
				}
			} else {
				if pl.p == nil {
					return errPiz
				}
				found := false
				for _, sym := range pl.p {
					l := int(hcode[sym] & 63)
					for lc < l && pos < ie {
						c = c<<8 | int64(in[pos])
						pos++
						lc += 8
					}
					if lc >= l && hcode[sym]>>6 == (c>>uint(lc-l))&((int64(1)<<uint(l))-1) {
						lc -= l
						if err := getCode(sym); err != nil {
							return err
						}
						found = true
						break
					}
				}
				if !found {
					return errPiz
				}
			}
		}
	}
	i := (8 - nBits) & 7
	c >>= uint(i)
	lc -= i
	for lc > 0 {
		pl := hdec[(c<<uint(hufDecBits-lc))&hufDecMask]
		if pl.len == 0 {
			return errPiz
		}
		lc -= pl.len
		if err := getCode(pl.lit); err != nil {
			return err
		}
	}
	if o != no {
		return errPiz
	}
	return nil
}

func hufUncompress(src []byte, out []uint16) error {
	if len(src) == 0 {
		if len(out) != 0 {
			return errPiz
		}
		return nil
	}
	if len(src) < 20 {
		return errPiz
	}
	im := int(binary.LittleEndian.Uint32(src))
	iM := int(binary.LittleEndian.Uint32(src[4:]))
	nBits := int(binary.LittleEndian.Uint32(src[12:]))
	if im < 0 || im >= hufEncSize || iM < 0 || iM >= hufEncSize {
		return errPiz
	}
	hcode := make([]int64, hufEncSize)
	used, err := hufUnpackEncTable(src[20:], im, iM, hcode)
	if err != nil {
		return err
	}
	rest := src[20+used:]
	if nBits > 8*len(rest) {
		return errPiz
	}
	hdec := make([]hufDec, hufDecSize)
	if err := hufBuildDecTable(hcode, im, iM, hdec); err != nil {
		return err
	}
	return hufDecode(hcode, hdec, rest, nBits, iM, out)
}

func wdec14(l, h uint16) (uint16, uint16) {
	ls, hs := int(int16(l)), int(int16(h))
	ai := ls + (hs & 1) + (hs >> 1)
	return uint16(int16(ai)), uint16(int16(ai - hs))
}

func wdec16(l, h uint16) (uint16, uint16) {
	m, d := int(l), int(h)
	bb := (m - (d >> 1)) & 0xFFFF
	aa := (d + bb - 0x8000) & 0xFFFF
	return uint16(aa), uint16(bb)
}

func wav2Decode(buf []uint16, base, nx, ox, ny, oy int, mx uint16) {
	dec := wdec16
	if mx < 1<<14 {
		dec = wdec14
	}
	n := mini(nx, ny)
	p := 1
	for p <= n {
		p <<= 1
	}
	p >>= 1
	p2 := p
	p >>= 1
	for p >= 1 {
		py := base
		ey := base + oy*(ny-p2)
		oy1, oy2, ox1, ox2 := oy*p, oy*p2, ox*p, ox*p2
		for ; py <= ey; py += oy2 {
			px := py
			ex := py + ox*(nx-p2)
			for ; px <= ex; px += ox2 {
				p01, p10 := px+ox1, px+oy1
				p11 := p10 + ox1
				i00, i10 := dec(buf[px], buf[p10])
				i01, i11 := dec(buf[p01], buf[p11])
				buf[px], buf[p01] = dec(i00, i01)
				buf[p10], buf[p11] = dec(i10, i11)
			}
			if nx&p != 0 {
				p10 := px + oy1
				var i00 uint16
				i00, buf[p10] = dec(buf[px], buf[p10])
				buf[px] = i00
			}
		}
		if ny&p != 0 {
			px := py
			ex := py + ox*(nx-p2)
			for ; px <= ex; px += ox2 {
				p01 := px + ox1
				var i00 uint16
				i00, buf[p01] = dec(buf[px], buf[p01])
				buf[px] = i00
			}
		}
		p2 = p
		p >>= 1
	}
}

func pizUnpack(src []byte, chans []exrChan, w, lines, size int) (out []byte, err error) {
	defer func() {
		if r := recover(); r != nil {
			out, err = nil, errPiz
		}
	}()
	type cd struct{ start, nx, ny, size int }
	var cds []cd
	total := 0
	for _, c := range chans {
		s := c.size / 2
		cds = append(cds, cd{total, w, lines, s})
		total += w * lines * s
	}
	tmp := make([]uint16, total)
	if len(src) < 4 {
		return nil, errPiz
	}
	minNZ := int(binary.LittleEndian.Uint16(src))
	maxNZ := int(binary.LittleEndian.Uint16(src[2:]))
	pos := 4
	if maxNZ >= 8192 {
		return nil, errPiz
	}
	var bitmap [8192]byte
	if minNZ <= maxNZ {
		n := maxNZ - minNZ + 1
		if pos+n > len(src) {
			return nil, errPiz
		}
		copy(bitmap[minNZ:], src[pos:pos+n])
		pos += n
	}
	lut := make([]uint16, 65536)
	k := 0
	for i := 0; i < 65536; i++ {
		if i == 0 || bitmap[i>>3]&(1<<uint(i&7)) != 0 {
			lut[k] = uint16(i)
			k++
		}
	}
	maxValue := uint16(k - 1)
	if pos+4 > len(src) {
		return nil, errPiz
	}
	length := int(int32(binary.LittleEndian.Uint32(src[pos:])))
	pos += 4
	if length < 0 || pos+length > len(src) {
		return nil, errPiz
	}
	if err := hufUncompress(src[pos:pos+length], tmp); err != nil {
		return nil, err
	}
	for _, c := range cds {
		for j := 0; j < c.size; j++ {
			wav2Decode(tmp, c.start+j, c.nx, c.size, c.ny, c.nx*c.size, maxValue)
		}
	}
	for i, v := range tmp {
		tmp[i] = lut[v]
	}
	out = make([]byte, 0, size)
	ends := make([]int, len(cds))
	for i, c := range cds {
		ends[i] = c.start
	}
	for y := 0; y < lines; y++ {
		for i, c := range cds {
			n := c.nx * c.size
			for _, v := range tmp[ends[i] : ends[i]+n] {
				out = append(out, byte(v), byte(v>>8))
			}
			ends[i] += n
		}
	}
	return out, nil
}
