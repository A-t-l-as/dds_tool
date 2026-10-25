package main

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Img is an interleaved multi-channel float image. Exactly one of F32/F64 is used.
type Img struct {
	W, H, C int
	F32     []float32
	F64     []float64
}

func NewImg(w, h, c int, wide bool) *Img {
	m := &Img{W: w, H: h, C: c}
	if wide {
		m.F64 = make([]float64, w*h*c)
	} else {
		m.F32 = make([]float32, w*h*c)
	}
	return m
}

func (m *Img) Wide() bool { return m.F64 != nil }

func (m *Img) Get(i int) float64 {
	if m.F64 != nil {
		return m.F64[i]
	}
	return float64(m.F32[i])
}

func (m *Img) Set(i int, v float64) {
	if m.F64 != nil {
		m.F64[i] = v
	} else {
		m.F32[i] = float32(v)
	}
}

func (m *Img) Len() int { return m.W * m.H * m.C }

func (m *Img) Clone() *Img {
	n := &Img{W: m.W, H: m.H, C: m.C}
	if m.F64 != nil {
		n.F64 = append([]float64(nil), m.F64...)
	} else {
		n.F32 = append([]float32(nil), m.F32...)
	}
	return n
}

// ---------------------------------------------------------------- half floats

func halfToFloat(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		f := float32(mant) * (1.0 / 16777216.0)
		if sign != 0 {
			f = -f
		}
		return f
	case 31:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

// floatToHalf converts with round-to-nearest-even.
func floatToHalf(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b >> 23) & 0xff)
	mant := b & 0x7fffff
	if exp == 255 {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	e := exp - 127 + 15
	if e >= 31 {
		return sign | 0x7c00
	}
	if e <= 0 {
		shift := uint(14 - e)
		if shift >= 25 {
			return sign
		}
		m := mant | 0x800000
		q := m >> shift
		rem := m & ((1 << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && q&1 == 1) {
			q++
		}
		return sign | uint16(q)
	}
	q := uint32(e)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && q&1 == 1) {
		q++
	}
	return sign | uint16(q)
}

// ---------------------------------------------------------------- helpers

func rint(v float64) float64 { return math.RoundToEven(v) }

func clamp(v, lo, hi float64) float64 {
	if v != v {
		v = 0
	}
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func signExtend(raw uint64, nbits int) int64 {
	v := int64(raw)
	if (raw>>uint(nbits-1))&1 == 1 {
		v -= int64(1) << uint(nbits)
	}
	return v
}

func rawToValue(raw uint64, nbits int, ct string) float64 {
	switch ct {
	case "unorm":
		return float64(raw) / float64((uint64(1)<<uint(nbits))-1)
	case "snorm":
		s := float64(signExtend(raw, nbits))
		return math.Max(s/float64((int64(1)<<uint(nbits-1))-1), -1)
	case "uint":
		return float64(raw)
	case "sint":
		return float64(signExtend(raw, nbits))
	case "float":
		if nbits == 16 {
			return float64(halfToFloat(uint16(raw)))
		}
		return float64(math.Float32frombits(uint32(raw)))
	case "uf11":
		return float64(halfToFloat(uint16(raw << 4)))
	case "uf10":
		return float64(halfToFloat(uint16(raw << 5)))
	}
	panic("unknown channel type " + ct)
}

func valueToRaw(v float64, nbits int, ct string) uint64 {
	mask := (uint64(1) << uint(nbits)) - 1
	switch ct {
	case "unorm":
		return uint64(rint(clamp(v, 0, 1) * float64(mask)))
	case "snorm":
		s := int64(rint(clamp(v, -1, 1) * float64((int64(1)<<uint(nbits-1))-1)))
		return uint64(s) & mask
	case "uint":
		return uint64(clamp(rint(v), 0, float64(mask)))
	case "sint":
		lo := -float64(int64(1) << uint(nbits-1))
		hi := float64((int64(1) << uint(nbits-1)) - 1)
		return uint64(int64(clamp(rint(v), lo, hi))) & mask
	case "float":
		if nbits == 16 {
			return uint64(floatToHalf(float32(v)))
		}
		return uint64(math.Float32bits(float32(v)))
	case "uf11", "uf10":
		sh, mx, top := uint(4), uint64(0x7BF), 65024.0
		if ct == "uf10" {
			sh, mx, top = 5, 0x3DF, 64512.0
		}
		f := float32(v)
		switch {
		case f != f:
			f = 0
		case math.IsInf(float64(f), 1):
			f = float32(top)
		case math.IsInf(float64(f), -1):
			f = 0
		}
		h := uint64(floatToHalf(float32(clamp(float64(f), 0, top))))
		r := (h + (1 << (sh - 1))) >> sh
		if r > mx {
			r = mx
		}
		return r
	case "x":
		return mask
	}
	panic("unknown channel type " + ct)
}

func rgb9e5Decode(word uint32) (float64, float64, float64) {
	e := float64((word >> 27) & 31)
	scale := math.Exp2(e - 15 - 9)
	return float64(word&511) * scale, float64((word>>9)&511) * scale, float64((word>>18)&511) * scale
}

func rgb9e5Encode(r, g, b float64) uint32 {
	maxv := 511.0 / 512.0 * 65536.0
	rr, gg, bb := clamp(r, 0, maxv), clamp(g, 0, maxv), clamp(b, 0, maxv)
	mx := math.Max(math.Max(rr, gg), bb)
	expP := math.Max(-16, math.Floor(math.Log2(math.Max(mx, 1e-30)))) + 1 + 15
	denom := math.Exp2(expP - 15 - 9)
	if math.Floor(mx/denom+0.5) == 512 {
		denom *= 2
		expP++
	}
	word := (uint32(expP) & 31) << 27
	for i, x := range []float64{rr, gg, bb} {
		m := clamp(math.Floor(x/denom+0.5), 0, 511)
		word |= uint32(m) << uint(9*i)
	}
	return word
}

// ---------------------------------------------------------------- surfaces

func readWord(p []byte, n int) uint64 {
	var w uint64
	for k := 0; k < n; k++ {
		w |= uint64(p[k]) << uint(8*k)
	}
	return w
}

func rawDecode(buf []byte, f *Fmt, w, h int, wide bool) (*Img, error) {
	es := f.ESize
	if len(buf) < w*h*es {
		return nil, fmt.Errorf("The DDS file is truncated (texture data missing)")
	}
	vch := f.ValueChannels()
	img := NewImg(w, h, len(vch), wide)
	n := w * h
	if f.Special == "rgb9e5" {
		for i := 0; i < n; i++ {
			r, g, b := rgb9e5Decode(binary.LittleEndian.Uint32(buf[i*4:]))
			img.Set(i*3, r)
			img.Set(i*3+1, g)
			img.Set(i*3+2, b)
		}
		return img, nil
	}
	var chans []Chan
	for _, c := range f.Channels {
		if c.Type != "x" {
			chans = append(chans, c)
		}
	}
	C := len(chans)
	parallelRows(h, func(y0, y1 int) {
		for i := y0 * w; i < y1*w; i++ {
			p := buf[i*es : (i+1)*es]
			var word uint64
			if es <= 8 {
				word = readWord(p, es)
			}
			for k, c := range chans {
				var raw uint64
				if es <= 8 {
					raw = (word >> uint(c.Off)) & ((uint64(1) << uint(c.Bits)) - 1)
				} else {
					raw = readWord(p[c.Off/8:], c.Bits/8)
				}
				img.Set(i*C+k, rawToValue(raw, c.Bits, c.Type))
			}
		}
	})
	return img, nil
}

// rawEncode packs an image whose channels follow f.ValueChannels() order.
func rawEncode(img *Img, f *Fmt) []byte {
	w, h := img.W, img.H
	n := w * h
	if f.Special == "rgb9e5" {
		out := make([]byte, n*4)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(out[i*4:], rgb9e5Encode(img.Get(i*3), img.Get(i*3+1), img.Get(i*3+2)))
		}
		return out
	}
	es := f.ESize
	out := make([]byte, n*es)
	idx := map[string]int{}
	for k, c := range f.ValueChannels() {
		idx[c.Name] = k
	}
	C := img.C
	parallelRows(h, func(y0, y1 int) {
		for i := y0 * w; i < y1*w; i++ {
			p := out[i*es : (i+1)*es]
			var word uint64
			for _, c := range f.Channels {
				v := 0.0
				if c.Type != "x" {
					v = img.Get(i*C + idx[c.Name])
				}
				raw := valueToRaw(v, c.Bits, c.Type) & ((uint64(1) << uint(c.Bits)) - 1)
				if es <= 8 {
					word |= raw << uint(c.Off)
				} else {
					for k := 0; k < c.Bits/8; k++ {
						p[c.Off/8+k] = byte(raw >> uint(8*k))
					}
				}
			}
			if es <= 8 {
				for k := 0; k < es; k++ {
					p[k] = byte(word >> uint(8*k))
				}
			}
		}
	})
	return out
}
