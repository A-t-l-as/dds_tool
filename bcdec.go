package main

import (
	"encoding/binary"
	"fmt"
	"math/bits"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------- common

var W2 = [4]int{0, 21, 43, 64}
var W3 = [8]int{0, 9, 18, 27, 37, 46, 55, 64}
var W4 = [16]int{0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64}

// bits128 reads little-endian bit fields from a 16-byte block.
type bits128 struct{ lo, hi uint64 }

func newBits128(b []byte) bits128 {
	return bits128{binary.LittleEndian.Uint64(b), binary.LittleEndian.Uint64(b[8:])}
}

func (b bits128) get(pos, n int) int {
	if n == 0 {
		return 0
	}
	var v uint64
	if pos >= 64 {
		v = b.hi >> uint(pos-64)
	} else {
		v = b.lo >> uint(pos)
		if pos+n > 64 {
			v |= b.hi << uint(64-pos)
		}
	}
	return int(v & ((uint64(1) << uint(n)) - 1))
}

func expand565(c uint16) [3]int {
	r := int(c>>11) & 31
	g := int(c>>5) & 63
	b := int(c) & 31
	return [3]int{r<<3 | r>>2, g<<2 | g>>4, b<<3 | b>>2}
}

func roundHalfEven2(sum int) int { // round(sum/2) with ties to even
	q := sum >> 1
	if sum&1 == 1 && q&1 == 1 {
		q++
	}
	return q
}

// bc1Palette returns 4 RGBA entries (0..255).
func bc1Palette(c0, c1 uint16, fourOnly bool) [4][4]int {
	e0, e1 := expand565(c0), expand565(c1)
	four := c0 > c1 || fourOnly
	var p [4][4]int
	for k := 0; k < 3; k++ {
		p[0][k], p[1][k] = e0[k], e1[k]
		if four {
			p[2][k] = (2*e0[k] + e1[k] + 1) / 3
			p[3][k] = (e0[k] + 2*e1[k] + 1) / 3
		} else {
			p[2][k] = roundHalfEven2(e0[k] + e1[k])
			p[3][k] = 0
		}
	}
	p[0][3], p[1][3], p[2][3] = 255, 255, 255
	if four {
		p[3][3] = 255
	}
	return p
}

func decodeBC1Block(b []byte, fourOnly bool, out *[16][4]float32) {
	c0 := binary.LittleEndian.Uint16(b)
	c1 := binary.LittleEndian.Uint16(b[2:])
	idx := binary.LittleEndian.Uint32(b[4:])
	pal := bc1Palette(c0, c1, fourOnly)
	for i := 0; i < 16; i++ {
		e := pal[(idx>>uint(2*i))&3]
		for k := 0; k < 4; k++ {
			out[i][k] = float32(e[k]) / 255
		}
	}
}

// bc4Palette returns 8 entries in 0..255 (unorm) or -127..127 (snorm) units.
func bc4Palette(r0, r1 int, signed bool) [8]float32 {
	lo, hi := float32(0), float32(255)
	if signed {
		lo, hi = -127, 127
	}
	a, b := float32(r0), float32(r1)
	var p [8]float32
	p[0], p[1] = a, b
	for i := 2; i < 8; i++ {
		if r0 > r1 {
			p[i] = (float32(8-i)*a + float32(i-1)*b) / 7
		} else if i <= 5 {
			p[i] = (float32(6-i)*a + float32(i-1)*b) / 5
		} else if i == 6 {
			p[i] = lo
		} else {
			p[i] = hi
		}
	}
	return p
}

func decodeBC4Block(b []byte, signed bool, out *[16]float32) {
	var r0, r1 int
	if signed {
		r0, r1 = maxi(int(int8(b[0])), -127), maxi(int(int8(b[1])), -127)
	} else {
		r0, r1 = int(b[0]), int(b[1])
	}
	pal := bc4Palette(r0, r1, signed)
	w := readWord(b[2:], 6)
	sc := float32(255)
	if signed {
		sc = 127
	}
	for i := 0; i < 16; i++ {
		out[i] = pal[(w>>uint(3*i))&7] / sc
	}
}

// ---------------------------------------------------------------- BC7

type bc7Mode struct{ ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 int }

var bc7Modes = [8]bc7Mode{
	{3, 4, 0, 0, 4, 0, 1, 0, 3, 0},
	{2, 6, 0, 0, 6, 0, 0, 1, 3, 0},
	{3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
	{2, 6, 0, 0, 7, 0, 1, 0, 2, 0},
	{1, 0, 2, 1, 5, 6, 0, 0, 2, 3},
	{1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
	{1, 0, 0, 0, 7, 7, 1, 0, 4, 0},
	{2, 6, 0, 0, 5, 5, 1, 0, 2, 0},
}

func weightsFor(n int) []int {
	switch n {
	case 2:
		return W2[:]
	case 3:
		return W3[:]
	}
	return W4[:]
}

func subsetOf(ns, part, i int) int {
	switch ns {
	case 2:
		return int(partition2[part][i])
	case 3:
		return int(partition3[part][i])
	}
	return 0
}

func isAnchor(ns, part, i int) bool {
	if i == 0 {
		return true
	}
	switch ns {
	case 2:
		return i == anchor2[part]
	case 3:
		return i == anchor3a[part] || i == anchor3b[part]
	}
	return false
}

func decodeBC7Block(blk []byte, out *[16][4]uint8) {
	b := newBits128(blk)
	mode := bits.TrailingZeros8(blk[0])
	if mode >= 8 {
		*out = [16][4]uint8{}
		return
	}
	m := bc7Modes[mode]
	pos := mode + 1
	part := b.get(pos, m.pb)
	pos += m.pb
	rot := b.get(pos, m.rb)
	pos += m.rb
	isel := b.get(pos, m.isb)
	pos += m.isb
	ne := m.ns * 2
	var ep [6][4]int
	for c := 0; c < 3; c++ {
		for e := 0; e < ne; e++ {
			ep[e][c] = b.get(pos, m.cb)
			pos += m.cb
		}
	}
	for e := 0; e < ne; e++ {
		if m.ab > 0 {
			ep[e][3] = b.get(pos, m.ab)
			pos += m.ab
		} else {
			ep[e][3] = 255
		}
	}
	cbits, abits := m.cb, m.ab
	if m.epb > 0 || m.spb > 0 {
		var pbit [6]int
		if m.epb > 0 {
			for e := 0; e < ne; e++ {
				pbit[e] = b.get(pos, 1)
				pos++
			}
		} else {
			for s := 0; s < m.ns; s++ {
				p := b.get(pos, 1)
				pos++
				pbit[2*s], pbit[2*s+1] = p, p
			}
		}
		for e := 0; e < ne; e++ {
			for c := 0; c < 3; c++ {
				ep[e][c] = ep[e][c]<<1 | pbit[e]
			}
			if m.ab > 0 {
				ep[e][3] = ep[e][3]<<1 | pbit[e]
			}
		}
		cbits++
		if m.ab > 0 {
			abits++
		}
	}
	for e := 0; e < ne; e++ {
		for c := 0; c < 3; c++ {
			ep[e][c] = (ep[e][c] << uint(8-cbits)) | (ep[e][c] >> uint(2*cbits-8))
		}
		if m.ab > 0 {
			ep[e][3] = (ep[e][3] << uint(8-abits)) | (ep[e][3] >> uint(2*abits-8))
		}
	}
	var idx1, idx2 [16]int
	for i := 0; i < 16; i++ {
		n := m.ib
		if isAnchor(m.ns, part, i) {
			n--
		}
		idx1[i] = b.get(pos, n)
		pos += n
	}
	if m.ib2 > 0 {
		for i := 0; i < 16; i++ {
			n := m.ib2
			if i == 0 {
				n--
			}
			idx2[i] = b.get(pos, n)
			pos += n
		}
	}
	for i := 0; i < 16; i++ {
		s := subsetOf(m.ns, part, i)
		e0, e1 := ep[2*s], ep[2*s+1]
		var px [4]int
		if m.ib2 == 0 {
			w := weightsFor(m.ib)[idx1[i]]
			for c := 0; c < 4; c++ {
				px[c] = ((64-w)*e0[c] + w*e1[c] + 32) >> 6
			}
		} else {
			wc := weightsFor(m.ib)[idx1[i]]
			wa := weightsFor(m.ib2)[idx2[i]]
			if isel == 1 {
				wc, wa = wa, wc
			}
			for c := 0; c < 3; c++ {
				px[c] = ((64-wc)*e0[c] + wc*e1[c] + 32) >> 6
			}
			px[3] = ((64-wa)*e0[3] + wa*e1[3] + 32) >> 6
		}
		switch rot {
		case 1:
			px[0], px[3] = px[3], px[0]
		case 2:
			px[1], px[3] = px[3], px[1]
		case 3:
			px[2], px[3] = px[3], px[2]
		}
		for c := 0; c < 4; c++ {
			out[i][c] = uint8(px[c])
		}
	}
}

// ---------------------------------------------------------------- BC6H

type bc6Field struct{ ep, ch, bit int } // ep 0..3 (w,x,y,z), ch 0..2

type bc6Mode struct {
	value       int
	modeBits    int
	layout      []bc6Field
	transformed bool
	twoRegions  bool
	epBits      int
	delta       [3]int
}

var bc6Modes []*bc6Mode

func parseBC6Layout(s string) []bc6Field {
	var out []bc6Field
	for _, tok := range strings.Fields(s) {
		ch := strings.IndexByte("rgb", tok[0])
		ep := int(tok[1] - '0')
		inner := tok[3 : len(tok)-1]
		a, b := inner, inner
		if i := strings.IndexByte(inner, ':'); i >= 0 {
			a, b = inner[:i], inner[i+1:]
		}
		hiBit, _ := strconv.Atoi(a)
		first, _ := strconv.Atoi(b)
		step := 1
		if hiBit < first {
			step = -1
		}
		for k := first; ; k += step {
			out = append(out, bc6Field{ep, ch, k})
			if k == hiBit {
				break
			}
		}
	}
	return out
}

func init() {
	add := func(value, mbits int, transformed, two bool, epb int, d [3]int, layout string) {
		m := &bc6Mode{value, mbits, parseBC6Layout(layout), transformed, two, epb, d}
		want := 65 - mbits
		if two {
			want = 77 - mbits
		}
		if len(m.layout) != want {
			panic(fmt.Sprintf("BC6H mode 0x%x layout has %d bits, want %d", value, len(m.layout), want))
		}
		bc6Modes = append(bc6Modes, m)
	}
	add(0x00, 2, true, true, 10, [3]int{5, 5, 5}, "g2[4] b2[4] b3[4] r0[9:0] g0[9:0] b0[9:0] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]")
	add(0x01, 2, true, true, 7, [3]int{6, 6, 6}, "g2[5] g3[4] g3[5] r0[6:0] b3[0] b3[1] b2[4] g0[6:0] b2[5] b3[2] g2[4] b0[6:0] b3[3] b3[5] b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0]")
	add(0x02, 5, true, true, 11, [3]int{5, 4, 4}, "r0[9:0] g0[9:0] b0[9:0] r1[4:0] r0[10] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] b1[3:0] b0[10] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]")
	add(0x06, 5, true, true, 11, [3]int{4, 5, 4}, "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] g3[4] g2[3:0] g1[4:0] g0[10] g3[3:0] b1[3:0] b0[10] b3[1] b2[3:0] r2[3:0] b3[0] b3[2] r3[3:0] g2[4] b3[3]")
	add(0x0A, 5, true, true, 11, [3]int{4, 4, 5}, "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10] b2[4] g2[3:0] g1[3:0] g0[10] b3[0] g3[3:0] b1[4:0] b0[10] b2[3:0] r2[3:0] b3[1] b3[2] r3[3:0] b3[4] b3[3]")
	add(0x0E, 5, true, true, 9, [3]int{5, 5, 5}, "r0[8:0] b2[4] g0[8:0] g2[4] b0[8:0] b3[4] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]")
	add(0x12, 5, true, true, 8, [3]int{6, 5, 5}, "r0[7:0] g3[4] b2[4] g0[7:0] b3[2] g2[4] b0[7:0] b3[3] b3[4] r1[5:0] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[5:0] r3[5:0]")
	add(0x16, 5, true, true, 8, [3]int{5, 6, 5}, "r0[7:0] b3[0] b2[4] g0[7:0] g2[5] g2[4] b0[7:0] g3[5] b3[4] r1[4:0] g3[4] g2[3:0] g1[5:0] g3[3:0] b1[4:0] b3[1] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]")
	add(0x1A, 5, true, true, 8, [3]int{5, 5, 6}, "r0[7:0] b3[1] b2[4] g0[7:0] b2[5] g2[4] b0[7:0] b3[5] b3[4] r1[4:0] g3[4] g2[3:0] g1[4:0] b3[0] g3[3:0] b1[5:0] b2[3:0] r2[4:0] b3[2] r3[4:0] b3[3]")
	add(0x1E, 5, false, true, 6, [3]int{6, 6, 6}, "r0[5:0] g3[4] b3[0] b3[1] b2[4] g0[5:0] g2[5] b2[5] b3[2] g2[4] b0[5:0] g3[5] b3[3] b3[5] b3[4] r1[5:0] g2[3:0] g1[5:0] g3[3:0] b1[5:0] b2[3:0] r2[5:0] r3[5:0]")
	add(0x03, 5, false, false, 10, [3]int{10, 10, 10}, "r0[9:0] g0[9:0] b0[9:0] r1[9:0] g1[9:0] b1[9:0]")
	add(0x07, 5, true, false, 11, [3]int{9, 9, 9}, "r0[9:0] g0[9:0] b0[9:0] r1[8:0] r0[10] g1[8:0] g0[10] b1[8:0] b0[10]")
	add(0x0B, 5, true, false, 12, [3]int{8, 8, 8}, "r0[9:0] g0[9:0] b0[9:0] r1[7:0] r0[10:11] g1[7:0] g0[10:11] b1[7:0] b0[10:11]")
	add(0x0F, 5, true, false, 16, [3]int{4, 4, 4}, "r0[9:0] g0[9:0] b0[9:0] r1[3:0] r0[10:15] g1[3:0] g0[10:15] b1[3:0] b0[10:15]")
}

func findBC6Mode(b bits128) *bc6Mode {
	m2 := b.get(0, 2)
	m5 := b.get(0, 5)
	for _, m := range bc6Modes {
		if (m.modeBits == 2 && m2 == m.value) || (m.modeBits == 5 && m2 >= 2 && m5 == m.value) {
			return m
		}
	}
	return nil
}

func sext(v, nbits int) int {
	if nbits >= 32 {
		return v
	}
	v &= (1 << uint(nbits)) - 1
	if v&(1<<uint(nbits-1)) != 0 {
		v -= 1 << uint(nbits)
	}
	return v
}

func bc6Unquantize(x, nbits int, signed bool) int {
	if !signed {
		if nbits >= 15 {
			return x
		}
		if x == 0 {
			return 0
		}
		if x == (1<<uint(nbits))-1 {
			return 0xFFFF
		}
		return ((x << 16) + 0x8000) >> uint(nbits)
	}
	if nbits >= 16 {
		return x
	}
	s := x < 0
	if s {
		x = -x
	}
	var u int
	if x == 0 {
		u = 0
	} else if x >= (1<<uint(nbits-1))-1 {
		u = 0x7FFF
	} else {
		u = ((x << 15) + 0x4000) >> uint(nbits-1)
	}
	if s {
		return -u
	}
	return u
}

func bc6Finish(v int, signed bool) uint16 {
	if !signed {
		return uint16((v * 31) >> 6)
	}
	if v < 0 {
		return 0x8000 | uint16(((-v)*31)>>5)
	}
	return uint16((v * 31) >> 5)
}

func decodeBC6HBlock(blk []byte, signed bool, out *[16][3]float32) {
	b := newBits128(blk)
	m := findBC6Mode(b)
	if m == nil {
		*out = [16][3]float32{}
		return
	}
	var e [4][3]int
	pos := m.modeBits
	for _, f := range m.layout {
		e[f.ep][f.ch] |= b.get(pos, 1) << uint(f.bit)
		pos++
	}
	part := 0
	nep := 2
	if m.twoRegions {
		part = b.get(pos, 5)
		pos += 5
		nep = 4
	}
	for c := 0; c < 3; c++ {
		if signed {
			e[0][c] = sext(e[0][c], m.epBits)
		}
		if m.transformed || signed {
			for i := 1; i < nep; i++ {
				e[i][c] = sext(e[i][c], m.delta[c])
			}
		}
		if m.transformed {
			for i := 1; i < nep; i++ {
				e[i][c] = (e[0][c] + e[i][c]) & ((1 << uint(m.epBits)) - 1)
				if signed {
					e[i][c] = sext(e[i][c], m.epBits)
				}
			}
		}
		for i := 0; i < nep; i++ {
			e[i][c] = bc6Unquantize(e[i][c], m.epBits, signed)
		}
	}
	for i := 0; i < 16; i++ {
		var w, s int
		if m.twoRegions {
			s = int(partition2[part][i])
			n := 3
			if i == 0 || i == anchor2[part] {
				n = 2
			}
			w = W3[b.get(pos, n)]
			pos += n
		} else {
			n := 4
			if i == 0 {
				n = 3
			}
			w = W4[b.get(pos, n)]
			pos += n
		}
		for c := 0; c < 3; c++ {
			v := (e[2*s][c]*(64-w) + e[2*s+1][c]*w + 32) >> 6
			out[i][c] = halfToFloat(bc6Finish(v, signed))
		}
	}
}

// ---------------------------------------------------------------- surfaces

// bcDecode returns an image with channels as in f.ValueChannels().
func bcDecode(buf []byte, f *Fmt, w, h int) (*Img, error) {
	bw, bh := maxi(1, (w+3)/4), maxi(1, (h+3)/4)
	bs := f.BlockBytes()
	if len(buf) < bw*bh*bs {
		return nil, fmt.Errorf("The DDS file is truncated (texture data missing)")
	}
	C := len(f.ValueChannels())
	img := NewImg(w, h, C, false)
	put := func(bx, by, i, c int, v float32) {
		x, y := bx*4+i%4, by*4+i/4
		if x < w && y < h {
			img.F32[(y*w+x)*C+c] = v
		}
	}
	parallelRows(bh, func(y0, y1 int) {
		var rgba [16][4]float32
		var u8 [16][4]uint8
		var hdr [16][3]float32
		var a, g [16]float32
		for by := y0; by < y1; by++ {
			for bx := 0; bx < bw; bx++ {
				blk := buf[(by*bw+bx)*bs:]
				switch f.BC {
				case 1:
					decodeBC1Block(blk, false, &rgba)
				case 2:
					decodeBC1Block(blk[8:], true, &rgba)
					aw := binary.LittleEndian.Uint64(blk)
					for i := 0; i < 16; i++ {
						rgba[i][3] = float32((aw>>uint(4*i))&15) / 15
					}
				case 3:
					decodeBC1Block(blk[8:], true, &rgba)
					decodeBC4Block(blk, false, &a)
					for i := 0; i < 16; i++ {
						rgba[i][3] = a[i]
					}
				case 4:
					decodeBC4Block(blk, f.Signed, &a)
					for i := 0; i < 16; i++ {
						rgba[i][0] = a[i]
					}
				case 5:
					decodeBC4Block(blk, f.Signed, &a)
					decodeBC4Block(blk[8:], f.Signed, &g)
					for i := 0; i < 16; i++ {
						rgba[i][0], rgba[i][1] = a[i], g[i]
					}
				case 6:
					decodeBC6HBlock(blk, f.Signed, &hdr)
					for i := 0; i < 16; i++ {
						rgba[i][0], rgba[i][1], rgba[i][2] = hdr[i][0], hdr[i][1], hdr[i][2]
					}
				case 7:
					decodeBC7Block(blk, &u8)
					for i := 0; i < 16; i++ {
						for c := 0; c < 4; c++ {
							rgba[i][c] = float32(u8[i][c]) / 255
						}
					}
				}
				for i := 0; i < 16; i++ {
					for c := 0; c < C; c++ {
						put(bx, by, i, c, rgba[i][c])
					}
				}
			}
		}
	})
	return img, nil
}
