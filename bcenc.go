package main

import (
	"encoding/binary"
	"math"
)

type EncOpts struct {
	BC1Alpha bool
	Quality  int // 0 = fast, 1 = normal, 2 = best (BC7)
}

type vec4 [4]float64

// ---------------------------------------------------------------- shared math

// pcaEndpoints returns endpoints along the principal axis of the masked pixels.
func pcaEndpoints(px *[16]vec4, mask *[16]bool, dim int) (vec4, vec4) {
	var mean vec4
	cnt := 0.0
	for i := 0; i < 16; i++ {
		if mask[i] {
			cnt++
			for c := 0; c < dim; c++ {
				mean[c] += px[i][c]
			}
		}
	}
	if cnt < 1 {
		cnt = 1
	}
	for c := 0; c < dim; c++ {
		mean[c] /= cnt
	}
	var cov [4][4]float64
	for i := 0; i < 16; i++ {
		if !mask[i] {
			continue
		}
		var d vec4
		for c := 0; c < dim; c++ {
			d[c] = px[i][c] - mean[c]
		}
		for a := 0; a < dim; a++ {
			for b := 0; b < dim; b++ {
				cov[a][b] += d[a] * d[b]
			}
		}
	}
	k := 0
	for c := 1; c < dim; c++ {
		if cov[c][c] > cov[k][k] {
			k = c
		}
	}
	axis := vec4(cov[k])
	for it := 0; it < 6; it++ {
		var na vec4
		nrm := 0.0
		for a := 0; a < dim; a++ {
			for b := 0; b < dim; b++ {
				na[a] += cov[a][b] * axis[b]
			}
			nrm += na[a] * na[a]
		}
		nrm = math.Max(math.Sqrt(nrm), 1e-12)
		for a := 0; a < dim; a++ {
			axis[a] = na[a] / nrm
		}
	}
	tmin, tmax := math.Inf(1), math.Inf(-1)
	for i := 0; i < 16; i++ {
		if !mask[i] {
			continue
		}
		t := 0.0
		for c := 0; c < dim; c++ {
			t += (px[i][c] - mean[c]) * axis[c]
		}
		tmin = math.Min(tmin, t)
		tmax = math.Max(tmax, t)
	}
	if math.IsInf(tmin, 0) {
		tmin, tmax = 0, 0
	}
	var e0, e1 vec4
	for c := 0; c < dim; c++ {
		e0[c] = mean[c] + tmin*axis[c]
		e1[c] = mean[c] + tmax*axis[c]
	}
	return e0, e1
}

// lsEndpoints fits endpoints by least squares for known interpolation weights t.
func lsEndpoints(px *[16]vec4, t *[16]float64, mask *[16]bool, dim int, f0, f1 vec4) (vec4, vec4) {
	var a, b, c float64
	var r0, r1 vec4
	for i := 0; i < 16; i++ {
		if !mask[i] {
			continue
		}
		s := 1 - t[i]
		a += s * s
		b += s * t[i]
		c += t[i] * t[i]
		for k := 0; k < dim; k++ {
			r0[k] += s * px[i][k]
			r1[k] += t[i] * px[i][k]
		}
	}
	det := a*c - b*b
	if math.Abs(det) <= 1e-8 {
		return f0, f1
	}
	var n0, n1 vec4
	for k := 0; k < dim; k++ {
		n0[k] = (c*r0[k] - b*r1[k]) / det
		n1[k] = (a*r1[k] - b*r0[k]) / det
	}
	return n0, n1
}

// assign picks the nearest palette entry for each pixel.
func assign(px *[16]vec4, pal []vec4, mask *[16]bool, dim int, idx *[16]int) float64 {
	total := 0.0
	for i := 0; i < 16; i++ {
		best, bi := math.Inf(1), 0
		for k := range pal {
			d := 0.0
			for c := 0; c < dim; c++ {
				x := px[i][c] - pal[k][c]
				d += x * x
			}
			if d < best {
				best, bi = d, k
			}
		}
		idx[i] = bi
		if mask == nil || mask[i] {
			total += best
		}
	}
	return total
}

var allMask = [16]bool{true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true}

func put128(lo, hi *uint64, pos, n int, val int) {
	v := uint64(val) & ((uint64(1) << uint(n)) - 1)
	switch {
	case pos+n <= 64:
		*lo |= v << uint(pos)
	case pos >= 64:
		*hi |= v << uint(pos-64)
	default:
		k := 64 - pos
		*lo |= (v & ((uint64(1) << uint(k)) - 1)) << uint(pos)
		*hi |= v >> uint(k)
	}
}

func block128(lo, hi uint64, out []byte) {
	binary.LittleEndian.PutUint64(out, lo)
	binary.LittleEndian.PutUint64(out[8:], hi)
}

// ---------------------------------------------------------------- BC1 color

func q565(e vec4) uint16 {
	r := int(rint(clamp(e[0], 0, 255) * 31 / 255))
	g := int(rint(clamp(e[1], 0, 255) * 63 / 255))
	b := int(rint(clamp(e[2], 0, 255) * 31 / 255))
	return uint16(r<<11 | g<<5 | b)
}

func expand565v(c uint16) vec4 {
	e := expand565(c)
	return vec4{float64(e[0]), float64(e[1]), float64(e[2]), 0}
}

var T4 = [4]float64{0, 1, 1.0 / 3, 2.0 / 3}
var T3 = [4]float64{0, 1, 0.5, 0}

type bc1Cand struct {
	c0, c1 uint16
	idx    [16]int
	err    float64
}

// encodeBC1Color: px in 0..255. transparent may be nil.
func encodeBC1Color(px *[16]vec4, transparent *[16]bool, out []byte) {
	mask := allMask
	three := false
	if transparent != nil {
		for i := 0; i < 16; i++ {
			mask[i] = !transparent[i]
			three = three || transparent[i]
		}
	}
	evaluate := func(e0, e1 vec4) bc1Cand {
		c0, c1 := q565(e0), q565(e1)
		hi, lo := c0, c1
		if lo > hi {
			hi, lo = lo, hi
		}
		var cd bc1Cand
		var pal []vec4
		if three {
			p := bc1Palette(lo, hi, false)
			pal = []vec4{{float64(p[0][0]), float64(p[0][1]), float64(p[0][2])},
				{float64(p[1][0]), float64(p[1][1]), float64(p[1][2])},
				{float64(p[2][0]), float64(p[2][1]), float64(p[2][2])}}
			cd.c0, cd.c1 = lo, hi
		} else {
			p := bc1Palette(hi, lo, true)
			pal = make([]vec4, 4)
			for k := 0; k < 4; k++ {
				pal[k] = vec4{float64(p[k][0]), float64(p[k][1]), float64(p[k][2])}
			}
			cd.c0, cd.c1 = hi, lo
		}
		cd.err = assign(px, pal, &mask, 3, &cd.idx)
		return cd
	}
	e0, e1 := pcaEndpoints(px, &mask, 3)
	best := evaluate(e0, e1)
	for it := 0; it < 2; it++ {
		var t [16]float64
		for i := 0; i < 16; i++ {
			if three {
				t[i] = T3[best.idx[i]]
			} else {
				t[i] = T4[best.idx[i]]
			}
		}
		a, b := lsEndpoints(px, &t, &mask, 3, expand565v(best.c0), expand565v(best.c1))
		if cand := evaluate(a, b); cand.err < best.err {
			best = cand
		}
	}
	idx := best.idx
	if transparent != nil {
		for i := 0; i < 16; i++ {
			if transparent[i] {
				idx[i] = 3
			}
		}
	}
	if best.c0 == best.c1 && !three {
		idx = [16]int{}
	}
	var word uint32
	for i := 0; i < 16; i++ {
		word |= uint32(idx[i]) << uint(2*i)
	}
	binary.LittleEndian.PutUint16(out, best.c0)
	binary.LittleEndian.PutUint16(out[2:], best.c1)
	binary.LittleEndian.PutUint32(out[4:], word)
}

// ---------------------------------------------------------------- BC4

var TT8 = [8]float64{0, 1, 1.0 / 7, 2.0 / 7, 3.0 / 7, 4.0 / 7, 5.0 / 7, 6.0 / 7}

// encodeBC4: px in 0..255 (unorm) or -127..127 (snorm).
func encodeBC4(in *[16]float64, signed bool, out []byte) {
	lo, hi := 0.0, 255.0
	if signed {
		lo, hi = -127, 127
	}
	var px [16]float64
	for i := range in {
		px[i] = clamp(in[i], lo, hi)
	}
	type cand struct {
		r0, r1 int
		idx    [16]int
		err    float64
	}
	evaluate := func(a, b float64, six bool) cand {
		ai, bi := int(clamp(rint(a), lo, hi)), int(clamp(rint(b), lo, hi))
		var c cand
		if six {
			c.r0, c.r1 = mini(ai, bi), maxi(ai, bi)
		} else {
			c.r0, c.r1 = maxi(ai, bi), mini(ai, bi)
		}
		pal := bc4Palette(c.r0, c.r1, signed)
		for i := 0; i < 16; i++ {
			best, bk := math.Inf(1), 0
			for k := 0; k < 8; k++ {
				d := px[i] - float64(pal[k])
				d *= d
				if d < best {
					best, bk = d, k
				}
			}
			c.idx[i] = bk
			c.err += best
		}
		return c
	}
	mn, mx := px[0], px[0]
	for _, v := range px {
		mn, mx = math.Min(mn, v), math.Max(mx, v)
	}
	best := evaluate(mx, mn, false)
	var p4 [16]vec4
	var t [16]float64
	for i := 0; i < 16; i++ {
		p4[i][0] = px[i]
		t[i] = TT8[best.idx[i]]
	}
	e0, e1 := lsEndpoints(&p4, &t, &allMask, 1, vec4{float64(best.r0)}, vec4{float64(best.r1)})
	if c := evaluate(e0[0], e1[0], false); c.err < best.err {
		best = c
	}
	// 6-value mode (with exact extremes) - useful for alpha masks
	imn, imx := math.Inf(1), math.Inf(-1)
	for _, v := range px {
		if v > lo && v < hi {
			imn, imx = math.Min(imn, v), math.Max(imx, v)
		}
	}
	if math.IsInf(imn, 0) {
		imn, imx = mn, mx
	}
	if c := evaluate(imn, imx, true); c.err < best.err {
		best = c
	}
	var word uint64
	for i := 0; i < 16; i++ {
		word |= uint64(best.idx[i]) << uint(3*i)
	}
	out[0], out[1] = byte(int8(best.r0)), byte(int8(best.r1))
	if !signed {
		out[0], out[1] = byte(best.r0), byte(best.r1)
	}
	for k := 0; k < 6; k++ {
		out[2+k] = byte(word >> uint(8*k))
	}
}

// ---------------------------------------------------------------- BC6H (modes 11 and 12)

func bc6Quant(f float64, nbits int, signed bool) int {
	if !signed {
		u := math.Max(f, 0) * 64 / 31
		x := rint((u - math.Exp2(float64(15-nbits))) / math.Exp2(float64(16-nbits)))
		return int(clamp(x, 0, float64(int(1)<<uint(nbits)-1)))
	}
	u := math.Abs(f) * 32 / 31
	b := nbits - 1
	x := clamp(rint((u-math.Exp2(float64(14-b)))/math.Exp2(float64(15-b))), 0, float64(int(1)<<uint(b)-1))
	if f < 0 {
		return -int(x)
	}
	return int(x)
}

func bc6PaletteEnc(x0, x1 [3]int, nbits int, signed bool) []vec4 {
	pal := make([]vec4, 16)
	for c := 0; c < 3; c++ {
		u0, u1 := bc6Unquantize(x0[c], nbits, signed), bc6Unquantize(x1[c], nbits, signed)
		for k := 0; k < 16; k++ {
			w := W4[k]
			v := (u0*(64-w) + u1*w + 32) >> 6
			if !signed {
				v = (v * 31) >> 6
			} else if v < 0 {
				v = -(((-v) * 31) >> 5)
			} else {
				v = (v * 31) >> 5
			}
			pal[k][c] = float64(v)
		}
	}
	return pal
}

func encodeBC6H(in *[16]vec4, signed bool, out []byte) {
	var f [16]vec4
	for i := 0; i < 16; i++ {
		for c := 0; c < 3; c++ {
			v := in[i][c]
			switch {
			case v != v:
				v = 0
			case math.IsInf(v, 1):
				v = 65504
			case math.IsInf(v, -1):
				v = -65504
			}
			if signed {
				v = clamp(v, -65504, 65504)
			} else {
				v = clamp(v, 0, 65504)
			}
			h := floatToHalf(float32(v))
			if h&0x8000 != 0 {
				f[i][c] = -float64(h & 0x7FFF)
			} else {
				f[i][c] = float64(h)
			}
		}
	}
	type cand struct {
		x0, x1 [3]int
		idx    [16]int
		err    float64
		mode   int
	}
	evaluate := func(e0, e1 vec4, mode int) cand {
		c := cand{mode: mode}
		bitsN := 10
		if mode == 12 {
			bitsN = 11
		}
		for k := 0; k < 3; k++ {
			c.x0[k], c.x1[k] = bc6Quant(e0[k], bitsN, signed), bc6Quant(e1[k], bitsN, signed)
			if mode == 12 {
				lim, low := 2047, 0
				if signed {
					lim, low = 1023, -1023
				}
				d := maxi(-255, mini(255, c.x1[k]-c.x0[k]))
				c.x1[k] = maxi(low, mini(lim, c.x0[k]+d))
			}
		}
		c.err = assign(&f, bc6PaletteEnc(c.x0, c.x1, bitsN, signed), nil, 3, &c.idx)
		return c
	}
	e0, e1 := pcaEndpoints(&f, &allMask, 3)
	var best cand
	for mi, mode := range []int{11, 12} {
		cur := evaluate(e0, e1, mode)
		for it := 0; it < 2; it++ {
			var t [16]float64
			for i := 0; i < 16; i++ {
				t[i] = float64(W4[cur.idx[i]]) / 64
			}
			a, b := lsEndpoints(&f, &t, &allMask, 3, e0, e1)
			if c := evaluate(a, b, mode); c.err < cur.err {
				cur = c
			}
		}
		if mi == 0 || cur.err < best.err {
			best = cur
		}
	}
	// pixel 0 (anchor) must have an index < 8
	x0, x1, idx := best.x0, best.x1, best.idx
	if idx[0] >= 8 {
		x0, x1 = x1, x0
		for i := range idx {
			idx[i] = 15 - idx[i]
		}
	}
	var lo, hi uint64
	if best.mode == 11 {
		put128(&lo, &hi, 0, 5, 0x03)
		for c := 0; c < 3; c++ {
			put128(&lo, &hi, 5+10*c, 10, x0[c]&0x3FF)
			put128(&lo, &hi, 35+10*c, 10, x1[c]&0x3FF)
		}
	} else {
		put128(&lo, &hi, 0, 5, 0x07)
		for c := 0; c < 3; c++ {
			w := x0[c] & 0x7FF
			put128(&lo, &hi, 5+10*c, 10, w&0x3FF)
			put128(&lo, &hi, 35+10*c, 9, (x1[c]-x0[c])&0x1FF)
			put128(&lo, &hi, 44+10*c, 1, w>>10)
		}
	}
	pos := 65
	for i := 0; i < 16; i++ {
		n := 4
		if i == 0 {
			n = 3
		}
		put128(&lo, &hi, pos, n, idx[i])
		pos += n
	}
	block128(lo, hi, out)
}

// ---------------------------------------------------------------- whole surfaces

// bcEncode encodes an image whose channels follow f.ValueChannels().
func bcEncode(img *Img, f *Fmt, opts EncOpts) []byte {
	w, h := img.W, img.H
	bw, bh := maxi(1, (w+3)/4), maxi(1, (h+3)/4)
	bs := f.BlockBytes()
	out := make([]byte, bw*bh*bs)
	C := img.C
	n := f.BC
	parallelRows(bh, func(y0, y1 int) {
		var px [16]vec4
		var ch [16]float64
		var tr [16]bool
		for by := y0; by < y1; by++ {
			for bx := 0; bx < bw; bx++ {
				for i := 0; i < 16; i++ {
					x := mini(bx*4+i%4, w-1)
					y := mini(by*4+i/4, h-1)
					base := (y*w + x) * C
					px[i] = vec4{0, 0, 0, 1}
					for c := 0; c < C && c < 4; c++ {
						px[i][c] = img.Get(base + c)
					}
					switch n {
					case 1, 2, 3, 7:
						for c := 0; c < 4; c++ {
							px[i][c] = clamp(px[i][c], 0, 1) * 255
						}
					case 4, 5:
						sc := 255.0
						if f.Signed {
							sc = 127
						}
						px[i][0] *= sc
						px[i][1] *= sc
					}
				}
				dst := out[(by*bw+bx)*bs:]
				switch n {
				case 1:
					var t *[16]bool
					if opts.BC1Alpha {
						for i := 0; i < 16; i++ {
							tr[i] = px[i][3] < 128
						}
						t = &tr
					}
					encodeBC1Color(&px, t, dst)
				case 2:
					var word uint64
					for i := 0; i < 16; i++ {
						a4 := uint64(clamp(rint(px[i][3]*15/255), 0, 15))
						word |= a4 << uint(4*i)
					}
					binary.LittleEndian.PutUint64(dst, word)
					encodeBC1Color(&px, nil, dst[8:])
				case 3:
					for i := 0; i < 16; i++ {
						ch[i] = px[i][3]
					}
					encodeBC4(&ch, false, dst)
					encodeBC1Color(&px, nil, dst[8:])
				case 4:
					for i := 0; i < 16; i++ {
						ch[i] = px[i][0]
					}
					encodeBC4(&ch, f.Signed, dst)
				case 5:
					for i := 0; i < 16; i++ {
						ch[i] = px[i][0]
					}
					encodeBC4(&ch, f.Signed, dst)
					for i := 0; i < 16; i++ {
						ch[i] = px[i][1]
					}
					encodeBC4(&ch, f.Signed, dst[8:])
				case 6:
					encodeBC6H(&px, f.Signed, dst)
				case 7:
					encodeBC7(&px, opts.Quality, dst)
				}
			}
		}
	})
	return out
}
