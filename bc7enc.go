package main

import (
	"math"
)

// BC7 encoder: tries several modes (and partitions) per block and keeps the best one.

type lineFit struct {
	codes [2][4]int
	pb    [2]int
	ex    [2]vec4
	idx   [16]int
	err   float64
}

func expandBits(v, n int) int {
	if n >= 8 {
		return v
	}
	return (v << uint(8-n)) | (v >> uint(2*n-8))
}

// quantizeEndpoint quantizes e (0..255 per channel) to the given bit counts, with optional p-bit.
func quantizeEndpoint(e vec4, dim int, bits [4]int, p int, hasP bool) ([4]int, vec4, float64) {
	var codes [4]int
	var ex vec4
	qerr := 0.0
	for c := 0; c < dim; c++ {
		b := bits[c]
		n := b
		if hasP {
			n++
		}
		v := clamp(e[c], 0, 255) * float64(int(1)<<uint(n)-1) / 255
		var guess int
		if hasP {
			guess = int(rint((v - float64(p)) / 2))
		} else {
			guess = int(rint(v))
		}
		bestD, bestC, bestX := math.Inf(1), 0, 0
		for code := guess - 1; code <= guess+1; code++ {
			if code < 0 || code > (1<<uint(b))-1 {
				continue
			}
			full := code
			if hasP {
				full = code<<1 | p
			}
			x := expandBits(full, n)
			d := math.Abs(float64(x) - e[c])
			if d < bestD {
				bestD, bestC, bestX = d, code, x
			}
		}
		codes[c], ex[c] = bestC, float64(bestX)
		qerr += bestD * bestD
	}
	return codes, ex, qerr
}

// fitLine fits one endpoint pair to the masked pixels.
// pmode: 0 = no p-bit, 1 = one p-bit per endpoint, 2 = p-bit shared by both endpoints.
func fitLine(px *[16]vec4, mask *[16]bool, dim int, bits [4]int, pmode int, ib int, iters int) lineFit {
	wts := weightsFor(ib)
	np := 1 << uint(ib)
	evaluate := func(e0, e1 vec4) lineFit {
		var f lineFit
		switch pmode {
		case 0:
			f.codes[0], f.ex[0], _ = quantizeEndpoint(e0, dim, bits, 0, false)
			f.codes[1], f.ex[1], _ = quantizeEndpoint(e1, dim, bits, 0, false)
		case 1:
			for k, e := range []vec4{e0, e1} {
				c0, x0, q0 := quantizeEndpoint(e, dim, bits, 0, true)
				c1, x1, q1 := quantizeEndpoint(e, dim, bits, 1, true)
				if q1 < q0 {
					f.codes[k], f.ex[k], f.pb[k] = c1, x1, 1
				} else {
					f.codes[k], f.ex[k], f.pb[k] = c0, x0, 0
				}
			}
		case 2:
			bestQ := math.Inf(1)
			for p := 0; p < 2; p++ {
				ca, xa, qa := quantizeEndpoint(e0, dim, bits, p, true)
				cb, xb, qb := quantizeEndpoint(e1, dim, bits, p, true)
				if qa+qb < bestQ {
					bestQ = qa + qb
					f.codes = [2][4]int{ca, cb}
					f.ex = [2]vec4{xa, xb}
					f.pb = [2]int{p, p}
				}
			}
		}
		var pal [16]vec4
		for k := 0; k < np; k++ {
			w := float64(wts[k])
			for c := 0; c < dim; c++ {
				pal[k][c] = math.Floor(((64-w)*f.ex[0][c] + w*f.ex[1][c] + 32) / 64)
			}
		}
		f.err = assignLine(px, pal[:np], mask, dim, &f.idx, f.ex[0], f.ex[1], wts)
		return f
	}
	e0, e1 := pcaEndpoints(px, mask, dim)
	best := evaluate(e0, e1)
	for it := 0; it < iters; it++ {
		var t [16]float64
		for i := 0; i < 16; i++ {
			t[i] = float64(wts[best.idx[i]]) / 64
		}
		a, b := lsEndpoints(px, &t, mask, dim, e0, e1)
		c := evaluate(a, b)
		if c.err < best.err {
			best = c
		} else {
			break
		}
	}
	return best
}

type bc7Cand struct {
	mode, part, rot, isel int
	codes                 [6][4]int // per endpoint: R,G,B,A codes
	pb                    [6]int
	idx1, idx2            [16]int // single-index modes use idx1; modes 4/5: idx1 = color, idx2 = alpha
	err                   float64
}

type bc7Params struct {
	ns, cb, ab, pmode, ib, dim int
}

var bc7P = map[int]bc7Params{
	0: {3, 4, 0, 1, 3, 3},
	1: {2, 6, 0, 2, 3, 3},
	2: {3, 5, 0, 0, 2, 3},
	3: {2, 7, 0, 1, 2, 3},
	6: {1, 7, 7, 1, 4, 4},
	7: {2, 5, 5, 1, 2, 4},
}

func encodeSingleIndexMode(px *[16]vec4, mode, part, iters int) bc7Cand {
	p := bc7P[mode]
	c := bc7Cand{mode: mode, part: part}
	bits := [4]int{p.cb, p.cb, p.cb, p.ab}
	for s := 0; s < p.ns; s++ {
		var mask [16]bool
		any := false
		for i := 0; i < 16; i++ {
			mask[i] = subsetOf(p.ns, part, i) == s
			any = any || mask[i]
		}
		if !any {
			continue
		}
		f := fitLine(px, &mask, p.dim, bits, p.pmode, p.ib, iters)
		c.codes[2*s], c.codes[2*s+1] = f.codes[0], f.codes[1]
		c.pb[2*s], c.pb[2*s+1] = f.pb[0], f.pb[1]
		for i := 0; i < 16; i++ {
			if mask[i] {
				c.idx1[i] = f.idx[i]
			}
		}
		c.err += f.err
	}
	return c
}

// encodeSeparateAlphaMode handles modes 4 and 5 (independent color and alpha indices).
func encodeSeparateAlphaMode(px *[16]vec4, mode, rot, isel, iters int) bc7Cand {
	var r [16]vec4
	var a [16]vec4
	for i := 0; i < 16; i++ {
		r[i] = px[i]
		if rot > 0 {
			r[i][rot-1], r[i][3] = px[i][3], px[i][rot-1]
		}
		a[i][0] = r[i][3]
	}
	cb, ab, cib, aib := 7, 8, 2, 2
	if mode == 4 {
		cb, ab, cib, aib = 5, 6, 2, 3
		if isel == 1 {
			cib, aib = 3, 2
		}
	}
	fc := fitLine(&r, &allMask, 3, [4]int{cb, cb, cb, 0}, 0, cib, iters)
	fa := fitLine(&a, &allMask, 1, [4]int{ab}, 0, aib, iters)
	c := bc7Cand{mode: mode, rot: rot, isel: isel, err: fc.err + fa.err}
	for k := 0; k < 2; k++ {
		c.codes[k] = [4]int{fc.codes[k][0], fc.codes[k][1], fc.codes[k][2], fa.codes[k][0]}
	}
	c.idx1, c.idx2 = fc.idx, fa.idx
	return c
}

// rankPartitions orders partitions by an estimate of the line-fit error
// (sum over subsets of the variance not explained by the principal axis).
// Statistics are always 4D; constant channels (e.g. opaque alpha) contribute nothing.
func rankPartitions(px *[16]vec4, dim, ns, count, keep int) []int {
	// per-pixel moments: x0..x3, then the 10 products of the upper triangle
	type mom [15]float64
	var pix [16]mom
	var all mom
	for i := 0; i < 16; i++ {
		x := px[i]
		pix[i] = mom{1, x[0], x[1], x[2], x[3],
			x[0] * x[0], x[0] * x[1], x[0] * x[2], x[0] * x[3],
			x[1] * x[1], x[1] * x[2], x[1] * x[3],
			x[2] * x[2], x[2] * x[3], x[3] * x[3]}
		for k := range all {
			all[k] += pix[i][k]
		}
	}
	residual := func(m *mom) float64 {
		n := m[0]
		if n < 2 {
			return 0
		}
		inv := 1 / n
		c00 := m[5] - m[1]*m[1]*inv
		c01 := m[6] - m[1]*m[2]*inv
		c02 := m[7] - m[1]*m[3]*inv
		c03 := m[8] - m[1]*m[4]*inv
		c11 := m[9] - m[2]*m[2]*inv
		c12 := m[10] - m[2]*m[3]*inv
		c13 := m[11] - m[2]*m[4]*inv
		c22 := m[12] - m[3]*m[3]*inv
		c23 := m[13] - m[3]*m[4]*inv
		c33 := m[14] - m[4]*m[4]*inv
		tr := c00 + c11 + c22 + c33
		// start from the row with the largest diagonal entry
		v0, v1, v2, v3 := c00, c01, c02, c03
		if c11 > c00 && c11 >= c22 && c11 >= c33 {
			v0, v1, v2, v3 = c01, c11, c12, c13
		} else if c22 > c00 && c22 > c11 && c22 >= c33 {
			v0, v1, v2, v3 = c02, c12, c22, c23
		} else if c33 > c00 && c33 > c11 && c33 > c22 {
			v0, v1, v2, v3 = c03, c13, c23, c33
		}
		lam := 0.0
		for it := 0; it < 3; it++ {
			n0 := c00*v0 + c01*v1 + c02*v2 + c03*v3
			n1 := c01*v0 + c11*v1 + c12*v2 + c13*v3
			n2 := c02*v0 + c12*v1 + c22*v2 + c23*v3
			n3 := c03*v0 + c13*v1 + c23*v2 + c33*v3
			vn := v0*v0 + v1*v1 + v2*v2 + v3*v3
			nn := n0*n0 + n1*n1 + n2*n2 + n3*n3
			if vn < 1e-18 || nn < 1e-18 {
				return math.Max(tr, 0)
			}
			lam = math.Sqrt(nn / vn)
			s := 1 / math.Sqrt(nn)
			v0, v1, v2, v3 = n0*s, n1*s, n2*s, n3*s
		}
		return math.Max(tr-lam, 0)
	}
	type pe struct {
		p int
		e float64
	}
	best := make([]pe, 0, keep+1)
	for p := 0; p < count; p++ {
		// accumulate subsets 1..ns-1; subset 0 = whole block minus the others
		var st [3]mom
		for i := 0; i < 16; i++ {
			k := subsetOf(ns, p, i)
			if k == 0 {
				continue
			}
			s, q := &st[k], &pix[i]
			for j := range s {
				s[j] += q[j]
			}
		}
		st[0] = all
		for k := 1; k < ns; k++ {
			for j := range st[0] {
				st[0][j] -= st[k][j]
			}
		}
		total := 0.0
		for k := 0; k < ns; k++ {
			total += residual(&st[k])
		}
		if len(best) == keep && total >= best[keep-1].e {
			continue
		}
		j := len(best)
		best = append(best, pe{p, total})
		for j > 0 && best[j-1].e > total {
			best[j] = best[j-1]
			j--
		}
		best[j] = pe{p, total}
		if len(best) > keep {
			best = best[:keep]
		}
	}
	out := make([]int, len(best))
	for i, b := range best {
		out[i] = b.p
	}
	return out
}

func encodeBC7(px *[16]vec4, quality int, out []byte) {
	opaque := true
	for i := 0; i < 16; i++ {
		if px[i][3] < 254.5 {
			opaque = false
			break
		}
	}
	iters := 2
	if quality >= 2 {
		iters = 4
	}
	best := encodeSingleIndexMode(px, 6, 0, iters)
	try := func(c bc7Cand) {
		if c.err < best.err {
			best = c
		}
	}
	// skip the slower modes when mode 6 is already nearly lossless
	if quality >= 1 && best.err > 16*1.0 {
		k2, k3 := 4, 0
		if quality >= 2 {
			k2, k3 = 16, 6
		}
		if opaque {
			var opx [16]vec4
			for i := range px {
				opx[i] = px[i]
				opx[i][3] = 255
			}
			for _, p := range rankPartitions(&opx, 3, 2, 64, k2) {
				try(encodeSingleIndexMode(&opx, 1, p, iters))
				try(encodeSingleIndexMode(&opx, 3, p, iters))
			}
			if k3 > 0 {
				for _, p := range rankPartitions(&opx, 3, 3, 16, k3) {
					try(encodeSingleIndexMode(&opx, 0, p, iters))
				}
				for _, p := range rankPartitions(&opx, 3, 3, 64, k3) {
					try(encodeSingleIndexMode(&opx, 2, p, iters))
				}
			}
			if quality >= 2 {
				for rot := 0; rot < 4; rot++ {
					try(encodeSeparateAlphaMode(px, 5, rot, 0, iters))
					try(encodeSeparateAlphaMode(px, 4, rot, 0, iters))
					try(encodeSeparateAlphaMode(px, 4, rot, 1, iters))
				}
			}
		} else {
			for _, p := range rankPartitions(px, 4, 2, 64, k2) {
				try(encodeSingleIndexMode(px, 7, p, iters))
			}
			rots := 1
			if quality >= 2 {
				rots = 4
			}
			for rot := 0; rot < rots; rot++ {
				try(encodeSeparateAlphaMode(px, 5, rot, 0, iters))
				try(encodeSeparateAlphaMode(px, 4, rot, 0, iters))
				try(encodeSeparateAlphaMode(px, 4, rot, 1, iters))
			}
		}
	}
	packBC7(&best, out)
}

func packBC7(c *bc7Cand, out []byte) {
	m := bc7Modes[c.mode]
	var lo, hi uint64
	put128(&lo, &hi, c.mode, 1, 1)
	pos := c.mode + 1
	put128(&lo, &hi, pos, m.pb, c.part)
	pos += m.pb
	put128(&lo, &hi, pos, m.rb, c.rot)
	pos += m.rb
	put128(&lo, &hi, pos, m.isb, c.isel)
	pos += m.isb
	codes, pb, idx1, idx2 := c.codes, c.pb, c.idx1, c.idx2
	if m.ib2 == 0 {
		// each subset's anchor index must have its top bit clear
		top := 1 << uint(m.ib-1)
		for s := 0; s < m.ns; s++ {
			anchor := 0
			if s == 1 && m.ns == 2 {
				anchor = anchor2[c.part]
			} else if s == 1 {
				anchor = anchor3a[c.part]
			} else if s == 2 {
				anchor = anchor3b[c.part]
			}
			if idx1[anchor] >= top {
				codes[2*s], codes[2*s+1] = codes[2*s+1], codes[2*s]
				pb[2*s], pb[2*s+1] = pb[2*s+1], pb[2*s]
				for i := 0; i < 16; i++ {
					if subsetOf(m.ns, c.part, i) == s {
						idx1[i] = (2*top - 1) - idx1[i]
					}
				}
			}
		}
	} else {
		cib, aib := m.ib, m.ib2
		if c.isel == 1 {
			cib, aib = m.ib2, m.ib
		}
		if idx1[0] >= 1<<uint(cib-1) {
			for k := 0; k < 3; k++ {
				codes[0][k], codes[1][k] = codes[1][k], codes[0][k]
			}
			for i := range idx1 {
				idx1[i] = (1<<uint(cib) - 1) - idx1[i]
			}
		}
		if idx2[0] >= 1<<uint(aib-1) {
			codes[0][3], codes[1][3] = codes[1][3], codes[0][3]
			for i := range idx2 {
				idx2[i] = (1<<uint(aib) - 1) - idx2[i]
			}
		}
		if c.isel == 1 { // primary indices hold alpha, secondary hold color
			idx1, idx2 = idx2, idx1
		}
	}
	ne := 2 * m.ns
	for ch := 0; ch < 3; ch++ {
		for e := 0; e < ne; e++ {
			put128(&lo, &hi, pos, m.cb, codes[e][ch])
			pos += m.cb
		}
	}
	if m.ab > 0 {
		for e := 0; e < ne; e++ {
			put128(&lo, &hi, pos, m.ab, codes[e][3])
			pos += m.ab
		}
	}
	if m.epb > 0 {
		for e := 0; e < ne; e++ {
			put128(&lo, &hi, pos, 1, pb[e])
			pos++
		}
	}
	if m.spb > 0 {
		for s := 0; s < m.ns; s++ {
			put128(&lo, &hi, pos, 1, pb[2*s])
			pos++
		}
	}
	for i := 0; i < 16; i++ {
		n := m.ib
		if isAnchor(m.ns, c.part, i) {
			n--
		}
		put128(&lo, &hi, pos, n, idx1[i])
		pos += n
	}
	if m.ib2 > 0 {
		for i := 0; i < 16; i++ {
			n := m.ib2
			if i == 0 {
				n--
			}
			put128(&lo, &hi, pos, n, idx2[i])
			pos += n
		}
	}
	block128(lo, hi, out)
}

// assignLine is like assign for palettes interpolated between e0 and e1: it estimates the
// index from the projection on the line and only checks the neighbouring entries.
func assignLine(px *[16]vec4, pal []vec4, mask *[16]bool, dim int, idx *[16]int, e0, e1 vec4, wts []int) float64 {
	np := len(pal)
	if np <= 4 {
		return assign(px, pal, mask, dim, idx)
	}
	var d vec4
	dd := 0.0
	for c := 0; c < dim; c++ {
		d[c] = e1[c] - e0[c]
		dd += d[c] * d[c]
	}
	if dd < 1e-9 {
		return assign(px, pal, mask, dim, idx)
	}
	total := 0.0
	for i := 0; i < 16; i++ {
		t := 0.0
		for c := 0; c < dim; c++ {
			t += (px[i][c] - e0[c]) * d[c]
		}
		w := clamp(t/dd*64, 0, 64)
		k0 := 0
		for k0 < np-1 && float64(wts[k0+1]) <= w {
			k0++
		}
		best, bi := math.Inf(1), 0
		for k := maxi(0, k0-1); k <= mini(np-1, k0+2); k++ {
			e := 0.0
			for c := 0; c < dim; c++ {
				x := px[i][c] - pal[k][c]
				e += x * x
			}
			if e < best {
				best, bi = e, k
			}
		}
		idx[i] = bi
		if mask == nil || mask[i] {
			total += best
		}
	}
	return total
}
