package main

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// testImage returns a smooth RGBA test image with some sharp edges.
func testImage(w, h int) *Img {
	img := NewImg(w, h, 4, false)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			fx, fy := float64(x)/float64(w), float64(y)/float64(h)
			v := [4]float64{0.5 + 0.5*math.Sin(fx*9), fy, float64((x/8+y/8)%2) * 0.8, 1}
			for c := 0; c < 4; c++ {
				img.F32[(y*w+x)*4+c] = float32(v[c])
			}
		}
	}
	return img
}

func psnr(a, b *Img, channels int) float64 {
	mse := 0.0
	n := 0
	for i := 0; i < a.W*a.H; i++ {
		for c := 0; c < channels; c++ {
			d := a.Get(i*a.C+c) - b.Get(i*b.C+c)
			mse += d * d
			n++
		}
	}
	mse /= float64(n)
	if mse == 0 {
		return 99
	}
	return 10 * math.Log10(1/mse)
}

func TestHalfFloat(t *testing.T) {
	for h := 0; h < 0x10000; h++ {
		if (h>>10)&31 == 31 && h&0x3ff != 0 {
			continue // NaN
		}
		if got := floatToHalf(halfToFloat(uint16(h))); got != uint16(h) {
			t.Fatalf("half 0x%04x -> 0x%04x", h, got)
		}
	}
}

func TestBCRoundTrip(t *testing.T) {
	src := testImage(64, 48)
	cases := []struct {
		dxgi     int
		channels int
		minPSNR  float64
	}{
		{71, 3, 30}, {74, 4, 30}, {77, 4, 30}, {80, 1, 40}, {83, 2, 40}, {98, 4, 38},
	}
	for _, tc := range cases {
		f := DXGI[tc.dxgi]
		in := NewImg(src.W, src.H, len(f.ValueChannels()), false)
		for i := 0; i < src.W*src.H; i++ {
			for c := 0; c < in.C; c++ {
				in.F32[i*in.C+c] = src.F32[i*4+c]
			}
		}
		for q := 0; q <= 2; q++ {
			out, err := bcDecode(bcEncode(in, f, EncOpts{Quality: q}), f, in.W, in.H)
			if err != nil {
				t.Fatal(err)
			}
			if p := psnr(in, out, tc.channels); p < tc.minPSNR {
				t.Errorf("%s quality %d: PSNR %.1f dB < %.1f", f.Name, q, p, tc.minPSNR)
			}
		}
	}
}

func TestRawFormatsLossless(t *testing.T) {
	for _, num := range sortedDXGI() {
		f := DXGI[num]
		if f.Kind != "raw" {
			continue
		}
		w, h := 7, 5
		buf := make([]byte, f.SurfaceSize(w, h))
		for i := range buf {
			buf[i] = byte(i*37 + num)
		}
		img, err := rawDecode(buf, f, w, h, f.Need64())
		if err != nil {
			t.Fatal(err)
		}
		// re-encode the decoded values, decode again: values must be stable
		img2, _ := rawDecode(rawEncode(img, f), f, w, h, f.Need64())
		for i := 0; i < img.Len(); i++ {
			a, b := img.Get(i), img2.Get(i)
			if math.IsNaN(a) || math.IsInf(a, 0) {
				continue // NaN/Inf are deliberately replaced on encode
			}
			if a != b {
				t.Fatalf("%s: value %d changed %v -> %v", f.Name, i, a, b)
			}
		}
	}
}

func TestExportImportCLI(t *testing.T) {
	dir := t.TempDir()
	png := filepath.Join(dir, "tex.png")
	if err := saveImage(png, testImage(40, 24), "png8"); err != nil {
		t.Fatal(err)
	}
	dds := filepath.Join(dir, "tex.dds")
	if code := run([]string{"create", png, "-f", "BC3_UNORM_SRGB", "-o", dds}); code != 0 {
		t.Fatalf("create exit code %d", code)
	}
	out := filepath.Join(dir, "out")
	if code := run([]string{"export", dds, "-o", out}); code != 0 {
		t.Fatalf("export exit code %d", code)
	}
	if code := run([]string{"import", filepath.Join(out, "tex.json")}); code != 0 {
		t.Fatalf("import exit code %d", code)
	}
	a, _ := os.ReadFile(dds)
	b, _ := os.ReadFile(filepath.Join(out, "tex.dds"))
	if len(a) != len(b) || string(a[:148]) != string(b[:148]) {
		t.Fatalf("rebuilt texture differs in size or header")
	}
	d, err := readDDS(filepath.Join(out, "tex.dds"))
	if err != nil || d.Tex.MipLevels != 6 {
		t.Fatalf("expected 6 mip levels, got %+v (%v)", d, err)
	}
}
