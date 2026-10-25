package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const ToolVersion = 3

type ImageEntry struct {
	File       string  `json:"file"`
	ArrayIndex int     `json:"array_index"`
	Face       *string `json:"face"`
	Slice      *int    `json:"slice"`
}

// Meta is the flat metadata used internally (and the JSON layout of tool_version 1).
type Meta struct {
	Tool         string       `json:"tool"`
	ToolVersion  int          `json:"tool_version"`
	Source       string       `json:"source"`
	Format       string       `json:"format"`
	Container    string       `json:"container"`
	Width        int          `json:"width"`
	Height       int          `json:"height"`
	Depth        int          `json:"depth"`
	MipLevels    int          `json:"mip_levels"`
	FullMipChain bool         `json:"full_mip_chain"`
	Dimension    string       `json:"dimension"`
	ArraySize    int          `json:"array_size"`
	Cubemap      bool         `json:"cubemap"`
	CubeFaces    []string     `json:"cube_faces"`
	Volume       bool         `json:"volume"`
	SRGB         bool         `json:"srgb"`
	BC1Alpha     *bool        `json:"bc1_alpha"`
	ImageKind    string       `json:"image_kind"`
	ImageLayout  []*string    `json:"image_layout"`
	Images       []ImageEntry `json:"images"`
	Header       Header       `json:"header"`
	DX10         *DX10        `json:"dx10"`
}

const (
	secEdit = "editable"
	secInfo = "info"
	secTech = "do_not_edit"
)

// section/key names used by tool_version 2 (still accepted on import)
var v2Sections = map[string]string{"edytowalne": secEdit, "informacje": secInfo, "nie_zmieniaj": secTech}
var v2Keys = map[string]string{"mipmapy": "mipmaps", "_opis": "_help"}
var fullWords = []string{"full", "max", "", "pelny", "pełny"}

var readme = []string{
	"Section 'editable' - safe to change.",
	"Section 'info' - for reference only, ignored on import (dimensions are taken from the images).",
	"Section 'do_not_edit' - technical data needed to reproduce the exact format. " +
		"To change the format use: dds_tool create image.png -f FORMAT",
}

type outHelp struct {
	Mipmaps  string `json:"mipmaps"`
	BC1Alpha string `json:"bc1_alpha,omitempty"`
	Images   string `json:"images"`
}

type outEdit struct {
	Mipmaps  interface{}  `json:"mipmaps"`
	BC1Alpha *bool        `json:"bc1_alpha,omitempty"`
	Images   []ImageEntry `json:"images"`
	Help     outHelp      `json:"_help"`
}

type outInfo struct {
	Source            string `json:"source"`
	Format            string `json:"format"`
	Container         string `json:"container"`
	Width             int    `json:"width"`
	Height            int    `json:"height"`
	Depth             int    `json:"depth"`
	ArraySize         int    `json:"array_size"`
	SRGB              bool   `json:"srgb"`
	OriginalMipLevels int    `json:"original_mip_levels"`
}

type outTech struct {
	Dimension   string    `json:"dimension"`
	Volume      bool      `json:"volume"`
	Cubemap     bool      `json:"cubemap"`
	CubeFaces   []string  `json:"cube_faces"`
	ImageKind   string    `json:"image_kind"`
	ImageLayout []*string `json:"image_layout"`
	Header      Header    `json:"header"`
	DX10        *DX10     `json:"dx10"`
}

type outJSON struct {
	Readme      []string `json:"_readme"`
	Tool        string   `json:"tool"`
	ToolVersion int      `json:"tool_version"`
	Editable    outEdit  `json:"editable"`
	Info        outInfo  `json:"info"`
	Tech        outTech  `json:"do_not_edit"`
}

func structureMeta(m *Meta) *outJSON {
	o := &outJSON{Readme: readme, Tool: "dds_tool", ToolVersion: ToolVersion}
	if m.FullMipChain {
		o.Editable.Mipmaps = "full"
	} else {
		o.Editable.Mipmaps = maxi(1, m.MipLevels)
	}
	o.Editable.Help.Mipmaps = `"full" = full chain computed from the image size, ` +
		`a number = exactly that many levels (1 = no mipmaps)`
	if m.BC1Alpha != nil {
		v := *m.BC1Alpha
		o.Editable.BC1Alpha = &v
		o.Editable.Help.BC1Alpha = "BC1/DXT1 only: true = 1-bit transparency (alpha < 50% -> transparent)"
	}
	o.Editable.Images = m.Images
	if o.Editable.Images == nil {
		o.Editable.Images = []ImageEntry{}
	}
	o.Editable.Help.Images = `image files; "file" may be renamed. The order of entries matters ` +
		`(cubemap: +X -X +Y -Y +Z -Z), for 3D textures the "slice" field`
	o.Info = outInfo{m.Source, m.Format, m.Container, m.Width, m.Height, m.Depth, m.ArraySize, m.SRGB, m.MipLevels}
	o.Tech = outTech{m.Dimension, m.Volume, m.Cubemap, m.CubeFaces, m.ImageKind, m.ImageLayout, m.Header, m.DX10}
	return o
}

func saveMeta(path string, m *Meta) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(structureMeta(m)); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0644)
}

func renameKeys(m map[string]json.RawMessage, names map[string]string) map[string]json.RawMessage {
	out := map[string]json.RawMessage{}
	for k, v := range m {
		if n, ok := names[k]; ok {
			k = n
		}
		out[k] = v
	}
	return out
}

func parseMipmaps(raw json.RawMessage) (full bool, n int, err error) {
	bad := func() (bool, int, error) {
		return false, 0, fmt.Errorf(`Invalid "mipmaps" value: %s (use "full" or a number)`,
			strings.ReplaceAll(string(raw), `"`, "'"))
	}
	if raw == nil {
		return true, 1, nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		t := strings.ToLower(strings.TrimSpace(s))
		for _, w := range fullWords {
			if t == w {
				return true, 1, nil
			}
		}
		v, e := strconv.Atoi(t)
		if e != nil {
			return bad()
		}
		n = v
	} else {
		var f float64
		if json.Unmarshal(raw, &f) != nil {
			return bad()
		}
		n = int(f)
	}
	return n <= 0, maxi(1, n), nil
}

// flattenMeta converts the sectioned JSON (or an old flat one) into Meta.
func flattenMeta(data []byte) (*Meta, error) {
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return nil, err
	}
	top = renameKeys(top, v2Sections)
	m := &Meta{Header: Header{MipMapCount: 1}}
	if _, ok := top[secTech]; !ok { // old flat file format (tool_version 1)
		if err := json.Unmarshal(data, m); err != nil {
			return nil, err
		}
		return m, nil
	}
	for _, sec := range []string{secInfo, secTech} {
		if raw, ok := top[sec]; ok && string(raw) != "null" {
			if err := json.Unmarshal(raw, m); err != nil {
				return nil, fmt.Errorf("section '%s': %v", sec, err)
			}
		}
	}
	edit := map[string]json.RawMessage{}
	if raw, ok := top[secEdit]; ok {
		if err := json.Unmarshal(raw, &edit); err != nil {
			return nil, fmt.Errorf("section '%s': %v", secEdit, err)
		}
	}
	edit = renameKeys(edit, v2Keys)
	full, n, err := parseMipmaps(edit["mipmaps"])
	if err != nil {
		return nil, err
	}
	m.FullMipChain, m.MipLevels = full, n
	m.BC1Alpha = nil
	if raw, ok := edit["bc1_alpha"]; ok {
		var b *bool
		if err := json.Unmarshal(raw, &b); err != nil {
			return nil, fmt.Errorf(`Invalid "bc1_alpha" value (use true or false)`)
		}
		m.BC1Alpha = b
	}
	m.Images = nil
	if raw, ok := edit["images"]; ok {
		if err := json.Unmarshal(raw, &m.Images); err != nil {
			return nil, fmt.Errorf(`Invalid "images" list: %v`, err)
		}
	}
	json.Unmarshal(top["tool"], &m.Tool)
	json.Unmarshal(top["tool_version"], &m.ToolVersion)
	return m, nil
}

func loadMeta(path string) (*Meta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := flattenMeta(data)
	if err != nil {
		if se, ok := err.(*json.SyntaxError); ok {
			line := 1 + bytes.Count(data[:minInt64(se.Offset, int64(len(data)))], []byte("\n"))
			return nil, fmt.Errorf("Syntax error in %s (line %d): %v", filepath.Base(path), line, se)
		}
		if _, ok := err.(*json.UnmarshalTypeError); ok {
			return nil, fmt.Errorf("%s is not a dds_tool metadata file", path)
		}
		return nil, fmt.Errorf("%s: %v", filepath.Base(path), err)
	}
	if m.Tool != "dds_tool" {
		return nil, fmt.Errorf("%s is not a dds_tool metadata file", path)
	}
	return m, nil
}

func minInt64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func strp(s string) *string { return &s }
func intp(i int) *int       { return &i }
