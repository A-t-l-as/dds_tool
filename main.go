// dds_tool - DDS texture converter (DDS <-> PNG/EXR) with full metadata stored in JSON.
//
//	Export:   dds_tool export texture.dds         -> texture.png + texture.json
//	Import:   dds_tool import texture.json        -> texture.dds (same format, mipmaps regenerated)
//	Create:   dds_tool create image.png -f BC7_UNORM_SRGB
//	Info:     dds_tool info texture.dds
//
// Files can also be dragged & dropped onto the executable.
package main

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Version is overridden at build time by the release workflow (-ldflags "-X main.Version=...").
var Version = "3.0"

const usage = `dds_tool %s - DDS texture converter (DDS <-> PNG/EXR + JSON)

Usage:
  dds_tool export <file.dds|folder>... [-o OUTDIR] [-r]
      DDS -> image(s) + JSON. Only the largest mipmap is exported.
  dds_tool import <file.json|image|folder>... [-o OUT] [-r] [--mips N] [--quality Q]
      JSON (+ edited image) -> DDS in the original format, mipmaps regenerated.
  dds_tool create <image>... (-f FORMAT | --like FILE) [-o OUT] [--mips N] [--cube] [--json] [--quality Q]
      New DDS texture from any image (PNG, EXR, JPG, BMP, TGA, GIF).
  dds_tool info <file.dds>...
      Show information about DDS files.
  dds_tool formats
      List supported formats.

  You can also drag & drop .dds / .json / image files onto the program:
  .dds files are exported, .json files (or edited images) are imported.

Options:
  -o, --out, --outdir  output file or folder
  -r, --recursive      also process subfolders
  --mips N             number of mipmaps (0 = full chain, 1 = no mipmaps)
  -f, --format NAME    format for 'create', e.g. BC7_UNORM_SRGB, BC5_UNORM, DXT5
  --like FILE          'create': copy format and header from an existing .dds or .json
  --cube               'create': 6 images in the order +X -X +Y -Y +Z -Z make a cubemap
  --json               'create': also write a .json for later editing
  --quality Q          BC7 quality: fast, normal (default) or best
`

type cliArgs struct {
	cmd       string
	files     []string
	out       string
	recursive bool
	mips      *int
	format    string
	like      string
	cube      bool
	json      bool
	quality   int
}

func parseArgs(argv []string) (*cliArgs, error) {
	a := &cliArgs{quality: 1}
	commands := map[string]bool{"export": true, "import": true, "create": true, "info": true, "formats": true}
	if len(argv) > 0 && commands[argv[0]] {
		a.cmd = argv[0]
		argv = argv[1:]
	} else {
		a.cmd = "auto"
	}
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		name, val, hasVal := arg, "", false
		if strings.HasPrefix(arg, "--") {
			if k := strings.IndexByte(arg, '='); k > 0 {
				name, val, hasVal = arg[:k], arg[k+1:], true
			}
		}
		next := func() (string, error) {
			if hasVal {
				return val, nil
			}
			if i+1 >= len(argv) {
				return "", fmt.Errorf("option %s needs a value", name)
			}
			i++
			return argv[i], nil
		}
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			a.files = append(a.files, arg)
			continue
		}
		var err error
		switch name {
		case "-o", "--out", "--outdir":
			a.out, err = next()
		case "-r", "--recursive":
			a.recursive = true
		case "--mips":
			var s string
			if s, err = next(); err == nil {
				n, e := strconv.Atoi(s)
				if e != nil {
					return nil, fmt.Errorf("--mips needs a number")
				}
				a.mips = &n
			}
		case "-f", "--format":
			a.format, err = next()
		case "--like":
			a.like, err = next()
		case "--cube":
			a.cube = true
		case "--json":
			a.json = true
		case "--quality", "--bc7-level":
			var s string
			if s, err = next(); err == nil {
				switch strings.ToLower(s) {
				case "fast", "0":
					a.quality = 0
				case "normal", "1", "2":
					a.quality = 1
				case "best", "3", "4":
					a.quality = 2
				default:
					return nil, fmt.Errorf("--quality must be fast, normal or best")
				}
			}
		case "--bc7-builtin":
			// accepted for compatibility with the Python version
		case "-h", "--help":
			a.cmd = "help"
		case "-v", "--version":
			a.cmd = "version"
		default:
			return nil, fmt.Errorf("unknown option %s (see dds_tool --help)", arg)
		}
		if err != nil {
			return nil, err
		}
	}
	return a, nil
}

func hasExt(path string, exts ...string) bool {
	e := strings.ToLower(filepath.Ext(path))
	for _, x := range exts {
		if e == x {
			return true
		}
	}
	return false
}

// collect expands folders into the files with the given extensions.
func collect(paths []string, recursive bool, exts ...string) []string {
	var out []string
	for _, p := range paths {
		st, err := os.Stat(p)
		if err != nil || !st.IsDir() {
			out = append(out, p)
			continue
		}
		if recursive {
			filepath.WalkDir(p, func(fp string, d fs.DirEntry, err error) error {
				if err == nil && !d.IsDir() && hasExt(fp, exts...) {
					out = append(out, fp)
				}
				return nil
			})
		} else {
			entries, _ := os.ReadDir(p)
			var names []string
			for _, e := range entries {
				if !e.IsDir() && hasExt(e.Name(), exts...) {
					names = append(names, filepath.Join(p, e.Name()))
				}
			}
			sort.Strings(names)
			out = append(out, names...)
		}
	}
	return out
}

func findJSONForImage(path string) (string, error) {
	abs, _ := filepath.Abs(path)
	dir := filepath.Dir(abs)
	name := filepath.Base(abs)
	cand := filepath.Join(dir, strings.TrimSuffix(name, filepath.Ext(name))+".json")
	if _, err := os.Stat(cand); err == nil {
		return cand, nil
	}
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.IsDir() || !hasExt(e.Name(), ".json") {
			continue
		}
		m, err := loadMeta(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		for _, im := range m.Images {
			if filepath.Base(resolveImagePath(dir, im.File)) == name {
				return filepath.Join(dir, e.Name()), nil
			}
		}
	}
	return "", fmt.Errorf("No .json file found for %s", path)
}

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(argv []string) int {
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	log := func(format string, a ...interface{}) {
		fmt.Fprintf(out, format+"\n", a...)
		out.Flush()
	}
	pause := launchedFromExplorer() // keep the window open after drag & drop / double-click
	defer func() {
		if pause {
			out.Flush()
			fmt.Print("\nPress Enter to close...")
			bufio.NewReader(os.Stdin).ReadString('\n')
		}
	}()
	if len(argv) == 0 {
		log(usage, Version)
		return 1
	}
	a, err := parseArgs(argv)
	if err != nil {
		log("ERROR: %v", err)
		return 2
	}
	switch a.cmd {
	case "help":
		log(usage, Version)
		return 0
	case "version":
		log("dds_tool %s", Version)
		return 0
	}
	if a.cmd != "formats" && len(a.files) == 0 {
		log("ERROR: no input files (see dds_tool --help)")
		return 2
	}
	errors := 0
	do := func(fn func() error) {
		defer func() {
			if r := recover(); r != nil {
				errors++
				log("  ERROR (internal): %v", r)
			}
		}()
		if err := fn(); err != nil {
			errors++
			log("  ERROR: %v", err)
		}
	}
	opts := EncOpts{Quality: a.quality}
	importOne := func(p string, done map[string]bool) {
		do(func() error {
			j := p
			if !hasExt(p, ".json") && (isImageExt(p) || hasExt(p, ".tif", ".tiff")) {
				var err error
				if j, err = findJSONForImage(p); err != nil {
					return err
				}
			}
			if done != nil {
				key, _ := filepath.Abs(j)
				if done[key] {
					return nil
				}
				done[key] = true
			}
			_, err := importJSON(j, a.out, a.mips, opts, log)
			return err
		})
	}
	switch a.cmd {
	case "export":
		for _, p := range collect(a.files, a.recursive, ".dds") {
			do(func() error { _, err := exportDDS(p, a.out, log); return err })
		}
	case "import":
		done := map[string]bool{}
		for _, p := range collect(a.files, a.recursive, ".json") {
			importOne(p, done)
		}
	case "create":
		do(func() error {
			_, err := createDDS(a.files, a.format, a.like, a.out, a.mips, a.cube, a.json, opts, log)
			return err
		})
	case "formats":
		listFormats(log)
	case "info":
		for _, p := range collect(a.files, a.recursive, ".dds") {
			do(func() error { return infoDDS(p, log) })
		}
	case "auto":
		done := map[string]bool{}
		for _, p := range collect(a.files, a.recursive, ".dds", ".json") {
			if hasExt(p, ".dds") {
				do(func() error { _, err := exportDDS(p, a.out, log); return err })
			} else {
				importOne(p, done)
			}
		}
		if errors == 0 {
			log("Done.")
		}
	}
	if errors > 0 {
		log("\n%d error(s).", errors)
		return 1
	}
	return 0
}
