# dds_tool

[![CI](https://github.com/A-t-l-as/dds_tool/actions/workflows/ci.yml/badge.svg)](https://github.com/A-t-l-as/dds_tool/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/A-t-l-as/dds_tool)](https://github.com/A-t-l-as/dds_tool/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A DDS texture converter: export DDS textures to ordinary images (PNG / EXR), edit them in any
image editor, and compile them back to DDS in **exactly the same format** as the original.

- Exports only the largest mipmap; mipmaps are regenerated on import.
- Every detail of the original file (format, header flags, cubemap/array layout, ...) is stored
  in a `.json` next to the image, so the rebuilt texture matches the original.
- A single self-contained program for Windows, macOS and Linux — nothing to install.

## Download

Get the latest version from the [Releases](https://github.com/A-t-l-as/dds_tool/releases/latest) page:

| System | File |
|---|---|
| Windows (most PCs) | `dds_tool-windows-x64.exe` |
| Windows on ARM | `dds_tool-windows-arm64.exe` |
| macOS, Apple Silicon (M1 and newer) | `dds_tool-macos-arm64` |
| macOS, Intel | `dds_tool-macos-x64` |
| Linux, x86-64 | `dds_tool-linux-x64` |
| Linux, ARM64 | `dds_tool-linux-arm64` |

You can rename the file to just `dds_tool` (or `dds_tool.exe`).

**macOS / Linux:** make the file executable first:

```sh
chmod +x dds_tool-macos-arm64
```

macOS blocks unsigned programs the first time they run. Right-click the file in Finder and choose
**Open**, or run:

```sh
xattr -d com.apple.quarantine dds_tool-macos-arm64
```

## Quick start

**Drag & drop** (Windows): drop files onto the program.

- `.dds` files are exported to `.png` / `.exr` + `.json`
- `.json` files (or the edited `.png` / `.exr`) are compiled back to `.dds`

**From a terminal:**

```sh
dds_tool export texture.dds                # -> texture.png + texture.json
dds_tool import texture.json               # -> texture.dds (original format, new mipmaps)

dds_tool export "C:\Game\textures" -r      # every .dds in the folder and all subfolders
dds_tool import "C:\Game\textures" -r      # rebuild all of them

dds_tool create image.png -f BC7_UNORM_SRGB     # new texture from any image
dds_tool create new.png --like original.dds     # same format as an existing texture

dds_tool info texture.dds
dds_tool formats                           # list of supported formats
dds_tool --help
```

> [!WARNING]
> `import` writes the `.dds` next to the `.json`, overwriting an existing file with the same name.
> Keep a backup of the original textures, or export to another folder with `-o`.

## Options

| Option | Description |
|---|---|
| `-o PATH` | output file or folder |
| `-r` | also process subfolders |
| `--mips N` | number of mipmaps (`0` = full chain, `1` = no mipmaps) |
| `--quality Q` | BC7 compression quality: `fast`, `normal` (default) or `best` |
| `-f FORMAT` | `create`: format, e.g. `BC7_UNORM_SRGB`, `BC5_UNORM`, `BC4_UNORM`, `DXT1`, `DXT5` |
| `--like FILE` | `create`: copy format and header from an existing `.dds` or `.json` |
| `--cube` | `create`: 6 images in the order +X −X +Y −Y +Z −Z make a cubemap |
| `--json` | `create`: also write a `.json` for later editing |

### Choosing a format for `create`

| Content | Format |
|---|---|
| Color (albedo / diffuse) | `BC7_UNORM_SRGB`; older games: `DXT1` (no alpha) or `DXT5` (alpha) |
| Normal map | `BC5_UNORM` (older games: `ATI2` or `DXT5`) |
| Single-channel mask (roughness, AO, ...) | `BC4_UNORM` |
| HDR (sky, lighting) | `BC6H_UF16`, ideally from an EXR image |
| Uncompressed | `R8G8B8A8_UNORM_SRGB` or `B8G8R8A8_UNORM` |

Use `_SRGB` formats only for color — normal maps and masks must not be sRGB, otherwise their
mipmaps are computed incorrectly.

## Supported textures

- **Block compressed:** BC1–BC7 in all variants (DXT1–5, ATI1/ATI2, BC4/BC5 SNORM,
  BC6H UF16/SF16, BC7 sRGB)
- **Uncompressed DXGI formats:** practically all — UNORM, SNORM, UINT, SINT, FLOAT, packed
  formats, depth/stencil, R9G9B9E5, R11G11B10, B4G4R4A4, ...
- **Legacy bit-mask formats:** R5G6B5, A1R5G5B5, X8R8G8B8, L8, A8L8, A8, V8U8, ...
- **Layouts:** cubemaps (also partial), texture arrays, 3D (volume) and 1D textures

Not supported: video (YUV / NV12) and palettized formats.

### Image files

| Texture format | Exported as |
|---|---|
| up to 8 bits per channel | 8-bit PNG |
| 9–16 bits per channel | 16-bit PNG |
| floating point / HDR | EXR (32-bit float) |

Edited images may be saved with a different bit depth, without alpha, or at a different size —
the program adapts. Input images for `create`: PNG, EXR, JPG, BMP, TGA, GIF.
EXR files may use no compression, RLE, ZIP, ZIPS, PIZ or PXR24 (not B44 / DWA).

## The `.json` file

| Section | Meaning |
|---|---|
| `editable` | safe to change: `mipmaps` (`"full"` or a number), `bc1_alpha`, image file names |
| `info` | for reference only (original size etc.), ignored on import |
| `do_not_edit` | technical data needed to reproduce the exact format |

The size of the texture is taken from the image, so to change the resolution just resize the
image. To change the format, use `dds_tool create image.png -f FORMAT`.

## Building from source

Requires [Go](https://go.dev) 1.24 or newer. The program uses only the Go standard library.

Install directly with Go:

```sh
go install github.com/A-t-l-as/dds_tool@latest
```

Or clone the repository and build it:

```sh
git clone https://github.com/A-t-l-as/dds_tool.git
cd dds_tool
go build -o dds_tool .       # build for this computer
go test ./...                # run the tests

# cross-compile, e.g. for Windows:
GOOS=windows GOARCH=amd64 go build -o dds_tool.exe .
```

## Releases

Pushing a version tag builds the programs for all systems and publishes them on the
Releases page (see `.github/workflows/release.yml`):

```sh
git tag v3.0.0
git push origin v3.0.0
```

## License

[MIT](LICENSE) © 2026 A-t-l-as
