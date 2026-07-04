import base64
from collections import deque
from dataclasses import dataclass
from html import escape
from math import sqrt
from pathlib import Path
import struct
import zlib


IMAGE_PATH = Path(
    r"C:\Users\ADMINI~1\AppData\Local\Temp\codex-clipboard-c1c8894c-856f-4796-8bf0-1f27e26b390d.png"
)
OUT_PATH = Path(__file__).with_suffix(".svg")


@dataclass
class Blob:
    n: int
    sx: float
    sy: float
    minx: int
    miny: int
    maxx: int
    maxy: int
    name: str = ""
    rel_distance: float = 0.0

    @property
    def cx(self) -> float:
        return self.sx / self.n

    @property
    def cy(self) -> float:
        return self.sy / self.n

    @property
    def width(self) -> int:
        return self.maxx - self.minx + 1

    @property
    def height(self) -> int:
        return self.maxy - self.miny + 1

    @property
    def area_size(self) -> float:
        return sqrt(self.n)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png_rgb(path: Path) -> tuple[int, int, list[bytearray]]:
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")
        width = height = channels = 0
        idat = bytearray()
        while True:
            size_data = f.read(4)
            if not size_data:
                break
            size = struct.unpack(">I", size_data)[0]
            kind = f.read(4)
            data = f.read(size)
            f.read(4)
            if kind == b"IHDR":
                width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", data)
                if bit_depth != 8 or compression or filter_method or interlace:
                    raise ValueError("only 8-bit non-interlaced PNG is supported")
                if color_type not in (2, 6):
                    raise ValueError("only RGB/RGBA PNG is supported")
                channels = {2: 3, 6: 4}[color_type]
            elif kind == b"IDAT":
                idat.extend(data)
            elif kind == b"IEND":
                break

    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    bpp = channels
    rows: list[bytearray] = []
    prev = bytearray(stride)
    pos = 0
    for _ in range(height):
        filt = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += 1 + stride
        for i, value in enumerate(row):
            left = row[i - bpp] if i >= bpp else 0
            up = prev[i]
            up_left = prev[i - bpp] if i >= bpp else 0
            if filt == 1:
                row[i] = (value + left) & 255
            elif filt == 2:
                row[i] = (value + up) & 255
            elif filt == 3:
                row[i] = (value + ((left + up) // 2)) & 255
            elif filt == 4:
                row[i] = (value + paeth(left, up, up_left)) & 255
            elif filt != 0:
                raise ValueError(f"unknown PNG filter: {filt}")
        rows.append(row if channels == 3 else bytearray(v for i, v in enumerate(row) if i % 4 != 3))
        prev = row
    return width, height, rows


def blue_mask(width: int, height: int, rows: list[bytearray]) -> bytearray:
    mask = bytearray(width * height)
    floor_y = int(height * 0.55)
    for y, row in enumerate(rows):
        if y < floor_y:
            continue
        for x in range(width):
            r, g, b = row[x * 3 : x * 3 + 3]
            if b > 75 and b > r + 20 and b > g + 5 and r < 150 and g < 150:
                mask[y * width + x] = 1
    return mask


def components(width: int, height: int, mask: bytearray) -> list[Blob]:
    blobs: list[Blob] = []
    for start, on in enumerate(mask):
        if not on:
            continue
        q = deque([start])
        mask[start] = 0
        n = sx = sy = 0
        minx = miny = 10**9
        maxx = maxy = -1
        while q:
            idx = q.popleft()
            x = idx % width
            y = idx // width
            n += 1
            sx += x
            sy += y
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height:
                    ni = ny * width + nx
                    if mask[ni]:
                        mask[ni] = 0
                        q.append(ni)
        if n >= 10:
            blobs.append(Blob(n, sx, sy, minx, miny, maxx, maxy))
    return blobs


def close(a: Blob, b: Blob, pad: int = 18) -> bool:
    return (
        a.minx <= b.maxx + pad
        and b.minx <= a.maxx + pad
        and a.miny <= b.maxy + pad
        and b.miny <= a.maxy + pad
    )


def merge_blobs(blobs: list[Blob]) -> list[Blob]:
    groups: list[list[Blob]] = []
    for blob in blobs:
        hits = [g for g in groups if any(close(blob, other) for other in g)]
        if not hits:
            groups.append([blob])
            continue
        hits[0].append(blob)
        for extra in hits[1:]:
            hits[0].extend(extra)
            groups.remove(extra)

    merged = []
    for group in groups:
        merged.append(
            Blob(
                n=sum(b.n for b in group),
                sx=sum(b.sx for b in group),
                sy=sum(b.sy for b in group),
                minx=min(b.minx for b in group),
                miny=min(b.miny for b in group),
                maxx=max(b.maxx for b in group),
                maxy=max(b.maxy for b in group),
            )
        )
    return [b for b in merged if b.n >= 20]


def detect_cups() -> tuple[int, int, list[Blob]]:
    width, height, rows = read_png_rgb(IMAGE_PATH)
    cups = merge_blobs(components(width, height, blue_mask(width, height, rows)))
    cups.sort(key=lambda b: b.cx)
    names = ["front_left", "back_left", "middle", "back_right", "front_right"]
    for cup, name in zip(cups, names):
        cup.name = name

    ref_height = max(c.height for c in cups)
    for cup in cups:
        cup.rel_distance = ref_height / cup.height

    assert len(cups) == 5, f"expected 5 cups, found {len(cups)}"
    return width, height, cups


def svg(width: int, height: int, cups: list[Blob]) -> str:
    panel_w = 360
    total_w = width + panel_w
    image_uri = "data:image/png;base64," + base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{height}" viewBox="0 0 {total_w} {height}">',
        "<style>",
        "text{font-family:Arial,sans-serif;font-size:13px;fill:#111}",
        ".box{fill:none;stroke:#ffeb3b;stroke-width:2}",
        ".dot{fill:#ffeb3b;stroke:#111;stroke-width:2}",
        ".panel{fill:#f7f7f7;stroke:#ddd}",
        ".bar{fill:#1e63ff}",
        "</style>",
        f'<image href="{escape(image_uri)}" x="0" y="0" width="{width}" height="{height}" />',
        f'<rect class="panel" x="{width}" y="0" width="{panel_w}" height="{height}" />',
        f'<text x="{width + 18}" y="28">Pixel-size relative distance</text>',
        f'<text x="{width + 18}" y="50">nearest = 1.00x, larger = farther</text>',
    ]

    for cup in cups:
        out.append(f'<rect class="box" x="{cup.minx}" y="{cup.miny}" width="{cup.width}" height="{cup.height}" />')
        out.append(f'<circle class="dot" cx="{cup.cx:.1f}" cy="{cup.cy:.1f}" r="6" />')
        out.append(
            f'<text x="{cup.minx}" y="{cup.miny - 7}">{escape(cup.name)} h={cup.height}px d={cup.rel_distance:.2f}x</text>'
        )

    ordered = sorted(cups, key=lambda c: c.rel_distance)
    max_dist = max(c.rel_distance for c in cups)
    for i, cup in enumerate(ordered):
        y = 88 + i * 58
        bar_w = 210 * cup.rel_distance / max_dist
        out.append(f'<text x="{width + 18}" y="{y}">{escape(cup.name)}</text>')
        out.append(f'<rect class="bar" x="{width + 18}" y="{y + 10}" width="{bar_w:.1f}" height="16" />')
        out.append(
            f'<text x="{width + 238}" y="{y + 23}">h={cup.height}px d={cup.rel_distance:.2f}x</text>'
        )

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    width, height, cups = detect_cups()
    OUT_PATH.write_text(svg(width, height, cups), encoding="utf-8")
    for cup in sorted(cups, key=lambda c: c.rel_distance):
        print(
            f"{cup.name:11s} center=({cup.cx:5.1f},{cup.cy:5.1f}) "
            f"size={cup.width:2d}x{cup.height:2d}px rel_distance={cup.rel_distance:.2f}x"
        )
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
