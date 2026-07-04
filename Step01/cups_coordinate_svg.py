import base64
from html import escape
from pathlib import Path
import struct


IMAGE_PATH = Path(
    r"C:\Users\ADMINI~1\AppData\Local\Temp\codex-clipboard-c1c8894c-856f-4796-8bf0-1f27e26b390d.png"
)
OUT_PATH = Path(__file__).with_suffix(".svg")

# ponytail: arbitrary units; replace 100/100 with measured cm if you have them.
WORLD_W = 100.0
WORLD_H = 100.0

# ponytail: blue-pixel centers from the photo; adjust these if you need sub-cup precision.
CUPS = [
    ("origin", (61.2, 341.7), (0.0, 0.0)),
    ("x+", (758.0, 347.0), (WORLD_W, 0.0)),
    ("xy", (636.8, 296.2), (WORLD_W, WORLD_H)),
    ("y+", (189.9, 292.5), (0.0, WORLD_H)),
]

MEASURED = [
    ("middle", (404.3, 308.8)),
]


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", header[16:24])


def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]

        scale = a[col][col]
        a[col] = [v / scale for v in a[col]]
        b[col] /= scale

        for row in range(n):
            if row == col:
                continue
            factor = a[row][col]
            a[row] = [v - factor * p for v, p in zip(a[row], a[col])]
            b[row] -= factor * b[col]
    return b


def homography(src: list[tuple[float, float]], dst: list[tuple[float, float]]) -> list[float]:
    a = []
    b = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
    return solve_linear(a, b) + [1.0]


def project(h: list[float], x: float, y: float) -> tuple[float, float]:
    den = h[6] * x + h[7] * y + h[8]
    return (
        (h[0] * x + h[1] * y + h[2]) / den,
        (h[3] * x + h[4] * y + h[5]) / den,
    )


def points_attr(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def line_svg(points: list[tuple[float, float]], klass: str) -> str:
    return f'<polyline class="{klass}" points="{points_attr(points)}" />'


def world_screen(x: float, y: float, ox: float, oy: float, w: float, h: float) -> tuple[float, float]:
    return ox + x / WORLD_W * w, oy + h - y / WORLD_H * h


def transforms() -> tuple[list[float], list[float]]:
    img_pts = [p for _, p, _ in CUPS]
    world_pts = [p for _, _, p in CUPS]
    world_to_img = homography(world_pts, img_pts)
    img_to_world = homography(img_pts, world_pts)

    for _, img_pt, world_pt in CUPS:
        got = project(img_to_world, *img_pt)
        assert abs(got[0] - world_pt[0]) < 1e-6 and abs(got[1] - world_pt[1]) < 1e-6

    return world_to_img, img_to_world


def build_svg() -> str:
    img_w, img_h = png_size(IMAGE_PATH)
    gap = 28
    plot_w = 500
    plot_h = img_h
    total_w = img_w + gap + plot_w
    plot_x = img_w + gap
    pad = 56
    graph_w = plot_w - pad * 2
    graph_h = plot_h - pad * 2
    world_to_img, img_to_world = transforms()
    measured = [(name, img_pt, project(img_to_world, *img_pt)) for name, img_pt in MEASURED]

    image_uri = "data:image/png;base64," + base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{img_h}" viewBox="0 0 {total_w} {img_h}">',
        "<style>",
        "text{font-family:Arial,sans-serif;font-size:13px;fill:#111}",
        ".grid{fill:none;stroke:#00a8a8;stroke-width:1.2;stroke-opacity:.55}",
        ".axis{fill:none;stroke:#e53935;stroke-width:3;stroke-linecap:round}",
        ".frame{fill:none;stroke:#111;stroke-width:2}",
        ".cup{fill:#1e63ff;stroke:white;stroke-width:2}",
        ".measured{fill:#ffeb3b;stroke:#111;stroke-width:2}",
        ".quad{fill:none;stroke:#ff9800;stroke-width:2.5}",
        ".soft{fill:#f7f7f7;stroke:#ddd}",
        "</style>",
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#e53935"/></marker></defs>',
        f'<image href="{escape(image_uri)}" x="0" y="0" width="{img_w}" height="{img_h}" />',
    ]

    for x in range(0, 101, 20):
        svg.append(line_svg([project(world_to_img, x, y) for y in range(0, 101, 10)], "grid"))
    for y in range(0, 101, 20):
        svg.append(line_svg([project(world_to_img, x, y) for x in range(0, 101, 10)], "grid"))

    quad = [project(world_to_img, *p) for p in [(0, 0), (WORLD_W, 0), (WORLD_W, WORLD_H), (0, WORLD_H), (0, 0)]]
    svg.append(line_svg(quad, "quad"))
    svg.append(line_svg([project(world_to_img, 0, 0), project(world_to_img, WORLD_W, 0)], "axis")[:-3] + ' marker-end="url(#arrow)" />')
    svg.append(line_svg([project(world_to_img, 0, 0), project(world_to_img, 0, WORLD_H)], "axis")[:-3] + ' marker-end="url(#arrow)" />')

    for name, (x, y), world in CUPS:
        svg.append(f'<circle class="cup" cx="{x}" cy="{y}" r="7" />')
        svg.append(f'<text x="{x + 9}" y="{y - 9}">{escape(name)} {world[0]:.0f},{world[1]:.0f}</text>')
    for name, (x, y), world in measured:
        svg.append(f'<circle class="measured" cx="{x}" cy="{y}" r="8" />')
        svg.append(f'<text x="{x + 10}" y="{y - 11}">{escape(name)} ({world[0]:.1f},{world[1]:.1f})</text>')

    svg.append(f'<rect class="soft" x="{plot_x}" y="0" width="{plot_w}" height="{plot_h}" />')
    svg.append(f'<text x="{plot_x + 18}" y="27">2D coordinate view</text>')

    gx = plot_x + pad
    gy = pad
    svg.append(f'<line class="axis" x1="{gx}" y1="{gy + graph_h}" x2="{gx + graph_w}" y2="{gy + graph_h}" marker-end="url(#arrow)" />')
    svg.append(f'<line class="axis" x1="{gx}" y1="{gy + graph_h}" x2="{gx}" y2="{gy}" marker-end="url(#arrow)" />')
    svg.append(f'<text x="{gx + graph_w - 36}" y="{gy + graph_h - 10}">x +inf</text>')
    svg.append(f'<text x="{gx + 8}" y="{gy + 18}">y +inf</text>')

    for tick in range(0, 101, 20):
        sx, _ = world_screen(tick, 0, gx, gy, graph_w, graph_h)
        _, sy = world_screen(0, tick, gx, gy, graph_w, graph_h)
        svg.append(f'<line class="grid" x1="{sx}" y1="{gy}" x2="{sx}" y2="{gy + graph_h}" />')
        svg.append(f'<line class="grid" x1="{gx}" y1="{sy}" x2="{gx + graph_w}" y2="{sy}" />')
        svg.append(f'<text x="{sx - 8}" y="{gy + graph_h + 20}">{tick}</text>')
        if tick:
            svg.append(f'<text x="{gx - 32}" y="{sy + 4}">{tick}</text>')

    rect = [world_screen(x, y, gx, gy, graph_w, graph_h) for x, y in [(0, 0), (WORLD_W, 0), (WORLD_W, WORLD_H), (0, WORLD_H), (0, 0)]]
    svg.append(line_svg(rect, "frame"))
    for name, _, (x, y) in CUPS:
        sx, sy = world_screen(x, y, gx, gy, graph_w, graph_h)
        svg.append(f'<circle class="cup" cx="{sx}" cy="{sy}" r="7" />')
        svg.append(f'<text x="{sx + 9}" y="{sy - 9}">{escape(name)} ({x:.0f},{y:.0f})</text>')
    for name, _, (x, y) in measured:
        sx, sy = world_screen(x, y, gx, gy, graph_w, graph_h)
        svg.append(f'<circle class="measured" cx="{sx}" cy="{sy}" r="8" />')
        svg.append(f'<text x="{sx + 10}" y="{sy - 11}">{escape(name)} ({x:.1f},{y:.1f})</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def main() -> None:
    OUT_PATH.write_text(build_svg(), encoding="utf-8")
    _, img_to_world = transforms()
    for name, img_pt in MEASURED:
        x, y = project(img_to_world, *img_pt)
        print(f"{name}: x={x:.2f}, y={y:.2f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
