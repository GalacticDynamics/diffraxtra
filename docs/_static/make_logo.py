# /// script
# requires-python = ">=3.11"
# dependencies = ["resvg-py"]
# ///
"""Draw the diffraxtra logo: diffrax's, plus a plus.

diffrax's logo is an integral sign cut out of a rounded square. diffraxtra's
cuts a big + out beside it, for the extras. Both marks are vector paths, so the
logo is written as an SVG, sharp at any size; for a bitmap, name a .png and give
its size::

    uv run docs/_static/make_logo.py                     # favicon.svg
    uv run docs/_static/make_logo.py --size 2048 big.png
"""

import argparse
from pathlib import Path

# The integral sign: Material Design Icons' "math-integral", by Pictogrammers,
# under the Apache License 2.0 (https://pictogrammers.com/library/mdi/). The box
# is that set's "math-integral-box", diffrax's logo, as a rounded square.
INTEGRAL = (
    "M11.5 19.1C11.3 20.2 10.9 21 10.2 21.5C9.5 22 8.6 22.1 7.5 21.9C7.1 21.8 "
    "6.3 21.7 6 21.5L6.5 20C6.8 20.1 7.4 20.3 7.7 20.3C8.8 20.5 9.4 20 9.6 "
    "18.8L12 5.2C12.2 4 12.7 3.2 13.4 2.6C14.1 2.1 15.1 1.9 16.2 2.1C16.6 2.2 "
    "17.4 2.3 18 2.6L17.5 4C17.3 3.9 16.6 3.8 16.3 3.7C15 3.5 14.3 4.1 14 "
    "5.6L11.5 19.1Z"
)
# In the icons' 24-unit grid: the integral is scaled to fit the box beside the
# plus, which is drawn as two round-capped strokes.
INTEGRAL_AT = "translate(9.2 12) scale(0.78) translate(-12 -12)"
PLUS = (15.6, 12, 2.9, 2.0)  # centre x, centre y, arm length, stroke width

SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="512" height="512">
  <mask id="cut">
    <rect width="24" height="24" fill="#fff"/>
    <path d="{integral}" transform="{integral_at}" fill="#000"/>
    <path d="{plus}" stroke="#000" stroke-width="{width:g}" stroke-linecap="round"/>
  </mask>
  <rect x="3" y="3" width="18" height="18" rx="2" fill="#000" mask="url(#cut)"/>
</svg>
"""


def svg() -> str:
    """Return the logo as SVG text."""
    x, y, arm, width = PLUS
    plus = f"M{x - arm:g} {y:g}H{x + arm:g}M{x:g} {y - arm:g}V{y + arm:g}"
    return SVG.format(
        integral=INTEGRAL, integral_at=INTEGRAL_AT, plus=plus, width=width
    )


def main() -> None:
    """Parse the command line and save the logo."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "out",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("favicon.svg"),
        help="output file, SVG or PNG by its extension (default: favicon.svg)",
    )
    parser.add_argument(
        "--size", type=int, default=512, help="pixels per side, for a PNG"
    )
    args = parser.parse_args()

    if args.out.suffix == ".svg":
        args.out.write_text(svg())
    else:
        import resvg_py  # noqa: PLC0415  # only a PNG needs a renderer

        png = resvg_py.svg_to_bytes(svg_string=svg(), width=args.size)
        args.out.write_bytes(bytes(png))


if __name__ == "__main__":
    main()
