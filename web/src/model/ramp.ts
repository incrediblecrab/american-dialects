/**
 * Map colour, read from the stylesheet rather than duplicated here.
 *
 * Canvas cannot read a CSS gradient, so the ramp has to exist as numbers
 * somewhere. Taking those numbers from the custom properties in tokens.css
 * means there is still only one place the palette is defined, and it means the
 * map follows the light and dark themes without a second set of stops.
 */

const STOPS = ["--map-0", "--map-1", "--map-2", "--map-3", "--map-4"] as const;

export type Ramp = Uint8ClampedArray;

function parse(css: string): [number, number, number] {
  const s = css.trim();
  if (s.startsWith("#")) {
    const h =
      s.length === 4
        ? s
            .slice(1)
            .split("")
            .map((c) => c + c)
            .join("")
        : s.slice(1);
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ];
  }
  const m = s.match(/-?\d+(\.\d+)?/g);
  if (!m || m.length < 3) return [0, 0, 0];
  return [Number(m[0]), Number(m[1]), Number(m[2])];
}

/** A computed colour, or null when the browser gave nothing usable — an
 * unresolved `color-mix` comes back fully transparent rather than as an
 * error, and silently painting a map black is worse than falling back. */
export function parseRGB(css: string): RGB | null {
  const s = css.trim();
  if (!s) return null;
  if (s.startsWith("#")) return parse(s);
  const m = s.match(/-?\d*\.\d+|-?\d+/g);
  if (!m || m.length < 3) return null;
  if (m.length > 3 && Number(m[3]) === 0) return null;
  // A resolved color-mix() serialises as color(srgb r g b) on a 0..1 scale.
  const k = s.startsWith("color(") ? 255 : 1;
  return [Number(m[0]) * k, Number(m[1]) * k, Number(m[2]) * k];
}

/** 256 RGB triples interpolated through the theme's map stops. */
export function buildRamp(el: Element = document.documentElement): Ramp {
  const style = getComputedStyle(el);
  const anchors = STOPS.map((v) => parse(style.getPropertyValue(v)));
  const out = new Uint8ClampedArray(256 * 3);
  const span = anchors.length - 1;
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * span;
    const lo = Math.min(Math.floor(x), span - 1);
    const f = x - lo;
    const a = anchors[lo];
    const b = anchors[lo + 1];
    out[i * 3] = a[0] + f * (b[0] - a[0]);
    out[i * 3 + 1] = a[1] + f * (b[1] - a[1]);
    out[i * 3 + 2] = a[2] + f * (b[2] - a[2]);
  }
  return out;
}

/** Ink colour for borders and rules, so canvas matches the surrounding text. */
export function inkColour(alpha: number, el: Element = document.documentElement): string {
  const [r, g, b] = parse(getComputedStyle(el).getPropertyValue("--ink-3"));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export type RGB = [number, number, number];

/**
 * Any token, as canvas numbers.
 *
 * Same reason as the ramp: a renderer that cannot read CSS still has to obey
 * the palette, and the alternative is a second set of hex codes that drifts
 * from the first the next time the theme is touched.
 */
export function cssColour(
  name: string,
  el: Element = document.documentElement,
): RGB {
  return parse(getComputedStyle(el).getPropertyValue(name));
}

/** As a CSS colour string with an alpha, for strokes. */
export function rgba([r, g, b]: RGB, alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
