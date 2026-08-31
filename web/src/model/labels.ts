/**
 * Which cities to name on a map, and where.
 *
 * A posterior surface is unreadable without anchors. The country's outline
 * alone tells a reader roughly where the coasts are, but "the bright patch is
 * around Pittsburgh" is a fact the picture cannot deliver on its own, and it
 * is the fact the reader wants.
 *
 * The places already ship in `manifest.json` for the guess list, sorted by
 * population, so naming them costs no payload. What it costs is space: at the
 * size these maps are drawn, more than about twenty labels collide into a
 * grey mat and the map is worse than it was unlabelled.
 *
 * Selection is therefore greedy over population with a rejection test, which
 * is the standard cartographic compromise: walk the places in order of
 * importance and take one only if it does not land on something already
 * taken. It is not optimal -- the optimal label set is NP-hard and nobody
 * needs it -- but it is stable, it never drops a large city for a small one,
 * and it degrades gracefully when the map shrinks, because the caller passes
 * a larger separation and the same walk simply accepts fewer.
 *
 * The rejection test is a box rather than a circle. Labels are text: roughly
 * eight times wider than they are tall, so two cities a centimetre apart
 * vertically are fine and two a centimetre apart horizontally are not.
 */

export interface Place {
  name: string;
  state: string;
  lat: number;
  lon: number;
  pop: number;
}

export interface PlacedLabel {
  name: string;
  /** Projected position, both axes in [0,1]. */
  x: number;
  y: number;
}

/** Cities whose name is ambiguous without the state, at this scale. */
const NEEDS_STATE = new Set(["Kansas City", "Portland", "Columbus", "Charleston", "Springfield"]);

/**
 * The Census names incorporated places by their legal type and by their
 * governing arrangement, so the file carries "Indianapolis city",
 * "Louisville/Jefferson County" and "Nashville-Davidson". Those are correct as
 * records and wrong on a map, where the reader wants the name they would say
 * out loud.
 *
 * The suffix strip is deliberately case-sensitive. The Census writes the legal
 * type in lower case precisely so it can be told apart from a name, and a
 * case-insensitive rule turns Oklahoma City into Oklahoma and Kansas City into
 * Kansas -- which is not a city and is a different place entirely.
 */
const RENAMED: Record<string, string> = { "Nashville-Davidson": "Nashville" };

function tidy(raw: string): string {
  const name = raw
    .split("/")[0]
    .replace(/\s(city|town|village|borough|municipality|CDP)$/, "")
    .replace(/\s+\(balance\)$/i, "")
    .trim();
  return RENAMED[name] ?? name;
}

/**
 * Pick up to `count` places to label, none within `sepX` by `sepY` of another.
 *
 * `forward` projects lat/lon to the same [0,1] square the map draws into, and
 * returns positions outside it for places off the grid, which are dropped.
 * Separations are in that square's x units, so a caller that knows the map is
 * narrow can widen them without knowing anything about the projection.
 *
 * `avoid` holds anything already on the map that a name must not land on --
 * in practice the guess marker, which is the one thing on the picture more
 * important than a label and the one most likely to be collided with, since
 * the marker lands where the probability is and the probability is usually
 * near a city.
 */
export function pickLabels(
  places: Place[],
  forward: (lat: number, lon: number) => { x: number; y: number },
  count: number,
  sepX: number,
  sepY: number,
  avoid: { x: number; y: number }[] = [],
): PlacedLabel[] {
  const out: PlacedLabel[] = [];
  const taken = avoid.slice();
  for (const p of places) {
    if (out.length >= count) break;
    const { x, y } = forward(p.lat, p.lon);
    // A margin, not the bare edge: a label at x = 0.99 is off the picture
    // even though its dot is on it.
    if (x < 0.03 || x > 0.97 || y < 0.04 || y > 0.96) continue;
    let clear = true;
    for (const q of taken) {
      if (Math.abs(q.x - x) < sepX && Math.abs(q.y - y) < sepY) {
        clear = false;
        break;
      }
    }
    if (!clear) continue;
    const name = tidy(p.name);
    out.push({ name: NEEDS_STATE.has(name) ? `${name}, ${p.state}` : name, x, y });
    taken.push({ x, y });
  }
  return out;
}
