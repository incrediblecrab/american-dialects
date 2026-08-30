import { useEffect, useId, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { curveFor, int } from "../content";
import type { CurveRow } from "../model/types";
import styles from "./Curve.module.css";

export interface Series {
  model: string;
  label: string;
  tone: "ink" | "accent" | "warn";
  note?: string;
}

interface Props {
  series: Series[];
  /** Vertical rule marking the deployed question count. */
  markK?: number;
  height?: number;
}

const W = 720;
const PAD = { t: 30, r: 18, b: 48, l: 52 };
/** Rendered size of an axis number, in real pixels, at any chart width. */
const AXIS_PX = 11;

/**
 * Median error against how many questions were asked.
 *
 * Hand-rolled SVG rather than a chart library. There are three lines and one
 * marker; a plotting library would be three megabytes to draw worse, and it
 * would bring its own visual language, which is the one thing this page is
 * trying not to borrow.
 *
 * Everything inside a viewBox is scaled by whatever width the chart is given,
 * which is fine for the curves and ruinous for everything else: on a phone the
 * box is squeezed to a little under half size, so eleven-pixel axis labels came
 * out at five, the two-pixel lines at one, and a chart three hundred pixels
 * tall at a hundred and forty. So the height, the gutters, the type and the
 * strokes are all held at their designed size in real pixels and only the plot
 * width flexes. At desktop widths the chart is exactly 720 across, the scale is
 * one, and none of this changes a thing.
 */
export default function Curve({ series, markK, height = 300 }: Props) {
  const id = useId();
  const [hoverK, setHoverK] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [u, setU] = useState(1);
  const [coarse, setCoarse] = useState(false);

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const w = el.getBoundingClientRect().width;
      if (w > 0) setU(W / w);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* A finger cannot hover, so the instruction under the chart has to know
   * which kind of pointer it is talking to. */
  useEffect(() => {
    const mq = window.matchMedia("(pointer: coarse)");
    const read = () => setCoarse(mq.matches);
    read();
    mq.addEventListener("change", read);
    return () => mq.removeEventListener("change", read);
  }, []);

  const data = useMemo(
    () => series.map((s) => ({ ...s, rows: curveFor(s.model) })),
    [series],
  );

  const all = data.flatMap((d) => d.rows);
  if (all.length === 0) return null;

  const ks = all.map((r) => r.k);
  const kMin = Math.min(...ks);
  const kMax = Math.max(...ks);

  const H = height * u;
  const pad = { t: PAD.t * u, r: PAD.r * u, b: PAD.b * u, l: PAD.l * u };

  /**
   * Scales and ticks from d3 rather than by hand.
   *
   * The y domain used to be rounded up to the next hundred and cut into
   * quarters, and the x ticks were the literal list [1, 5, 10, 15, 20, 25,
   * 30] filtered to the data. Both worked only because the data happens to
   * run 1 to 30: the x list silently thins to three labels on any shorter
   * curve, and quartering an arbitrary maximum gives axis numbers like 388.
   * d3's nice() and ticks() choose round numbers that suit whatever range
   * they are given, so the axis stays legible if the curve is ever re-run
   * over a different number of questions.
   */
  const xScale = useMemo(
    () => scaleLinear().domain([kMin, kMax]).range([pad.l, W - pad.r]),
    [kMin, kMax, pad.l, pad.r],
  );
  const yScale = useMemo(
    () =>
      scaleLinear()
        .domain([0, Math.max(...all.map((r) => r.medianKm))])
        .nice()
        .range([H - pad.b, pad.t]),
    [all, H, pad.b, pad.t],
  );

  const x = (k: number) => xScale(k);
  const y = (v: number) => yScale(v);

  const ticksY = yScale.ticks(5);
  const ticksX = xScale.ticks(7).filter((k) => Number.isInteger(k));

  const path = useMemo(
    () =>
      line<CurveRow>()
        .x((r) => xScale(r.k))
        .y((r) => yScale(r.medianKm)),
    [xScale, yScale],
  );

  const at = hoverK ?? markK ?? null;

  const track = (clientX: number, target: SVGSVGElement) => {
    const box = target.getBoundingClientRect();
    const px = ((clientX - box.left) / box.width) * W;
    const k = Math.round(xScale.invert(px));
    setHoverK(Math.min(kMax, Math.max(kMin, k)));
  };

  return (
    <figure className={styles.figure}>
      <svg
        ref={svgRef}
        className={styles.svg}
        viewBox={`0 0 ${W} ${H}`}
        style={{ fontSize: `${AXIS_PX * u}px` }}
        role="img"
        aria-labelledby={`${id}-title`}
        onPointerLeave={(e) => {
          if (e.pointerType === "mouse") setHoverK(null);
        }}
        onPointerDown={(e) => track(e.clientX, e.currentTarget)}
        onPointerMove={(e) => track(e.clientX, e.currentTarget)}
      >
        <title id={`${id}-title`}>
          Median error in kilometres against number of questions asked
        </title>

        {ticksY.map((v) => (
          <g key={v}>
            <line
              className={styles.grid}
              x1={pad.l}
              x2={W - pad.r}
              y1={y(v)}
              y2={y(v)}
            />
            <text
              className={styles.axis}
              x={pad.l - 8 * u}
              y={y(v) + 4 * u}
              textAnchor="end"
            >
              {int.format(v)}
            </text>
          </g>
        ))}

        {ticksX.map((k) => (
          <text
            key={k}
            className={styles.axis}
            x={x(k)}
            y={H - pad.b + 18 * u}
            textAnchor="middle"
          >
            {k}
          </text>
        ))}

        {at !== null ? (
          <line
            className={styles.marker}
            x1={x(at)}
            x2={x(at)}
            y1={pad.t}
            y2={H - pad.b}
          />
        ) : null}

        {data.map((d) => (
          <path
            key={d.model}
            className={`${styles.line} ${styles[d.tone]}`}
            d={path(d.rows) ?? undefined}
          />
        ))}

        {at !== null
          ? data.map((d) => {
              const r = d.rows.find((row) => row.k === at);
              if (!r) return null;
              return (
                <circle
                  key={d.model}
                  className={`${styles.dot} ${styles[d.tone]}`}
                  cx={x(r.k)}
                  cy={y(r.medianKm)}
                  r={4 * u}
                />
              );
            })
          : null}

        <text
          className={styles.unit}
          x={pad.l}
          y={pad.t - 12 * u}
          textAnchor="start"
        >
          median error, km
        </text>
        <text
          className={styles.unit}
          x={W - pad.r}
          y={H - 6 * u}
          textAnchor="end"
        >
          questions asked
        </text>
      </svg>

      <div className={styles.legend}>
        {data.map((d) => {
          const r = at === null ? null : d.rows.find((row) => row.k === at);
          return (
            <div key={d.model} className={styles.key}>
              <span className={`${styles.swatch} ${styles[d.tone]}`} />
              <span className={styles.name}>{d.label}</span>
              <span className={styles.value}>
                {r ? `${int.format(Math.round(r.medianKm))} km` : "—"}
              </span>
            </div>
          );
        })}
        <p className={styles.hint}>
          {hoverK === null && markK
            ? `Showing ${markK} questions, the number the quiz asks. ${
                coarse ? "Drag across to move." : "Hover to move."
              }`
            : `At ${at} question${at === 1 ? "" : "s"}.`}
        </p>
      </div>
    </figure>
  );
}
