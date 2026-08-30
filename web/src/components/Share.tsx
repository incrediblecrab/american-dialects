import { useCallback, useMemo, useRef, useState } from "react";
import type { CardSpec } from "./ShareCard";
import { renderCard, toPngBlob } from "./ShareCard";
import styles from "./Share.module.css";

interface Props {
  /** The plain-text result. Also the caption when the card is shared. */
  text: string;
  /** Built on demand: rendering the card costs a full map draw. */
  buildCard: () => CardSpec;
  fileName: string;
  children?: React.ReactNode;
}

type State = { kind: "idle" | "working" | "done" | "failed"; message: string };

const IDLE: State = { kind: "idle", message: "" };

/**
 * Somewhere for the result to go.
 *
 * GitHub Pages serves static files, so there is no endpoint to POST a game to
 * and nothing here talks to a server. Both affordances are entirely local: the
 * clipboard write happens in this tab, and the card is drawn in this tab and
 * handed to the operating system. That is a constraint of the hosting, but it
 * is also the honest arrangement for a page that tells the reader nothing is
 * recorded -- a share button that quietly logged the result would make that
 * sentence false.
 *
 * Both paths have a floor. If the clipboard is unavailable, which is the norm
 * inside embedded browsers and over plain http, the text is revealed in a
 * field the reader can select by hand. If the Web Share sheet is absent or
 * dismissed with an error, the card downloads instead.
 */
export default function Share({ text, buildCard, fileName, children }: Props) {
  const [state, setState] = useState<State>(IDLE);
  const [fallback, setFallback] = useState<string | null>(null);
  const areaRef = useRef<HTMLTextAreaElement>(null);

  const canShareFiles = useMemo(() => {
    try {
      if (!navigator.share || !navigator.canShare) return false;
      const probe = new File([new Uint8Array()], fileName, { type: "image/png" });
      return navigator.canShare({ files: [probe] });
    } catch {
      return false;
    }
  }, [fileName]);

  const copy = useCallback(async () => {
    setFallback(null);
    try {
      await navigator.clipboard.writeText(text);
      setState({ kind: "done", message: "Copied. It is plain text, so paste it anywhere." });
    } catch {
      setFallback(text);
      setState({
        kind: "failed",
        message: "This browser would not let the page write to the clipboard. The text is below.",
      });
      window.setTimeout(() => areaRef.current?.select(), 0);
    }
  }, [text]);

  const save = useCallback(async () => {
    setFallback(null);
    setState({ kind: "working", message: "Drawing the card…" });
    try {
      const blob = await toPngBlob(renderCard(buildCard()));
      const file = new File([blob], fileName, { type: "image/png" });

      if (canShareFiles) {
        try {
          await navigator.share({ files: [file], text });
          setState({ kind: "done", message: "Sent to the share sheet." });
          return;
        } catch (e) {
          // A dismissed sheet is not a failure, and must not silently trigger
          // a download the reader did not ask for.
          if (e instanceof DOMException && e.name === "AbortError") {
            setState(IDLE);
            return;
          }
        }
      }

      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = fileName;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 60_000);
      setState({ kind: "done", message: `Saved as ${fileName}.` });
    } catch (e) {
      setState({
        kind: "failed",
        message: `The card could not be drawn: ${e instanceof Error ? e.message : String(e)}`,
      });
    }
  }, [buildCard, canShareFiles, fileName, text]);

  return (
    <div className={styles.take}>
      <p className={styles.kicker}>Take it with you</p>
      <div className={styles.actions}>
        <button className={styles.action} onClick={() => void copy()}>
          Copy the result
        </button>
        <button
          className={styles.action}
          onClick={() => void save()}
          disabled={state.kind === "working"}
        >
          {canShareFiles ? "Share the map" : "Save the map"}
        </button>
      </div>
      <p
        className={state.kind === "failed" ? styles.statusWarn : styles.status}
        aria-live="polite"
      >
        {state.message}
      </p>
      {fallback ? (
        <textarea
          ref={areaRef}
          className={styles.fallback}
          readOnly
          rows={8}
          value={fallback}
        />
      ) : null}
      {children ? <div className={styles.note}>{children}</div> : null}
    </div>
  );
}
