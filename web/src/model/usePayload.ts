import { useEffect, useState } from "react";
import type { Payload } from "./payload";
import { sharedPayload } from "./payload";

/**
 * What App used to do, once per island.
 *
 * Each interactive section is now mounted independently, so each one has to
 * ask for the payload itself. The request behind this is shared, so three
 * islands calling it still make one fetch; what is not shared is the render
 * state, which is correct, because an island that hydrates later should show
 * its own loading state rather than inherit somebody else's.
 */
export function usePayload(): { payload: Payload | null; error: string | null } {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    sharedPayload().then(
      (p) => {
        if (live) setPayload(p);
      },
      (e: unknown) => {
        if (live) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      live = false;
    };
  }, []);

  return { payload, error };
}
