import { content, int } from "../content";
import styles from "./Colophon.module.css";

const REPO = "https://github.com/incrediblecrab/american-dialects";

export default function Colophon() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div>
          <p className={styles.h}>Sources</p>
          <ul className={styles.list}>
            <li>
              Harvard Dialect Survey (2003), Bert Vaux and Scott Golder — the
              published answer maps and state percentages.
            </li>
            <li>
              Cambridge Survey of World Englishes — the same question set, a
              later and differently recruited population.
            </li>
            <li>
              <em>Pop vs Soda</em>, Alan McConchie — county-level returns used
              as an independent tuning target.
            </li>
            <li>
              Yale Grammatical Diversity Project — the only public source with
              real people and known raised locations.
            </li>
            <li>US Census — population, place names and the geographic prior.</li>
          </ul>
        </div>

        <div>
          <p className={styles.h}>Method</p>
          <p className={styles.body}>
            Every number on this page is generated from{" "}
            <span className={styles.stat}>
              {int.format(content.inventory.length)}
            </span>{" "}
            data files by a single export step, so nothing here is typed by
            hand and nothing can drift from the analysis behind it.
          </p>
          <p className={styles.body}>
            The model runs in your browser; no answer is sent anywhere, there
            is no analytics on this page, and nothing you copy or save is
            recorded. That is also why the accuracy figures here are simulated
            rather than measured on the people who play the quiz —{" "}
            <a href="#limits">the reason it was left that way</a>.
          </p>
          <p className={styles.body}>
            <a href={REPO}>Code, data and the full write-up</a> — including the
            findings this page summarises and the ones it does not.
          </p>
        </div>
      </div>
    </footer>
  );
}
