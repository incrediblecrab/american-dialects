"""Download the Pop vs. Soda county data (Alan McConchie, popvssoda.com).

This is the one classic dialect question with real county-level counts:
~401,000 self-reported responses for pop / soda / coke / other.

Outputs:
  data/popvssoda/counties.csv - 3,141 counties with counts and shares
  data/popvssoda/states.csv   - per state/province totals
"""

import csv
import html
import io
import re

from common import fetch, out_dir

SITE = "https://www.popvssoda.com/"
COUNTY_TSV = SITE + "d3/pvscounty_fips.tsv"
STATE_PAGE = SITE + "statistics/ALL.html"

KEEP = ["State", "County_Name", "FIPS_State", "FIPS_County", "FIPS_combo",
        "SUMCOUNT", "SUMPOP", "SUMSODA", "SUMCOKE", "SUMOTHER",
        "PCTPOP", "PCTSODA", "PCTCOKE", "PCTOTHER"]


def main():
    d = out_dir("popvssoda")

    # the site rejects requests without a same-origin referer
    tsv = fetch(COUNTY_TSV, "popvssoda", "pvscounty_fips.tsv", referer=SITE)
    rows = list(csv.DictReader(io.StringIO(tsv), delimiter="\t"))
    with open(d / "counties.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, KEEP, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    total = sum(int(r["SUMCOUNT"]) for r in rows if r["SUMCOUNT"].isdigit())
    print(f"counties={len(rows)} responses={total}")

    page = fetch(STATE_PAGE, "popvssoda", "ALL.html", referer=SITE)
    body = re.sub(r"(?s)<script.*?</script>", "", page)
    states = []
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", body):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", tr)]
        cells = [c for c in cells if c != ""]
        if len(cells) >= 6 and all(c.replace(",", "").isdigit() for c in cells[1:5]):
            if cells[0].strip().lower() == "total":
                continue
            states.append({
                "state": cells[0],
                "pop": int(cells[1].replace(",", "")),
                "soda": int(cells[2].replace(",", "")),
                "coke": int(cells[3].replace(",", "")),
                "other": int(cells[4].replace(",", "")),
                "total": int(cells[5].replace(",", "")),
            })

    with open(d / "states.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["state", "pop", "soda", "coke", "other", "total"])
        w.writeheader()
        w.writerows(states)
    print(f"states/provinces={len(states)} responses={sum(s['total'] for s in states)}")


if __name__ == "__main__":
    main()
