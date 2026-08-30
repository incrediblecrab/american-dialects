"""Scrape the UWM Dialect Survey (Vaux & Samuels, 2004-2006, published 2018).

The survey ran online 2004-2006 with 3,000+ respondents and was cleaned and
republished by Jøhndal in November 2018 (with further updates through 2020) at
    https://dialectsurvey.wordpress.com

The sitemap lists 158 URLs; 148 have q-numbers in the slug (one of which covers
q94-97), and 5 others contain question maps reachable only via their blog-post
slugs (Q286, Q357, Q414, Q427, Q451). The survey has 544 questions total; 147
post HTML pages carry heatmap images.

Each post page embeds one or more heatmap images (one per answer choice) as
WordPress uploads.  Image filenames encode the question number and choice index;
for older posts the filename also encodes a shortened answer label.

Outputs:
  data/uwm/questions.csv   one row per published question:
                           question, slug, url, text, image_urls, n_images
  data/raw/uwm/images/     all heatmap PNG files
"""

import csv
import html as htmllib
import re
from pathlib import Path

from common import fetch, out_dir, DATA

SITEMAP = "https://dialectsurvey.wordpress.com/sitemap.xml"
SURVEY_Q = "https://dialectsurvey.wordpress.com/survey-questions/"

# Noise words to ignore when comparing question text (for crosswalk, not here).
_STOPWORDS = frozenset(
    "a an the do does did how what which would you say your use call"
    " is are was were be been have has had do does that this"
    " i me my we our they them their it its of in on to for"
    " with at from as by or and if not no yes".split()
)


def _clean(fragment):
    """Strip HTML tags, unescape entities, collapse whitespace."""
    s = re.sub(r"<[^>]+>", "", fragment)
    s = htmllib.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _load_canonical_texts():
    """Return {1: text, 2: text, ...} from the survey-questions index page."""
    page = fetch(SURVEY_Q, "uwm", "survey_questions.html")
    m = re.search(r"<article[^>]*>(.*?)</article>", page, re.S)
    if not m:
        raise RuntimeError("could not find <article> in survey-questions page")
    content = m.group(1)
    lis = re.findall(r"<li[^>]*>(.*?)</li>", content, re.S)
    out = {}
    idx = 1
    for li in lis:
        clean = _clean(li)
        # strip markdown italic underscores the blog uses for emphasis
        clean = clean.replace("_", "")
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue
        if "Share on" in clean or "Opens in new window" in clean:
            continue
        out[idx] = clean
        idx += 1
        if idx > 544:
            break
    return out


def _parse_post(url):
    """Fetch a post and return (q_numbers, question_texts, image_url_list).

    q_numbers  : list of int, UWM question numbers covered by this post
    h1_text    : full h1 text (e.g. 'Q178 How do you pronounce ...')
    image_urls : list of unique image URLs, in document order
    """
    slug = url.rstrip("/").split("/")[-1]
    html = fetch(url, "uwm", f"{slug[:60]}.html")

    # --- Question H1 heading ------------------------------------------------
    h1_match = None
    for pat in [r"<h1[^>]*>\s*(Q\d.*?)</h1>", r"<h2[^>]*>\s*(Q\d.*?)</h2>",
                r"<h3[^>]*>\s*(Q\d.*?)</h3>"]:
        m = re.search(pat, html, re.S)
        if m:
            h1_match = _clean(m.group(1))
            break

    # parse q numbers from the H1: handles "Q529 ...", "Q94-97 ..."
    q_numbers = []
    if h1_match:
        m2 = re.match(r"Q(\d+)(?:-(\d+))?", h1_match)
        if m2:
            lo = int(m2.group(1))
            hi = int(m2.group(2)) if m2.group(2) else lo
            q_numbers = list(range(lo, hi + 1))

    # fallback: parse from URL slug
    if not q_numbers:
        m3 = re.search(r"/q(\d+)(?:-(\d+))?-", url)
        if m3:
            lo = int(m3.group(1))
            hi = int(m3.group(2)) if m3.group(2) else lo
            q_numbers = list(range(lo, hi + 1))

    # --- Heatmap images -------------------------------------------------------
    # WordPress injects both data-orig-file (full-res) and src (resized).
    # Collect data-orig-file values that point to heatmap PNGs.
    raw_imgs = re.findall(
        r'data-orig-file="(https://dialectsurvey\.wordpress\.com'
        r'/wp-content/uploads/[^"]+\.png)"', html
    )
    # Preserve insertion order, deduplicate
    seen = set()
    image_urls = []
    for img in raw_imgs:
        if "heatmap" in img.lower() and img not in seen:
            seen.add(img)
            image_urls.append(img)

    return q_numbers, h1_match or "", image_urls


def _images_for_question(q_num, all_image_urls):
    """Filter a post's image list down to those for a specific question number.

    Image filenames look like:
      heatmap-{Q}-{choice}-{text}.png  (older style)
      heatmap.{Q}.{choice}.png         (newer style, no choice text)
      heatmap-{Q}-{choice}.png         (short form)
    We match any image whose first numeric field equals q_num.
    """
    matched = []
    for url in all_image_urls:
        fname = url.split("/")[-1]
        # strip extension
        base = fname.rsplit(".", 1)[0]
        # find the first run of digits
        m = re.search(r"(\d+)", base)
        if m and int(m.group(1)) == q_num:
            matched.append(url)
    # If no explicit match (single-question post), return all
    return matched if matched else all_image_urls


def _answer_label_from_image_url(url, img_title):
    """Extract a human-readable answer label from the image URL filename or title.

    Older posts embed the label in the filename:
      heatmap-1-1-bin-not-ben.png → 'bin not ben'
    The data-image-title attribute (if passed) may also carry it:
      'heatmap.1.1 bin not ben' → 'bin not ben'
    Returns empty string when no label is recoverable.
    """
    # Try the data-image-title first (more reliable)
    if img_title:
        m = re.match(r"heatmap[.\-]\d+[.\-]\d+(?:[.\-]\d+)?\s+(.*)", img_title, re.I)
        if m:
            return m.group(1).strip()

    # Try the filename
    fname = url.split("/")[-1].rsplit(".", 1)[0]  # strip .png
    # strip leading 'heatmap-{q}-{choice}-' or 'heatmap.{q}.{choice}'
    fname_clean = re.sub(r"heatmap[-.]?\d+[-.]?\d+[-.]?", "", fname, flags=re.I)
    fname_clean = fname_clean.lstrip("-.")
    if fname_clean:
        return fname_clean.replace("-", " ").strip()
    return ""


def main():
    d = out_dir("uwm")
    img_cache = "uwm/images"

    # 1) Load canonical question texts from the survey-questions index
    print("loading canonical question texts …")
    canon = _load_canonical_texts()
    print(f"  {len(canon)} question texts (q1–q{max(canon)})")

    # 2) Collect all post URLs from the sitemap
    print("fetching sitemap …")
    sitemap = fetch(SITEMAP, "uwm", "sitemap.xml")
    all_urls = re.findall(
        r"<loc>(https://dialectsurvey\.wordpress\.com/[^<]+)</loc>", sitemap
    )

    # Keep only map-bearing posts (exclude static pages and non-map posts)
    SKIP = {"survey-questions", "maps", "contact"}
    post_urls = [u for u in all_urls
                 if not any(u.rstrip("/").endswith(p) for p in SKIP)]
    print(f"  {len(post_urls)} post URLs to process")

    # 3) Parse every post
    questions = {}  # q_num -> dict row

    for i, url in enumerate(post_urls):
        slug_full = url.rstrip("/").split("/")[-1]
        print(f"  [{i+1}/{len(post_urls)}] {slug_full[:60]}")
        q_nums, h1_text, img_urls = _parse_post(url)
        if not q_nums:
            print(f"    WARN: no question numbers found — skipping {url}")
            continue

        for qn in q_nums:
            if qn in questions:
                # Prefer the first encounter (earlier URL is usually canonical)
                continue
            q_imgs = _images_for_question(qn, img_urls)
            text = canon.get(qn, "")
            if not text:
                # Fall back to H1 text with prefix stripped
                h1_clean = re.sub(r"^Q\d+(?:-\d+)?\s*", "", h1_text).strip()
                text = h1_clean
            questions[qn] = {
                "question": qn,
                "slug": slug_full,
                "url": url,
                "text": text,
                "image_urls": ";".join(q_imgs),
                "n_images": len(q_imgs),
            }

    # 4) Download every heatmap image
    print(f"\ndownloading {sum(r['n_images'] for r in questions.values())} images …")
    for row in sorted(questions.values(), key=lambda r: r["question"]):
        for img_url in row["image_urls"].split(";"):
            if not img_url:
                continue
            fname = img_url.split("/")[-1]
            fetch(img_url, img_cache, fname, binary=True)

    # 5) Write questions.csv
    sorted_qs = sorted(questions.values(), key=lambda r: r["question"])
    out_path = d / "questions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["question", "slug", "url", "text", "image_urls",
                               "n_images"])
        w.writeheader()
        w.writerows(sorted_qs)

    qs = sorted(questions.keys())
    print(f"\nwrote {out_path}: {len(sorted_qs)} questions")
    print(f"  q numbers: {qs[0]}..{qs[-1]}, total images "
          f"{sum(r['n_images'] for r in sorted_qs)}")
    print(f"  questions with 0 images: "
          f"{sum(1 for r in sorted_qs if r['n_images'] == 0)}")


if __name__ == "__main__":
    main()
