# notes

random stuff from building this. not cleaned up

---

## ner

- wanted one model for all 4 langs. wikineural is fine for en/de but no hindi/tamil in training data
- -> route by lang. wikineural en/de, ai4bharat/IndicNER hi/ta
- lang id for ta/hi = just count chars in unicode block. no need for anything clever
- marathi/nepali are also devanagari -> will go to "hi". indicner has marathi anyway so prob ok? haven't checked

- tokenizer splits indic words into pieces sometimes (दिल् + ली). merge spans if they touch or have a `-` between
- NOT across spaces, otherwise "Berlin Hamburg" = one place

- long docs: xlm-r = 512 subword tokens max, so chunk by sentence
- first chunker version did packing + hard splits in one loop, couldn't tell if offsets were right. rewrote as 2 passes (find sentences -> pack them)
- test: `text[off:off+len(chunk)] == chunk` for every chunk. that's the only thing that matters really
- remember `।` (danda) as a sentence end

## tamil suffixes

`சென்னையில்` = "in chennai", one word. nominatim returns nothing

options:
1. lemmatize everything first
2. try as-is, strip suffix only if nothing comes back

went w/ 2. most entities don't need it + stripping could break names that just happen to end like a suffix

suffix changes shape depending on the stem:
- `சென்னை` + `இல்` -> `சென்னையில்` (ய inserted) -> strip `யில்`
- `பெங்களூர்` + `இல்` -> `பெங்களூரில்` (virama gone) -> strip `ில்`, put `்` back

bug: list stops at first match and `க்கு` was before `ுக்கு` so the longer one never hit. longest first now

`isalnum()` is False for tamil/hindi vowel signs (unicode cat M)!! any punctuation stripping / word boundary code eats the vowels if you don't handle it

german: genitive s (Münchens). hindi: postposition sometimes ends up inside the span (मुंबई में)

## frankfurt not found

first offline eval: FN on "Frankfurt" in de-2
- gazetteer word list comes from OSM names, which are "Frankfurt am Main" / "Frankfurt (Oder)". no plain "Frankfurt"
- added OSM `short_name` -> fixed. F1 0.968 -> 0.984

## disambiguation

just taking the highest `importance` = Paris is always France, Frankfurt is always am Main

test sentences:
- "The storm passed over Paris, Texas, about 160 km north-east of Dallas."
- "...erreichte Frankfurt und Słubice; Brandenburg rief den Katastrophenfall aus."

first try: bonus if the candidate's *country* is mentioned. useless here, texas + brandenburg are states
-> containment check instead: is the candidate inside the bbox of a state/region/county mentioned in the text

+ coherence = exp(-d/scale), d = median dist to other mentions' picks

these depend on the other picks so it has to loop till nothing changes. max 2 rounds on demo set (7/9 docs done after 1)

scale 750km -> 400km. at 750 basically everything in germany counts as "close", the two frankfurts are only ~480km apart

ablation (doc features weight 0):

```
              mean err   acc@161   auc
with context    9.0 km    1.000   0.067
without       293.9 km    0.931   0.115
```

both errors = the cases above. paris -> france 7783 km, frankfurt -> am main 478 km
2 mistakes out of 29 and mean error goes up 30x. that's why median/acc@161 are in there too

weights are hand-set. should learn them at some point

## will break

- "floods in chennai, paris and houston". coherence pulls everything to the middle of nowhere. no test for this yet
- maybe turn coherence down when other mentions are far apart?
- importance comes from wikipedia links -> skewed to places w/ good english wiki pages. indian villages prob suffer. not measured
- bbox containment breaks across 180° (fiji etc)

## demo data

- `examples/geocode_cache.json` is hand-written in nominatim's format, coords approximate. NOT real api output
- point was offline + same result every time for tests/ci
- put the nasty ones in on purpose: 2 frankfurts, 3 parises, hyderabad IN/PK, berlin NH
- `build_demo_data.py` makes cache + gold. offsets via `str.find`, not typing tamil offsets by hand
- rivers (Oder, गंगा) = span only, no gold point. "distance error" for a 900km river means nothing
- kanchipuram left out of the gazetteer on purpose. baseline should miss something

## map png

- matplotlib font has no tamil glyphs -> labels = resolved english name
- state bboxes covered whole panel on first render, now only draw boxes < 3°

## todo

- [ ] run wikineural/indicner for real on LGL or GeoWebNews. right now neural path only has mocked tests and every number is from the gazetteer baseline on 9 docs
- [ ] learn weights
- [ ] antimeridian
- [ ] polygons for areas/rivers (nominatim `polygon_geojson`)
