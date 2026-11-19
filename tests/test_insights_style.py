import pytest

from wine_crew.tools.eda_tools import INSIGHT_HEADINGS, OPENING_LINE, insight_style_violations

H1, H2, H3, H4 = INSIGHT_HEADINGS

SUMMARY = {
    "by_country": [{"name": "US"}, {"name": "Austria"}, {"name": "Chile"}],
    "by_variety": [{"name": "Red Blend"}, {"name": "Bordeaux-style Red Blend"}, {"name": "Riesling"}],
    "value_countries": [{"name": "Austria"}],
    "value_varieties": [{"name": "Riesling"}, {"name": "Bordeaux-style Red Blend"}],
    "price_bands": {"plateau_from_price": 500.0, "plateau_gain_per_doubling": 0.74},
    "overall_median_points": 88.0,
    "overall_median_price": 25.0,
    "country_bias": {"top_country": "US", "top_country_share_pct": 45.04, "top3_share_pct": 73.55,
                     "top3_countries": ["US", "France", "Italy"]},
    "missing_pct": {"region_1": 16.1, "designation": 28.73},
}
PLATEAU_LINE = "החל מ-$500, כל הכפלה של המחיר מוסיפה רק 0.74 נקודות ניקוד."
DEFINITION = ("נכללו מדינות וזנים שהניקוד החציוני שלהם הוא 88 ומעלה והמחיר החציוני שלהם הוא $25 ומטה. "
              "אלה הניקוד החציוני והמחיר החציוני של כלל היינות.")
BIAS_LINE = "- 45% מהיינות בנתונים הם מ-US, ו-74% מגיעים משלוש המדינות המובילות: US, France ו-Italy."
MISSING_LINE = "- ערכים חסרים: ל-16% מהיינות חסרים נתוני אזור, ול-29% חסר שם הסדרה/היין."
REC_COUNTRIES = "- מדינות שכדאי להתחיל מהן: **Austria**."
REC_VARIETIES = "- זנים שכדאי להתחיל מהם: **Riesling** ו-**Bordeaux-style Red Blend**."
S3 = f"{BIAS_LINE}\n{MISSING_LINE}"


def value_section(countries="- **Austria**: ניקוד חציוני 90, מחיר חציוני $25",
                  varieties="- **Riesling**: ניקוד חציוני 90, מחיר חציוני $22"):
    return f"{DEFINITION}\n### מדינות\n{countries}\n### זנים\n{varieties}"


def doc(s1=f"{OPENING_LINE}\n- {PLATEAU_LINE}", s2=None, s3=S3,
        s4=f"{REC_COUNTRIES}\n{REC_VARIETIES}\n- {PLATEAU_LINE}"):
    s2 = value_section() if s2 is None else s2
    return f"{H1}\n{s1}\n\n{H2}\n{s2}\n\n{H3}\n{s3}\n\n{H4}\n{s4}\n"


def test_valid_document_passes():
    assert insight_style_violations(doc(), SUMMARY) == []


def test_missing_heading():
    text = doc().replace(H3 + "\n", "")
    assert f"missing heading: {H3}" in insight_style_violations(text, SUMMARY)


def test_headings_out_of_order():
    text = doc()
    swapped = text.replace(H1, "@@").replace(H2, H1).replace("@@", H2)
    assert "headings are not in the required order" in insight_style_violations(swapped, SUMMARY)


@pytest.mark.parametrize("written, rule", [
    ("מפסיק לעלות", "מפסיק לעלות"),
    ("האזור הראשון", "אזור הראשון"),
    ("באזור הראשון", "אזור הראשון"),       # the prefix variant the last run used
    ("קטגוריית ההגדרה", "קטגוריית ההגדרה"),
    ("בעמודת ההגדרה", "עמודת ההגדרה"),     # the prefix variant the last run used
    ("דירוג", "דירוג"),
    ("מעל טווח זה", "מעל טווח זה"),
])
def test_each_forbidden_phrase_is_caught(written, rule):
    violations = insight_style_violations(doc(s3=f"{S3}\n- {written} כאן."), SUMMARY)
    assert any(v.startswith(f"do not write '{rule}'") for v in violations)


def test_word_hagdara_alone_is_allowed():
    assert insight_style_violations(doc(s3=f"{S3}\n- הגדרה של יין טוב."), SUMMARY) == []


def test_trailing_zero_and_long_percent_are_caught_but_whole_percent_passes():
    s2 = value_section(countries="- **Austria**: ניקוד חציוני 90.0, מחיר חציוני $25")
    violations = insight_style_violations(doc(s2=s2, s3=f"{S3}\n- 45.04% מ-US."), SUMMARY)
    assert "write 90.0 as a whole number (90)" in violations
    assert "round the percentage 45.04% to a whole number" in violations
    assert insight_style_violations(doc(s3=f"{S3}\n- 45% מ-US."), SUMMARY) == []


def test_value_section_only_names_value_countries():
    s2 = value_section(countries="- **Austria**\n- **Chile**")
    assert insight_style_violations(doc(s2=s2), SUMMARY) == [
        "Chile is not in value_countries or value_varieties; name only those"
    ]
    assert insight_style_violations(doc(s2=value_section(countries="- **Austria**")), SUMMARY) == []


def test_other_sections_may_name_any_country():
    assert insight_style_violations(doc(s3=f"{S3}\n- Chile ו-US."), SUMMARY) == []


def test_red_blend_not_matched_inside_bordeaux_style_red_blend():
    ok = value_section(varieties="- **Bordeaux-style Red Blend**\n- **Riesling**")
    assert insight_style_violations(doc(s2=ok), SUMMARY) == []
    bad = value_section(varieties="- **Red Blend**")
    assert insight_style_violations(doc(s2=bad), SUMMARY) == [
        "Red Blend is not in value_countries or value_varieties; name only those"
    ]


@pytest.mark.parametrize("sub", ["### מדינות", "### זנים"])
def test_value_subheadings_required(sub):
    text = doc().replace(sub + "\n", "")
    assert f"section 2 needs the sub-heading '{sub}'" in insight_style_violations(text, SUMMARY)


@pytest.mark.parametrize("term", ["נתוני אזור", "שם הסדרה/היין"])
def test_missing_values_template_required(term):
    text = doc().replace(term, "משהו")
    assert any(term in v for v in insight_style_violations(text, SUMMARY))


@pytest.mark.parametrize("heading, kwargs", [
    (H1, {"s1": f"{OPENING_LINE}\n- הניקוד עולה, אבל מעל $500 העלייה קטנה."}),
    (H4, {"s4": f"{REC_COUNTRIES}\n{REC_VARIETIES}\n- כסף שווה יותר בטווחים נמוכים."}),
])
def test_plateau_template_required_in_section_1_and_recommendations(heading, kwargs):
    violations = insight_style_violations(doc(**kwargs), SUMMARY)
    assert any(v.startswith(f"section '{heading}' must use the template 'החל מ-$500") for v in violations)


def test_plateau_template_skipped_when_summary_has_no_plateau():
    summary = {k: v for k, v in SUMMARY.items() if k != "price_bands"}
    assert insight_style_violations(doc(s1=OPENING_LINE), summary) == []


# --------------------------------------------------------------------------- fixed template lines


@pytest.mark.parametrize("line, rule", [
    (OPENING_LINE, "section 1 must start with the line"),
    (DEFINITION, "section 2 must use the definition template"),
    (BIAS_LINE, "section 3 must use the template '- <top_country_share_pct>%"),
    (MISSING_LINE, "section 3 must use the template '- ערכים חסרים"),
    (REC_COUNTRIES, "section 4 must include the line '- מדינות שכדאי להתחיל מהן"),
    (REC_VARIETIES, "section 4 must include the line '- זנים שכדאי להתחיל מהם"),
])
def test_each_fixed_line_is_required(line, rule):
    violations = insight_style_violations(doc().replace(line, "- משהו אחר."), SUMMARY)
    assert any(v.startswith(rule) for v in violations), violations


@pytest.mark.parametrize("bad, expected", [
    (BIAS_LINE.replace("US, France ו-Italy", "US, Spain ו-Italy"), "top three countries as 'US, France ו-Italy'"),
    (BIAS_LINE.replace("74%", "73%"), "top3_share_pct rounded"),
    (BIAS_LINE.replace("45%", "40%"), "top_country_share_pct rounded"),
    (BIAS_LINE.replace("מ-US,", "מ-France,"), "name the top country as US"),
])
def test_bias_line_numbers_and_names_are_checked(bad, expected):
    violations = insight_style_violations(doc(s3=f"{bad}\n{MISSING_LINE}"), SUMMARY)
    assert any(expected in v for v in violations), violations


def test_top3_share_must_be_a_whole_number():
    long = BIAS_LINE.replace("74%", "73.55%")
    violations = insight_style_violations(doc(s3=f"{long}\n{MISSING_LINE}"), SUMMARY)
    assert "round the percentage 73.55% to a whole number" in violations
    assert any(v.startswith("section 3 must use the template '- <top_country_share_pct>%") for v in violations)


def test_definition_and_missing_numbers_are_checked():
    s2 = value_section().replace("הוא 88 ומעלה", "הוא 85 ומעלה")
    assert any("overall_median_points" in v for v in insight_style_violations(doc(s2=s2), SUMMARY))
    s3 = f"{BIAS_LINE}\n" + MISSING_LINE.replace("ל-16%", "ל-20%")
    assert any("missing_pct.region_1" in v for v in insight_style_violations(doc(s3=s3), SUMMARY))


def test_recommendations_name_the_first_two_value_entries():
    wrong = REC_VARIETIES.replace("**Bordeaux-style Red Blend**", "**Red Blend**")
    violations = insight_style_violations(doc(s4=f"{REC_COUNTRIES}\n{wrong}\n- {PLATEAU_LINE}"), SUMMARY)
    assert any("the first two entries of value_varieties" in v for v in violations)
    trailing = doc().replace(REC_COUNTRIES, REC_COUNTRIES + "  ")
    assert insight_style_violations(trailing, SUMMARY) == []


def test_new_document_passes_the_number_check():
    from wine_crew.tools.eda_tools import insight_number_violations

    summary = {**SUMMARY, "value_countries": [{"name": "Austria", "median_points": 90, "median_price": 25}],
               "value_varieties": [{"name": "Riesling", "median_points": 90, "median_price": 22}]}
    assert insight_number_violations(doc(), summary) == []
