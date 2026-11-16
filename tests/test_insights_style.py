import pytest

from wine_crew.tools.eda_tools import INSIGHT_HEADINGS, insight_style_violations

H1, H2, H3, H4 = INSIGHT_HEADINGS

SUMMARY = {
    "by_country": [{"name": "US"}, {"name": "Austria"}, {"name": "Chile"}],
    "by_variety": [{"name": "Red Blend"}, {"name": "Bordeaux-style Red Blend"}, {"name": "Riesling"}],
    "value_countries": [{"name": "Austria"}],
    "value_varieties": [{"name": "Riesling"}, {"name": "Bordeaux-style Red Blend"}],
    "price_bands": {"plateau_from_price": 500.0, "plateau_gain_per_doubling": 0.74},
}
PLATEAU_LINE = "החל מ-$500, כל הכפלה של המחיר מוסיפה רק 0.74 נקודות ניקוד."
MISSING_LINE = "- ערכים חסרים: 16% מהיינות ללא נתוני אזור, ו-29% ללא שם הסדרה/היין."


def value_section(countries="- **Austria**: ניקוד חציוני 90, מחיר חציוני $25",
                  varieties="- **Riesling**: ניקוד חציוני 90, מחיר חציוני $22"):
    return f"ההגדרה: ניקוד חציוני של 88 ומעלה.\n### מדינות\n{countries}\n### זנים\n{varieties}"


def doc(s1=f"הניקוד ממשיך לעלות עם המחיר. {PLATEAU_LINE}", s2=None,
        s3=f"- 45% מהיינות מ-US.\n{MISSING_LINE}", s4=f"- המלצה.\n- המלצה.\n- {PLATEAU_LINE}"):
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
    violations = insight_style_violations(doc(s3=f"- {written} כאן.\n{MISSING_LINE}"), SUMMARY)
    assert any(v.startswith(f"do not write '{rule}'") for v in violations)


def test_word_hagdara_alone_is_allowed():
    assert insight_style_violations(doc(s3=f"- הגדרה של יין טוב.\n{MISSING_LINE}"), SUMMARY) == []


def test_trailing_zero_and_long_percent_are_caught_but_whole_percent_passes():
    s2 = value_section(countries="- **Austria**: ניקוד חציוני 90.0, מחיר חציוני $25")
    violations = insight_style_violations(doc(s2=s2, s3=f"- 45.04% מ-US.\n{MISSING_LINE}"), SUMMARY)
    assert "write 90.0 as a whole number (90)" in violations
    assert "round the percentage 45.04% to a whole number" in violations
    assert insight_style_violations(doc(s3=f"- 45% מ-US.\n{MISSING_LINE}"), SUMMARY) == []


def test_value_section_only_names_value_countries():
    s2 = value_section(countries="- **Austria**\n- **Chile**")
    assert insight_style_violations(doc(s2=s2), SUMMARY) == [
        "Chile is not in value_countries or value_varieties; name only those"
    ]
    assert insight_style_violations(doc(s2=value_section(countries="- **Austria**")), SUMMARY) == []


def test_other_sections_may_name_any_country():
    assert insight_style_violations(doc(s3=f"- Chile ו-US.\n{MISSING_LINE}"), SUMMARY) == []


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
    (H1, {"s1": "הניקוד עולה, אבל מעל $500 העלייה קטנה."}),
    (H4, {"s4": "- המלצה.\n- המלצה.\n- כסף שווה יותר בטווחים נמוכים."}),
])
def test_plateau_template_required_in_section_1_and_recommendations(heading, kwargs):
    violations = insight_style_violations(doc(**kwargs), SUMMARY)
    assert any(v.startswith(f"section '{heading}' must use the template 'החל מ-$500") for v in violations)


def test_plateau_template_skipped_when_summary_has_no_plateau():
    summary = {k: v for k, v in SUMMARY.items() if k != "price_bands"}
    assert insight_style_violations(doc(s1="הניקוד עולה."), summary) == []
