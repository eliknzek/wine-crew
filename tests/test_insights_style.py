import pytest

from wine_crew.tools.eda_tools import INSIGHT_HEADINGS, insight_style_violations

H1, H2, H3, H4 = INSIGHT_HEADINGS

SUMMARY = {
    "by_country": [{"name": "US"}, {"name": "Austria"}, {"name": "Chile"}],
    "by_variety": [{"name": "Red Blend"}, {"name": "Bordeaux-style Red Blend"}, {"name": "Riesling"}],
    "value_countries": [{"name": "Austria"}],
    "value_varieties": [{"name": "Riesling"}, {"name": "Bordeaux-style Red Blend"}],
}


def doc(s1="- העלייה בניקוד מאטה מעל $500.", s2="- Austria: ניקוד חציוני 90 ב-$25.",
        s3="- 45% מהיינות מ-US.", s4="- המלצה."):
    return f"{H1}\n{s1}\n\n{H2}\n{s2}\n\n{H3}\n{s3}\n\n{H4}\n{s4}\n"


def test_valid_document_passes():
    assert insight_style_violations(doc(), SUMMARY) == []


def test_missing_heading():
    text = doc().replace(H3 + "\n", "")
    assert f"missing heading: {H3}" in insight_style_violations(text, SUMMARY)


def test_headings_out_of_order():
    text = f"{H2}\n- Austria\n\n{H1}\n- x\n\n{H3}\n- y\n\n{H4}\n- z\n"
    assert "headings are not in the required order" in insight_style_violations(text, SUMMARY)


@pytest.mark.parametrize("phrase", ["מפסיק לעלות", "האזור הראשון", "קטגוריית ההגדרה", "דירוג"])
def test_each_forbidden_phrase_is_caught(phrase):
    violations = insight_style_violations(doc(s3=f"- {phrase} כאן."), SUMMARY)
    assert any(v.startswith(f"do not write '{phrase}'") for v in violations)


def test_word_hagdara_alone_is_allowed():
    assert insight_style_violations(doc(s3="- הגדרה של יין טוב."), SUMMARY) == []


def test_trailing_zero_and_long_percent_are_caught_but_whole_percent_passes():
    violations = insight_style_violations(doc(s2="- Austria: 90.0 ב-$25.", s3="- 45.04% מ-US."), SUMMARY)
    assert "write 90.0 as a whole number (90)" in violations
    assert "round the percentage 45.04% to a whole number" in violations
    assert insight_style_violations(doc(s3="- 45% מ-US."), SUMMARY) == []


def test_value_section_only_names_value_countries():
    violations = insight_style_violations(doc(s2="- Austria ו-Chile משתלמות."), SUMMARY)
    assert violations == ["Chile is not in value_countries or value_varieties; name only those"]
    assert insight_style_violations(doc(s2="- Austria משתלמת."), SUMMARY) == []


def test_other_sections_may_name_any_country():
    assert insight_style_violations(doc(s3="- Chile ו-US הן חלק גדול מהנתונים."), SUMMARY) == []


def test_red_blend_not_matched_inside_bordeaux_style_red_blend():
    text = doc(s2="- Bordeaux-style Red Blend ו-Riesling.")
    assert insight_style_violations(text, SUMMARY) == []
    caught = insight_style_violations(doc(s2="- Red Blend משתלם."), SUMMARY)
    assert caught == ["Red Blend is not in value_countries or value_varieties; name only those"]
