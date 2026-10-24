"""Pure-python parts of pipeline/compare_checkpoints.py (no torch needed)."""
from pipeline import compare_checkpoints as C


def test_fk_grade_orders_hard_above_easy():
    easy = "The cat sat on the mat. It was a sunny day."
    hard = "Intraoperative chemoradiotherapy demonstrated significantly improved locoregional control."
    assert C.fk_grade(hard) > C.fk_grade(easy) + 5
    assert C.fk_grade("") == 0.0


def test_count_syllables():
    assert [C.count_syllables(w) for w in ("cat", "cancer", "make", "table")] == [1, 2, 1, 2]


def test_numbers_faithful():
    src = "In stage III disease, 45% of 1,200 patients responded."
    assert C.numbers_faithful(src, "Of 1200 people, 45% got better in stage 3.")
    assert not C.numbers_faithful(src, "About 50% got better.")
    assert not C.numbers_faithful(src, "This is stage II disease.")
