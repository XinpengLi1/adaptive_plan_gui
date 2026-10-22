from adaptive_plan.data.schema import Action
from adaptive_plan.match import match_actions


def test_type_mismatch():
    assert not match_actions(Action("click"), Action("scroll")).matched


def test_click_long_press_equivalent_lenient():
    assert match_actions(Action("long_press", 1, 1), Action("click", 500, 500)).matched


def test_type_text_normalised():
    assert match_actions(Action("type", text=" Coffee  Shops "), Action("type", text="coffee shops")).matched
    assert not match_actions(Action("type", text="tea"), Action("type", text="coffee")).matched


def test_scroll_direction():
    assert match_actions(Action("scroll", direction="DOWN"), Action("scroll", direction="down")).matched
    assert not match_actions(Action("scroll", direction="up"), Action("scroll", direction="down")).matched


def test_strict_bbox():
    gold = Action("click", 540, 180, bbox=[400, 150, 680, 210])
    assert match_actions(Action("click", 600, 200), gold, strict=True).matched
    assert not match_actions(Action("click", 100, 100), gold, strict=True).matched


def test_strict_radius_without_bbox():
    gold = Action("click", 540, 180)
    assert match_actions(Action("click", 560, 190), gold, strict=True, screen_w=1080, screen_h=2400).matched
    assert not match_actions(Action("click", 540, 2000), gold, strict=True, screen_w=1080, screen_h=2400).matched


def test_strict_missing_point_fails():
    assert not match_actions(Action("click"), Action("click", 1, 1, bbox=[0, 0, 2, 2]), strict=True).matched
