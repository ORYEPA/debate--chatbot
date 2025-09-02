from app.service.profiles import PROFILE, AVAILABLE_PROFILES


def test_profiles_exist_and_have_style():
    assert "smart_shy" in AVAILABLE_PROFILES
    assert "conspiracy_edge" in AVAILABLE_PROFILES
    assert "rude_arrogant" in AVAILABLE_PROFILES

    for pid, p in PROFILE.items():
        assert isinstance(p["name"], str) and p["name"]
        assert isinstance(p["system"], str) and p["system"]
        assert "Stance:" not in p["system"]
        assert isinstance(p["style"], dict)
        for k in ("temperature", "top_p", "num_predict"):
            assert k in p["style"]
