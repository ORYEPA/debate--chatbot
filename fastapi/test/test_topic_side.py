from app import config

def test_universal_system_prompt_contract():
    usp = config.UNIVERSAL_SYSTEM_PROMPT

    assert "Return ONLY a valid JSON object" in usp
    assert '"stance":"pro|contra"' in usp

    assert "never neutral" in usp.lower()

    low = usp.lower()
    assert ("prefixes like 'stance:'" in low) or ("no prefixes" in low) or ("no prefix" in low)
