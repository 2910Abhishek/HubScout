"""Language-code matching: Hub cards mix ISO 639-1 ('hi') and ISO 639-3 ('hin') codes."""

from __future__ import annotations

ISO3_TO_ISO1 = {
    "ara": "ar", "ben": "bn", "deu": "de", "ger": "de", "eng": "en", "spa": "es", "fas": "fa",
    "per": "fa", "fra": "fr", "fre": "fr", "guj": "gu", "hin": "hi", "ind": "id", "ita": "it",
    "jpn": "ja", "kan": "kn", "kor": "ko", "mal": "ml", "mar": "mr", "nld": "nl", "dut": "nl",
    "pan": "pa", "pol": "pl", "por": "pt", "rus": "ru", "swa": "sw", "tam": "ta", "tel": "te",
    "tha": "th", "tur": "tr", "ukr": "uk", "urd": "ur", "vie": "vi", "zho": "zh", "chi": "zh",
}  # fmt: skip


def normalize(code: str) -> str:
    code = code.strip().lower().split("-")[0].split("_")[0]
    return ISO3_TO_ISO1.get(code, code)


def lacks_key_language(required: list[str], available: list[str]) -> bool:
    """True if the card lists languages but none of the ones that matter.

    English appears on almost every card, so when other languages are required (e.g. Hindi for
    Hindi-English), at least one of THOSE must be listed; English alone is not enough.
    """
    if not required or not available:
        return False
    key = [lang for lang in required if normalize(lang) != "en"] or required
    return len(missing_languages(key, available)) == len(key)


def missing_languages(required: list[str], available: list[str]) -> list[str]:
    have = {normalize(c) for c in available}
    return [lang for lang in required if normalize(lang) not in have]
