"""
ULTRON Audio — English Speech & Hallucination Validator
========================================================
Validates that transcribed text is genuine, spoken English and filters out:
- Whisper silence/noise hallucinations (e.g. "You", "Thank you.", "Thanks for watching.")
- Non-speech sound descriptions (e.g. "[Music]", "(Applause)", "*laughter*")
- Foreign non-English scripts
- Non-linguistic phonetic artifacts (e.g. "tktk", "brrr", "shhh", "zzzz")
"""

import re

# Common hallucinations Whisper outputs when presented with silence, static, or noise
# Unconditional hallucinations (YouTube training artifacts, never valid in security system)
UNCONDITIONAL_HALLUCINATIONS = {
    "thanks for watching",
    "thanks for watching.",
    "thank you for watching",
    "thank you for watching.",
    "subscribe",
    "subtitles by",
    "subtitles by amara.org",
    "closed captioning",
    "viewers like you",
    "music",
    "applause",
    "silence",
    "transcription by",
}

# Ambiguous short single words that are often hallucinated on low-volume clicks/noise
SUSPICIOUS_SINGLE_TOKENS = {
    "you",
    "thank you",
    "thank you.",
    "bye",
    "bye.",
    "bye bye",
    "okay",
    "okay.",
    "so",
    "oh",
    "ah",
    "um",
    "uh",
}

# Regex to strip bracketed or parenthesized sound effects
RE_SOUND_EFFECTS = re.compile(r"\[.*?\]|\(.*?\)|<.*?>|\*.*?\*")

# Regex to detect any non-Latin / non-ASCII character (e.g., Cyrillic, Chinese, Arabic, Devanagari)
RE_NON_LATIN = re.compile(r"[^\x00-\x7F]")

# Regex to find word tokens
RE_WORDS = re.compile(r"[a-zA-Z]+")

# Vowel check (English words require at least one vowel or 'y')
RE_VOWELS = re.compile(r"[aeiouyAEIOUY]")


def validate_english_speech(
    text: str,
    avg_logprob: float = 0.0,
    no_speech_prob: float = 0.0,
) -> str:
    """
    Validate and clean transcribed speech.

    Args:
        text: Raw transcription string from Whisper
        avg_logprob: Average token log probability
        no_speech_prob: Probability that the segment contains no speech

    Returns:
        Cleaned English string, or empty string "" if rejected as noise/invalid.
    """
    if not text:
        return ""

    # 1. Reject if Whisper's acoustic confidence indicates high probability of non-speech
    if no_speech_prob > 0.40:
        return ""

    # 2. Reject if overall log probability is too low (mumbled noise / hallucination)
    if avg_logprob < -0.85:
        return ""

    # 3. Strip sound effect tags: [Music], (Applause), *giggle*, etc.
    cleaned = RE_SOUND_EFFECTS.sub("", text).strip()
    if not cleaned:
        return ""

    # 4. Strict English filter: reject if any non-ASCII / foreign script characters appear
    if RE_NON_LATIN.search(cleaned):
        return ""

    # 5. Check against known Whisper hallucinations
    normalized = cleaned.lower().strip(" .,!?:;\"'-\t\n")
    if normalized in UNCONDITIONAL_HALLUCINATIONS:
        return ""
    if normalized in SUSPICIOUS_SINGLE_TOKENS:
        # Require strong acoustic confidence for isolated single words like 'you' or 'okay'
        if no_speech_prob > 0.15 or avg_logprob < -0.45:
            return ""

    # 6. Extract individual words and validate linguistic structure
    words = RE_WORDS.findall(cleaned)
    if not words:
        return ""

    # Must have at least one word with length >= 2
    if not any(len(w) >= 2 for w in words):
        return ""

    # Every valid English word must contain at least one vowel or 'y'
    # Filters out noise artifacts like 'tktk', 'grrr', 'psst', 'clck'
    vowel_words = [w for w in words if RE_VOWELS.search(w)]
    if len(vowel_words) < len(words) * 0.7:
        return ""

    # Clean redundant whitespace
    final_text = " ".join(cleaned.split())
    return final_text
