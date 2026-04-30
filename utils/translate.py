from deep_translator import GoogleTranslator

def is_arabic(text):
    return any('\u0600' <= c <= '\u06FF' for c in text)

def to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

def to_arabic(text):
    return GoogleTranslator(source="auto", target="ar").translate(text)
