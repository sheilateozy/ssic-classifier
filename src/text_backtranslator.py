from deep_translator import GoogleTranslator


def back_translate(text: str) -> str:
    '''
    Implements back-translation to obtain additional training text data:
    Translates a text from the original language (English) to a target language using Google Translate
    and then translates it back to the original language.
    The resulting text will be somewhat different from the original text due to translation "noise",
    but it will convey the same general meaning. 
    This provides a way to artificially generate new text for model training that are variations of the text in the original training data.
    
    
    Returns
    -------
    backtranslated_text: str
        The resulting text from back-translation
    '''

    translation1 = GoogleTranslator('en', 'zh-CN').translate(text)
    translation2 = GoogleTranslator('zh-CN', 'ar').translate_batch(translation1)
    backtranslated_text = GoogleTranslator('ar', 'en').translate_batch(translation2)
    return backtranslated_text