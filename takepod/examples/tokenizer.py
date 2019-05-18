PUNCTUATION_SYMBOLS = {'.', '?', '!', ',', ':', ';'}
SENTENCE_END_SYMBOLS = {'.', '?', '!'}
QUOTE_SYMBOLS = {'"', "'", "«", '»'}
PARENTHESES = {'(', ')'}


def _should_trim(character):
    return character in PUNCTUATION_SYMBOLS \
        or character in QUOTE_SYMBOLS \
        or character in PARENTHESES


def tokenize(text):
    text = text.replace('\n', ' ')
    tokens = text.split()

    processed_tokens = []
    for token_text in tokens:

        # split start punctuation, quote or parentheses
        while _should_trim(token_text[0]):
            processed_tokens.append(token_text[0])
            token_text = token_text[1:]
            if len(token_text) == 0:
                break

        if len(token_text) == 0:
            continue

        # split end punctuation, quote or parentheses,
        # but append later reversed
        append_end = []
        while _should_trim(token_text[-1]):
            append_end.append(token_text[-1])
            token_text = token_text[:-1]
            if len(token_text) == 0:
                break

        if len(token_text) > 0:
            processed_tokens.append(token_text)

        append_end.reverse()
        processed_tokens += append_end

    return processed_tokens
