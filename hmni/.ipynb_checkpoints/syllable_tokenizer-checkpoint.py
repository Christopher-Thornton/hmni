import re


def syllables(word):
    # single syllable word
    if len(re.findall('[aeiouy]', word)) <= 1:
        return [word]

    # sonority hierarchy: vowels, nasals, fricatives, stops
    hierarchy = {
        'a': 4, 'e': 4, 'i': 4, 'o': 4, 'u': 4, 'y': 4,
        'l': 3, 'm': 3, 'n': 3, 'r': 3, 'w': 3,
        'f': 2, 's': 2, 'v': 2, 'z': 2,
        'b': 1, 'c': 1, 'd': 1, 'g': 1, 'h': 1, 'j': 1, 'k': 1, 'p': 1, 'q': 1, 't': 1, 'x': 1,
    }
    syllables_values = [(c, hierarchy[c]) for c in word]

    syllables = []
    syll = syllables_values[0][0]
    for trigram in zip(*[syllables_values[i:] for i in range(3)]):
        (phonemes, values) = zip(*trigram)
        (previous, val, following) = values
        phoneme = phonemes[1]

        if previous > val < following:
            syllables.append(syll)
            syll = phoneme
        elif previous >= val == following:
            syll += phoneme
            syllables.append(syll)
            syll = ''
        else:
            syll += phoneme
    syll += syllables_values[-1][0]
    syllables.append(syll)

    final_syllables = []
    front = ''
    for (i, syllable) in enumerate(syllables):
        if not re.search('[aeiouy]', syllable):
            if len(final_syllables) == 0:
                front += syllable
            else:
                final_syllables = final_syllables[:-1] \
                                  + [final_syllables[-1] + syllable]
        else:
            if len(final_syllables) == 0:
                final_syllables.append(front + syllable)
            else:
                final_syllables.append(syllable)
    return final_syllables
