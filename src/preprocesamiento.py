def tokenizar(texto):
    return texto.split()

def construir_vocabulario(tokens):
    vocab = sorted(set(tokens))
    word_to_ix = {w: i for i, w in enumerate(vocab)}
    ix_to_word = {i: w for w, i in word_to_ix.items()}
    return vocab, word_to_ix, ix_to_word

def generar_pares_skipgram(tokens, window_size=2):
    pares = []
    for i, target in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if j != i:
                pares.append((target, tokens[j]))
    return pares
