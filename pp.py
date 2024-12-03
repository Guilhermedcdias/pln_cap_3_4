from gensim.models import KeyedVectors
# LINK PARA DOWNLOAD: https://fasttext.cc/docs/en/crawl-vectors.html
# Caminho para o arquivo cc.pt.300.vec
model_path = "./cc.pt.300.vec"

# Carregar o modelo
model = KeyedVectors.load_word2vec_format(model_path, binary=False)

# Verificar a similaridade entre palavras
print(model.most_similar("produto"))


# Exemplo de frase
input_sentence = "Ele vai terminar o curso"


def replace_with_synonyms(sentence, model):
    from nltk.tokenize import word_tokenize

    words = word_tokenize(sentence.lower())
    new_sentence = []

    for word in words:
        if word in model.key_to_index:
            # Obter o sinônimo mais próximo
            synonym = model.most_similar(word, topn=1)[0][0]
            new_sentence.append(synonym)
        else:
            new_sentence.append(word)

    return " ".join(new_sentence)


output_sentence = replace_with_synonyms(input_sentence, model)
print("Frase original:", input_sentence)
print("Frase com sinônimos:", output_sentence)
