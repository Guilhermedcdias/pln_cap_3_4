from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Corpus treinado baseado nos exercícios
corpus = [
    "eu gostei muito do produto",
    "o produto parece ruim",
    "este produto é excelente",
    "eu não gostei do serviço",
    "este serviço é bom",
    "o atendimento foi rápido e eficiente",
    "a qualidade do produto é excepcional",
    "não recomendo este serviço, foi péssimo",
    "tive uma experiência incrível com este produto",
    "o suporte técnico demorou muito para responder",
    "o design do produto é inovador e atraente",
    "o preço é alto, mas a qualidade compensa",
    "não vale o preço pago, fiquei insatisfeito",
    "o produto veio com defeito e não funciona",
    "recomendo este produto, superou minhas expectativas",
    "o serviço de entrega foi muito rápido",
    "a embalagem estava danificada, mas o produto estava intacto",
    "o produto atende bem às necessidades do dia a dia",
    "o serviço de atendimento ao cliente foi muito atencioso",
    "o desempenho do produto é melhor do que o esperado",
    "tive problemas com a instalação, mas o suporte ajudou rapidamente",
    "o manual de instruções não é claro, dificulta o uso",
    "fiquei surpreso com a durabilidade do produto",
    "a experiência de compra foi simples e prática",
    "o serviço online funciona bem, mas poderia ser mais intuitivo"
]

# Tokenização
tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

# Treinamento do modelo Word2Vec com o novo corpus
model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1)


# Substituir palavras por sinônimos
def replace_with_synonyms(sentence):
    words = word_tokenize(sentence.lower())
    new_sentence = []
    for word in words:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=1)[0][0]
            new_sentence.append(similar)
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

# Frase de teste
input_sentence = "Ele vai terminar o curso"
print("Frase original:", input_sentence)
print("Frase modificada:", replace_with_synonyms(input_sentence))






















































# 1. Critério de Similaridade de Cosseno
# A similaridade de cosseno mede o quão similar é a direção de dois vetores em um espaço vetorial. Ela é definida como:

# Valor próximo de 1: Indica que os vetores têm uma direção similar.
# Valor próximo de 0: Indica que os vetores são ortogonais.
# Valor negativo: Vetores apontam em direções opostas.
# A função most_similar do Gensim retorna palavras com base na maior similaridade de cosseno (menor ângulo).