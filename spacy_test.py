import spacy
# import textacy.extract

# Загрузка английской NLP-модели
nlp = spacy.load('en_core_web_lg')

# Текст для анализа
text = """London is [.. shortened for space ..]"""

# Анализ
doc = nlp(text)

# Извлечение фрагментов
noun_chunks = textacy.extract.noun_chunks(doc, min_freq=3)

# Перевод в нижний регистр
noun_chunks = map(str, noun_chunks)
noun_chunks = map(str.lower, noun_chunks)

# вывод всех фрагментов, состоящих из 2 слов и более
for noun_chunk in set(noun_chunks):
    if len(noun_chunk.split(" ")) > 1:
        print(noun_chunk)


