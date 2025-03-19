from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()


palavras = ['running', 'better', 'studies', 'wolves', 'mice', 'children', 'was', 
            'ate', 'swimming', 'parties', 'leaves', 'knives', 'happier', 'studying', 
            'played', 'goes', 'driving', 'talked']

lemmas = [lemmatizer.lemmatize(palavra, pos='v') if palavra in ['running', 'swimming', 'played', 'talked', 'driving', 'goes', 'ate'] else lemmatizer.lemmatize(palavra) for palavra in palavras]


print(lemmas)
