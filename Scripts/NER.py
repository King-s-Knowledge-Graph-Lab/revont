# Install transformers 
!pip install transformers
# Install sentencepiece
!pip install sentencepiece

# Import JSON
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Opening  file
data = open('Data/Questions.json')
CQ = json.load(data)

with open('Data/QuestionsNER.json', "r+") as file:
    output = json.load(file)
    for i in data:
      NERp = nlp(i['PAnswer']['propertyCQ'])
      NERo = nlp(i['OAnswer']['objectCQ'])
      for j in NERp:
        j['score'] = str(j['score'])
        output.append(j)
      for j in NERo:
        j['score'] = str(j['score'])
        output.append(j)
        file.seek(0)
        json.dump(output, file)

# Closing file
file.close()
