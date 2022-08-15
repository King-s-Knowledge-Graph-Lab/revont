# install transformers 
!pip install transformers
# install sentencepiece
!pip install sentencepiece

import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")

from transformers import pipeline

nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Opening  file
data = open('/content/drive/MyDrive/KCL experiment/University CQs/CQs.txt')

with open('/content/drive/MyDrive/KCL experiment/UniNER.json', "r+") as file:
    output = json.load(file)
    for i in data:
      NER = nlp(i)
      print(NER)

      #output.append(NER)
      #file.seek(0)
      #json.dump(output, file)

# Closing file
file.close()
