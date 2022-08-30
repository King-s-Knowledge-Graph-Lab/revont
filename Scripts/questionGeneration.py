#install transformers 
!pip install transformers
#install sentencepiece
!pip install sentencepiece

import json
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# Function to generate questions using the T5 model
def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s " % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

# Function to "refine" the generated question

def refine_question(question):
    j = question.replace('<pad> question: ', '')
    k = j.replace('</s>', '')
    return(k)

# Open and load dataset JSON file
f = open('/content/drive/MyDrive/KCL experiment/WDV_JSON.json')
data = json.load(f)

# Generate questions when providing the property or the object as answer and the verbalization as context
with open('/content/drive/MyDrive/KCL experiment/Questions.json', "r+") as file:
  output = json.load(file)
  for doc in data:
    propertyCQ = get_question(doc['property_label'], doc['verbalisation_unk_replaced'])
    objectCQ = get_question(doc['object_label'], doc['verbalisation_unk_replaced'])
  
     # Create JSON entry with new questions
    entry = {
           "claim_id": doc['claim_id'],
           "theme": doc['theme_label'],
           "PAnswer":{
           "property_label": doc['property_label'],
           "context": doc['verbalisation_unk_replaced'],
           "propertyCQ": refine_question(propertyCQ) 
           },
           "OAnswer": {
           "object_label": doc['object_label'],
           "context": doc['verbalisation_unk_replaced'],
           "objectCQ": refine_question(objectCQ)
           }
          }
    # Append the entry to the new JSON file
    output.append(entry)
    file.seek(0)
    json.dump(output, file)
  

# Closing file
f.close()
output.close()

