#install transformers 
#!pip install transformers
#install sentencepiece
#!pip install sentencepiece

import simplejson as json
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s " % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])

def detectSpecialCharacters(mystring):  
  special_characters = '"!@#$%^&*()+_=,<>/"'
  if any(c in special_characters for c in mystring):
    return True
  else:
    return False

def refine_question(question):
    j = question.replace('<pad> question: ', '')
    k = j.replace('</s>', '')
    return(k)

# Opening JSON file
f = open('Data/WDV_dataset.json')
data = json.load(f)

def questionGeneration(items, theme_label, readingLimit):
  # Iterating through the json list
  with open('Data/Questions/Airport.json', "r+") as file:
    output = json.load(file)
    for doc in data:
      if doc['theme_label'] == "Airport":
        
        propertyCQ = refine_question(get_question(doc['property_label'], doc['verbalisation_unk_replaced']))
        objectCQ = refine_question(get_question(doc['object_label'], doc['verbalisation_unk_replaced']))
        if detectSpecialCharacters(propertyCQ):
          propertyCQ = ""
        if detectSpecialCharacters(objectCQ):
          objectCQ = ""
        
        if "subject_id" in doc:
          subjectID = doc['subject_id']
        else:
          subjectID = ""
        if "property_id" in doc:
          propertyID = doc['property_id']
        else:
          propertyID = ""
        if "id" in doc['object']['value'] and doc['object_datatype'] == "wikibase-item":
          objectID = doc['object']['value']['id']
        else: 
          objectID = ""

        # Create JSON entry
        entry = {
            "claim_id": doc['claim_id'],
            "theme": doc['theme_label'],
            "subject_label": doc['subject_label'],
            "subject_id": subjectID,
            "property_label": doc['property_label'],
            "property_id": propertyID,
            "object_label": doc['object_label'],
            "object_id": objectID,
            "context": doc['verbalisation_unk_replaced'],
            "propertyCQ": propertyCQ,
            "objectCQ": objectCQ,
            "generalizedPropertyCQ": "",
            "generalizedObjectCQ": ""
            }
        output.append(entry)
        file.seek(0)
        json.dump(output, file)
    

  # Closing file
  f.close()
  file.close()

