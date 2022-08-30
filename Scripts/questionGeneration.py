# Install transformers 
!pip install transformers
# Install sentencepiece
!pip install sentencepiece

from transformers import AutoModelWithLMHead, AutoTokenizer

# Import JSON
import JSON

# T5 model for question generation
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# Define function for generating questions
def get_question(answer, context, max_length=64):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)

  return tokenizer.decode(output[0])
  
# Opening and load JSON file
f = open('/content/drive/MyDrive/KCL experiment/WDV_JSON.json')
data = json.load(f)

# Iterate through the dataset
with open('/content/drive/MyDrive/KCL experiment/Themed questions/Street.json', "r+") as file:
  output = json.load(file)
  for doc in data:
    if doc['theme_label'] == 'Street':
      subjectCQ = get_question(doc['subject_label'], doc['verbalisation_unk_replaced'])
      propertyCQ = get_question(doc['property_label'], doc['verbalisation_unk_replaced'])
      objectCQ = get_question(doc['object_label'], doc['verbalisation_unk_replaced'])
  
      # Create JSON entry
      entry = {"claim_id": doc['claim_id'],
           "SAnswer":{
           "subject_label": doc['subject_label'],
           "context": doc['verbalisation_unk_replaced'],
           "subjectCQ": subjectCQ
           },
           "PAnswer":{
           "property_label": doc['property_label'],
           "context": doc['verbalisation_unk_replaced'],
           "propertyCQ": propertyCQ 
           },
           "OAnswer": {
           "object_label": doc['object_label'],
           "context": doc['verbalisation_unk_replaced'],
           "objectCQ": objectCQ
           }
          }

    # Append the entry in the file
      output.append(entry)
      file.seek(0)
      json.dump(output, file)

# Closing file
f.close()
output.close()
