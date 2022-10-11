# Generalize before question generation.

# Load dataset
file = open('/content/drive/MyDrive/KCL experiment/WDV_JSON.json')
data = json.load(file) 

sep = '.'
patterns = dict()

# Considering the size of the dataset, we select a theme to reduce the processing time.
for i in data:
  if i['theme_label'] == "ComicsCharacter":
    # Create the sentence embeddings of the description of the subject or object if provided
    if i['subject_dec'] != 'no-desc':
      sDescriptionEmbedding = sentenceEmbedding(i['subject_dec']) 
    if i['object_desc'] != 'no-desc':
      oDescriptionEmbedding = sentenceEmbedding(i['object_desc'])
   
    # Define variables that will store the most similar synset to the definition of the 
    ValDefSim = 0
    simSynsetSubject = ""
    simSynsetObject = ""

    # Get the wikidata ID of the subject and object
    if "subject_id" in i:
      subjectID = i['subject_id']
    else:
      subjectID = ""
    if "id" in i['object']['value'] and i['object_datatype'] == "wikibase-item":
      objectID = i['object']['value']['id']
    else: 
      objectID = ""
    
    # Remove unwanted characters from labels of subject and object
    if '.' in i['subject_label']:
      sLabel = i['subject_label']
      sLabel == sLabel.replace('.', '')
      i['subject_label'] = sLabel

    if '.' in i['object_label']:
      sLabel = i['object_label']
      sLabel == sLabel.replace('.', '')
      i['object_label'] = sLabel
    
    # Get the most similar synset name of the subject if there exists an ID and there are synsets, otherwise asign the first superclass as the generalization
    if i['subject_id'] != "":
      # Get superclasses
      res_s = runSPARQLQuery(i['subject_id'])
      resValue = getSPARQLResult(res_s)
      # Get synsets
      for k in resValue: 
        resValueSynset = getSynsets(k)
        if len(resValueSynset) != 0:
          for l in resValueSynset: 
            synsetDefinition = getSynsetDefinition(l)
            resValueEmbedding = sentenceEmbedding(synsetDefinition)
            resValueS = sentenceSimilarity(resValueEmbedding, sDescriptionEmbedding)
            if resValueS > ValDefSim:
              ValDefSim = resValueS
              simSynsetSubject = l.name()
            else:
              continue   
        else:
          simSynsetSubject = resValue[0]
    else:
      continue
    
    # Get the most similar synset name of the object if there exists an ID and there are synsets, otherwise asign the first superclass as the generalization   
    if objectID != "":
      # Get superclasses
      res_s = runSPARQLQuery(objectID)
      resValue = getSPARQLResult(res_s)
      # Get synsets
      for k in resValue: 
        resValueSynset = getSynsets(k)
        if len(resValueSynset) != 0:
          for l in resValueSynset: 
            synsetDefinition = getSynsetDefinition(l)
            resValueEmbedding = sentenceEmbedding(synsetDefinition)
            resValueS = sentenceSimilarity(resValueEmbedding, oDescriptionEmbedding)
            if resValueS > ValDefSim:
              ValDefSim = resValueS
              simSynsetObject = l.name()
            else:
              continue   
        else:
          simSynsetObject = resValue[0]
    else:
      simSynsetObject = i['object_datatype']

    simSynsetSubject = simSynsetSubject.split(sep, 1)[0]
    patternItem = {i['subject_label']: simSynsetSubject}
    print(patternItem)
    if i['subject_label'] in patterns:
      continue
    else:
      patterns.update(patternItem)
      entry = {"subject_abstraction": simSynsetSubject}
      i.update(entry)

    simSynsetObject = simSynsetObject.split(sep, 1)[0]
    patternItem = {i['object_label']: simSynsetObject}
    print(patternItem)
    if i['object_label'] in patterns:
      continue
    else:
      patterns.update(patternItem)
      entry = {"object_abstraction": simSynsetObject}
      i.update(entry)
    time.sleep(5)

# Print pattern for check reasons
print(patterns)

for i in data:
  if i['theme_label'] == "ComicsCharacter":
    if 'verbalisation_unk_replaced' in i:
      genContext = generalize(i['verbalisation_unk_replaced'], patterns)
      genContext = sentenceGrammarCheck(genContext)
      entry = {"generalizedContext": genContext}
      i.update(entry)
      print(genContext)

# Update file
file = open('/content/drive/MyDrive/KCL experiment/WDV_JSON_Try.json', "w")
json.dump(data, file)

# Close file
file.close()
