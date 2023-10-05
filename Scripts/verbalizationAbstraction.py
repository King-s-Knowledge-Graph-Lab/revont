# Generalize before question generation.
from Scripts.functions import sentenceEmbedding, runSPARQLQuery, getSPARQLResult, getSynsets, getSynsetDefinition, sentenceSimilarity, generalize, sentenceGrammarCheck
import simplejson as json
from tqdm import tqdm


def VerbalizationAbstaction(data, theme_label):
  sep = '.'
  patterns = dict()
  print(f"(1/3) Start verbalization abstraction for theme: {theme_label}")
  # Considering the size of the dataset, we select a theme to reduce the processing time.
  for i in tqdm(data):

    if i['theme_label'] == theme_label:
      # Create the sentence embeddings of the description of the subject or object if provided
      #if i['subject_dec'] != 'no-desc':
      sDescriptionEmbedding = sentenceEmbedding(i['subject_dec']) 
      #if i['object_desc'] != 'no-desc':
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
            synsetDefinitions = [getSynsetDefinition(l) for l in resValueSynset]
            # embedding batch processing and similarity calculation
            resValueEmbeddings = sentenceEmbedding(synsetDefinitions)
            for index, resValueEmbedding in enumerate(resValueEmbeddings):
                resValueS = sentenceSimilarity(resValueEmbedding, sDescriptionEmbedding)    
                if resValueS > ValDefSim:
                    ValDefSim = resValueS
                    simSynsetSubject = resValueSynset[index].name()
          else:
            simSynsetSubject = resValue[0]
      else:
        pass

      # Get the most similar synset name of the object if there exists an ID and there are synsets, otherwise asign the first superclass as the generalization   
      ValDefSim = 0
      if objectID != "":
        # Get superclasses
        res_s = runSPARQLQuery(objectID)
        resValue = getSPARQLResult(res_s)
        # Get synsets
        for k in resValue: 
          resValueSynset = getSynsets(k)
          if len(resValueSynset) != 0:
            for l in resValueSynset: 
              synsetDefinitions = [getSynsetDefinition(l) for l in resValueSynset]
              # embedding batch processing and similarity calculation
              resValueEmbeddings = sentenceEmbedding(synsetDefinitions)
              if resValueS > ValDefSim:
                ValDefSim = resValueS
                simSynsetObject = l.name()
              else:
                pass   
          else:
            simSynsetObject = resValue[0]
      else:
        simSynsetObject = i['object_datatype']
      simSynsetSubject = simSynsetSubject.split(sep, 1)[0]
      patternItem = {i['subject_label']: simSynsetSubject}
      #print(f"extracted 'subject_label': {patternItem}")
      if i['subject_label'] in patterns:
        pass
      else:
        patterns.update(patternItem)
        entry = {"subject_abstraction": simSynsetSubject}
        i.update(entry)

      simSynsetObject = simSynsetObject.split(sep, 1)[0]
      patternItem = {i['object_label']: simSynsetObject}
      #print(f"extracted 'object_label': {patternItem}")
      if i['object_label'] in patterns:
        pass
      else:
        patterns.update(patternItem)
        entry = {"object_abstraction": simSynsetObject}
        i.update(entry)




  # Print pattern for check reasons
  print(f"print extracted patterns: {patterns}")
  print("Start context generation")
  for i in tqdm(data):
    if i['theme_label'] == theme_label:
      if 'verbalisation_unk_replaced' in i:

        genContext = generalize(i['verbalisation_unk_replaced'], patterns)
        genContext = sentenceGrammarCheck(genContext)
        #addition new generalizationText 
        entry = {"generalizedContext": genContext}
        i.update(entry)


  # Update file
  with open(f'Data/Temp/verbalizationAbstraction-{data[0]["theme_label"]}.json', 'w') as json_file:
    json.dump(data, json_file, use_decimal=True)

  with open(f'Data/Temp/patterns-{data[0]["theme_label"]}.json', 'w') as json_file:
    json.dump(patterns, json_file, use_decimal=True)
