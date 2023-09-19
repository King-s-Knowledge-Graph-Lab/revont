import simplejson as json
from tqdm import tqdm
from Scripts.functions import sentenceEmbedding, sentenceNER, sentenceSimilarity, runSPARQLQuery, getSPARQLResult, getSynsets, getSynsetDefinition, sentenceGrammarCheck, generalize

# Generalization based on Wikidata ID and label descriptions
def generlizationPipeline():
  print(f"(3/3) Start Generalization")
  f = open(f'Data/Temp/questionGeneration.json')
  data = json.load(f) 
  NERoutput = list()
  sep = '.'
  R_patterns = [("\'s"," is")]
  # Perform NER for all CQs and store the tuples in a list
  print("Performing NER for all CQs")
  for i in tqdm(data):
    entryList = list()
    for j in sentenceNER(i['propertyCQ']):
      entryList.append(j['word'])
    for j in sentenceNER(i['objectCQ']):
      entryList.append(j['word'])
    entry = {"NER": entryList}
    i.update(entry)

  print("Performing generalization for all CQs")
  for i in tqdm(data):
    sDescriptionEmbedding = sentenceEmbedding(i['subject_dec']) 
    # print(i['subject_description'])
    oDescriptionEmbedding = sentenceEmbedding(i['object_desc'])
    
    sLabelEmbedding = sentenceEmbedding(i['subject_label'])
    oLabelEmbedding = sentenceEmbedding(i['object_label'])
    
    for j in i['NER']:
      NERsSim = sentenceSimilarity(sentenceEmbedding(j), sLabelEmbedding)
      NERoSim = sentenceSimilarity(sentenceEmbedding(j), oLabelEmbedding) 
      
      ValDefSim = 0
      simSynset = ""

      if NERsSim > NERoSim:
        if i['subject_id'] != "":
          res_s = runSPARQLQuery(i['subject_id'])
          resValue = getSPARQLResult(res_s)
          #print("Result values are ", resValue, " for NE ", j)

          for k in resValue: 
            resValueSynset = getSynsets(k)
            
            for l in resValueSynset: 
              synsetDefinition = getSynsetDefinition(l)
              resValueEmbedding = sentenceEmbedding(synsetDefinition)
              resValueS = sentenceSimilarity(resValueEmbedding, sDescriptionEmbedding)
              if resValueS > ValDefSim:
                ValDefSim = resValueS
                simSynset = l.name()
              else:
                continue
          
        else:
          continue
          
      else: 
        if i['object_id'] != "":
          res_s = runSPARQLQuery(i['object_id'])
          resValue = getSPARQLResult(res_s)
          #print("Result values are ", resValue, " for NE ", j)

          for k in resValue: 
            resValueSynset = getSynsets(k)
            
            for l in resValueSynset: 
              synsetDefinition = getSynsetDefinition(l)
              resValueEmbedding = sentenceEmbedding(synsetDefinition)
              resValueS = sentenceSimilarity(resValueEmbedding, oDescriptionEmbedding)
              if resValueS > ValDefSim:
                ValDefSim = resValueS
                simSynset = l.name()
              else:
                continue
          
        else:
          continue
      simSynset = simSynset.split(sep, 1)[0]
      if [j, simSynset] in R_patterns:
        continue
      else:
        R_patterns.append(tuple((j, simSynset)))


  print("grammer check for all CQs")
  # Extract patterns
  with open('Data/Temp/patterns.json', 'r') as json_file:
    patterns = json.load(json_file)

  for i in tqdm(data):
    newpCQ = generalize(i['propertyCQ'],patterns)
    newoCQ = generalize(i['objectCQ'],patterns)
    genP = sentenceGrammarCheck(newpCQ)
    genO = sentenceGrammarCheck(newoCQ)
    entry = {"generalizedPropertyCQ": genP}
    entry1 = {"generalizedObjectCQ": genO}
    i.update(entry)
    i.update(entry1)
  with open('Data/Temp/GeneralizedQuestion.json', 'w') as json_file:
    json.dump(data, json_file, use_decimal=True)
  print(f"Finishhhhhh!! Results are saved in 'Data/Temp/GeneralizedQuestion.json'")
  f.close()
