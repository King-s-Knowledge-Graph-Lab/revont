# Generalization based on Wikidata ID and label descriptions

# Opening JSON file
f = open('/content/drive/MyDrive/KCL experiment/Themed questions/Athlete/Athlete.json')
data = json.load(f) 

NERoutput = list()
sep = '.'
R_patterns = [('\'s',' is')]

# Perform NER for all CQs and store the tuples in a list
for i in data:
  entryList = list()
  for j in sentenceNER(i['propertyCQ']):
    entryList.append(j['word'])
  for j in sentenceNER(i['objectCQ']):
    entryList.append(j['word'])
  entry = {"NER": entryList}
  i.update(entry)
  print(i)

for i in data:
  sDescriptionEmbedding = sentenceEmbedding(i['subject_description']) 
  # print(i['subject_description'])
  oDescriptionEmbedding = sentenceEmbedding(i['object_description'])
  
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

print(R_patterns)

generalizer = REReplacer()
for i in data:
  newpCQ = generalizer.replace(i['propertyCQ'])
  newoCQ = generalizer.replace(i['objectCQ'])
  genP = sentenceGrammarCheck(newpCQ)
  genO = sentenceGrammarCheck(newoCQ)
  entry = {"generalizedPropertyCQ": genP}
  entry1 = {"generalizedObjectCQ": genO}
  i.update(entry)
  i.update(entry1)

f.close()
