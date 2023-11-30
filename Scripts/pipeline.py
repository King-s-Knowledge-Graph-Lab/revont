import simplejson as json
from tqdm import tqdm
from Scripts.functions import sentenceEmbedding, sentenceNER, sentenceSimilarity, runSPARQLQuery, getSPARQLResult, getSynsets, getSynsetDefinition, sentenceGrammarCheck, generalize
from collections import defaultdict

# Generalization based on Wikidata ID and label descriptions
from collections import defaultdict

# Generalization based on Wikidata ID and label descriptions
def generlizationPipeline(questionData, theme_label):
    print(f"(3/3) Start Generalization")
    data = questionData
    NERoutput = list()
    sep = '.'
    R_patterns = [("\'s", " is")]
    # Perform NER for all CQs and store the tuples in a list
    print("Performing NER for all CQs")

    for i in tqdm(data):
        subject_ner = sentenceNER([i['subjectCQ']])
        property_ner = sentenceNER([i['propertyCQ']])
        object_ner = sentenceNER([i['objectCQ']])
        i['NER'] = property_ner + object_ner

    print("Performing generalization for all CQs")

    embedding_cache = defaultdict(lambda: None)

    for i in tqdm(data):
        # Cache usage for main sentences
        keys_to_embed = ['subject_dec', 'object_desc', 'subject_label', 'object_label']
        sentences_to_embed = [i[key] for key in keys_to_embed if embedding_cache[i[key]] is None]

        if sentences_to_embed:
            embeddings = sentenceEmbedding(sentences_to_embed)
            for sentence, embedding in zip(sentences_to_embed, embeddings):
                embedding_cache[sentence] = embedding

        sDescriptionEmbedding = embedding_cache[i['subject_dec']]
        oDescriptionEmbedding = embedding_cache[i['object_desc']]
        sLabelEmbedding = embedding_cache[i['subject_label']]
        oLabelEmbedding = embedding_cache[i['object_label']]

        ner_sentences = [j for j in i['NER'] if embedding_cache[j] is None]
        if ner_sentences:
            ner_embeddings = sentenceEmbedding(ner_sentences)
            for sentence, embedding in zip(ner_sentences, ner_embeddings):
                embedding_cache[sentence] = embedding

        for j in i['NER']:
            NER_embedding = embedding_cache[j]
            NERsSim = sentenceSimilarity(NER_embedding, sLabelEmbedding)
            NERoSim = sentenceSimilarity(NER_embedding, oLabelEmbedding)

            ValDefSim = 0
            simSynset = ""

            if NERsSim > NERoSim:
                if i['subject_id'] != "":
                    res_s = runSPARQLQuery(i['subject_id'])
                    resValue = getSPARQLResult(res_s)
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
    with open(f'Data/Temp/patterns-{theme_label}.json', 'r') as json_file:
      patterns = json.load(json_file)

    for i in tqdm(data):
      newsCQ = generalize(i['subjectCQ'],patterns)
      newpCQ = generalize(i['propertyCQ'],patterns)
      newoCQ = generalize(i['objectCQ'],patterns)
      
      genS = sentenceGrammarCheck(newsCQ)
      genP = sentenceGrammarCheck(newpCQ)
      genO = sentenceGrammarCheck(newoCQ)
      
      entry = {"generalizedSubjectCQ": genS}
      entry1 = {"generalizedPropertyCQ": genP}
      entry2 = {"generalizedObjectCQ": genO}
      i.update(entry)
      i.update(entry1)
      i.update(entry2)
    with open(f'Data/Temp/generalizedQuestion-{theme_label}.json', 'w') as json_file:
      json.dump(data, json_file, use_decimal=True)
    print(f"Finishhhhhh!! Results are saved in 'Data/Temp/generalizedQuestion.json'")

