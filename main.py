from Scripts.verbalizationAbstraction import VerbalizationAbstaction as VA
from Scripts.questionGeneration import questionGeneration as QG
from Scripts.pipeline import generlizationPipeline as GP
from Scripts.questionMapping import questionMapping as QM
from Scripts.questionReduction import CQClustering, ParaphraseDetection
from sentence_transformers import SentenceTransformer
import ijson
import json
import os
import pandas as pd
def readingJson(Path, theme_label):
    with open(Path, 'r') as f:
        parser = ijson.items(f, 'item')
        # Extract a set of items that have a specific value for the 'theme_label' key
        theme_label = theme_label
        items = []
        for item in parser:
            if item.get('theme_label') == theme_label:
                if item not in items:
                    items.append(item)
    return items

def simpleInterface():
    theme = {
    '1': 'Airport',
    '2': 'Artist',
    '3': 'Astronaut',
    '4': 'Athlete',
    '5': 'Building',
    '6': 'CelestialBody',
    '7': 'ChemicalCompound',
    '8': 'City',
    '9': 'ComicsCharacter',
    '10': 'Food',
    '11': 'MeanOfTransportation',
    '12': 'Monument',
    '13': 'Mountain',
    '14': 'Painting',
    '15': 'Politician',
    '16': 'SportsTeam',
    '17': 'Street',
    '18': 'Taxon',
    '19': 'University',
    '20': 'WrittenWork'
    }
    print('Please select a theme_label from the list below:')
    for key, value in theme.items():
        print(key, value)

    while True:
        theme_input = input()
        if theme_input in theme.keys() or theme_input == '0':
            break
        else:
            print('Please select a theme_label from the list')

    if theme_input == '0':
        theme_labels = list(theme.values())[:-1]  # 'All' 제외
    else:
        theme_labels = [theme[theme_input]]

    print('Please input a reading limit. If you want to test all data, then type "0"')
    while True:
        readingLimit_input = input()
        if readingLimit_input >= '0':
            break
        else:
            print('Please input integer from 0')

    return theme_labels, int(readingLimit_input)

def listingQuestions(inputdata_path):
    with open(inputdata_path, 'r') as file:
        data = json.load(file)

    # Open the output text file for writing
    with open('Data/Temp/questions.txt', 'w') as file:
        # Iterate over each item in the JSON data
        for item in data:
            # Extract the questions and write them to the file
            questions = [
                item.get('propertyCQ'),
                item.get('objectCQ'),
                item.get('generalizedPropertyCQ'),
                item.get('generalizedObjectCQ')
            ]
            # Write each question to the file on a new line without double quotations
            for question in questions:
                if question:  # check if question is not None
                    file.write(question + '\n')

if __name__ == "__main__":
    directory_path = 'Data/Temp'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    #raw data loading with a specific theme_label
    theme_labels, readingLimit = simpleInterface()
    for theme_label in theme_labels:
        rawData = readingJson('Data/WDV_dataset.json', theme_label)
        #### PROCESS: VA -> QG -> GP ####
        #1. Run verbalizationAbstraction.py with parsed data with a specific theme_label
        if readingLimit == 0:
            Prunned_rawData = rawData
        else:
            Prunned_rawData =  rawData[:readingLimit]
        VA(Prunned_rawData, theme_label) #save the results in the JSON file "Data/Temp/VerbalizationAbstraction.json"

        #2. Run questionGeneration.py  with parsed data with a specific theme_label
        verbalData = readingJson(f'Data/Temp/verbalizationAbstraction-{theme_label}.json', theme_label)
        QG(verbalData, theme_label) #save the results in the JSON file "Data/Temp/questionGeneration.json"

        #3. Run pipeline.py with parsed data with a specific theme_label
        questionData = readingJson(f'Data/Temp/questionGeneration-{theme_label}.json', theme_label)
        GP(questionData, theme_label) #generalization for above QG

        #4. Run questinMapping.py with BigCQ dataset
        generalizedQuestions = readingJson(f"Data/Temp/generalizedQuestion-{theme_label}.json", theme_label)
        QM(questionData, theme_label) #generalization for above QG

        #5. Run questionReduction.py with generated questions
        #Candidate model list
        listingQuestions(f"Data/Temp/generalizedQuestion-{theme_label}.json")
        model_st1 = SentenceTransformer('all-mpnet-base-v2')
        model_st2 = SentenceTransformer('all-MiniLM-L6-v2')
        model_st3 = SentenceTransformer('paraphrase-mpnet-base-v2')
        model_st4 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        with open('Data/Temp/questions.txt') as f:
            questions = f.readlines()
        clustering_results = CQClustering(questions, model_st2)
        print(clustering_results)
        qpp_data = pd.read_csv("Data/qqp.tsv", sep="\t")
        reading_limits = 5000 #preventing exhausting reading and processing 
        qpp_data = qpp_data[:reading_limits]
        ParaphraseDetection(qpp_data, model_st2)

