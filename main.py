from Scripts.verbalizationAbstraction import VerbalizationAbstaction as VA
from Scripts.questionGeneration import questionGeneration as QG
from Scripts.pipeline import generlizationPipeline as GP
from Scripts.questionMapping import questionMapping as QM
import ijson
import simplejson as json

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
        if theme_input in theme.keys():
            break
        else:
            print('Please select a theme_label from the list')
    theme_label = theme[theme_input]

    print('Please input a reading limit. If you want to test all data, then type "0"')
    while True:
        readingLimit_input = input()
        if readingLimit_input >= '0':
            break
        else:
            print('Please input integer from 0')
    return theme_label, int(readingLimit_input)



if __name__ == "__main__":
    #raw data loading with a specific theme_label
    theme_label, readingLimit = simpleInterface()
    rawData = readingJson('Data/WDV_dataset.json', theme_label)
    #### PROCESS: VA -> QG -> GP ####
    #1. Run verbalizationAbstraction.py with parsed data with a specific theme_label
    if readingLimit == 0:
        Prunned_rawData = rawData
    else:
        Prunned_rawData =  rawData[:readingLimit]
    #VA(Prunned_rawData, theme_label) #save the results in the JSON file "Data/Temp/VerbalizationAbstraction.json"

    #2. Run questionGeneration.py  with parsed data with a specific theme_label
    verbalData = readingJson(f'Data/Temp/verbalizationAbstraction-{theme_label}.json', theme_label)
    #QG(verbalData, theme_label) #save the results in the JSON file "Data/Temp/questionGeneration.json"

    #3. Run pipeline.py with parsed data with a specific theme_label
    questionData = readingJson(f'Data/Temp/questionGeneration-{theme_label}.json', theme_label)
    #GP(questionData, theme_label) #generalization for above QG

    #. Run questinMapping.py with BigCQ dataset
    generalizedQuestions = readingJson(f"Data/Temp/generalizedQuestion-{theme_label}.json", theme_label)
    QM(questionData, theme_label) #generalization for above QG