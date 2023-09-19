from Scripts.verbalizationAbstraction import VerbalizationAbstaction as VA
from Scripts.questionGeneration import VerbalizationAbstaction as VA
import ijson
import simplejson as json

if __name__ == "__main__":
    theme_label = 'Building'
    readingLimit = 1 # Number of items to read from the JSON file to avoid long processing times, 'all' = no limit
    
    # Open the JSON file and create a parser object for the specific theme_label
    with open('Data/WDV_dataset.json', 'r') as f:
        parser = ijson.items(f, 'item')
        # Extract a set of items that have a specific value for the 'theme_label' key
        theme_label = theme_label
        items = []
        for item in parser:
            if item.get('theme_label') == theme_label:
                if item not in items:
                    items.append(item)
    
    #Run verbalizationAbstraction.py with parsed data with a specific theme_label
    VA(items, theme_label, readingLimit) #save the results in the JSON file "Data/Temp/VerbalizationAbstraction.json"

    #Run questionGeneration.py