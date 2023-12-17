<div align="center">

 <!-- <img src="Materials/Ciroku (2).png" alt="logo" width="200" height="auto" /> -->
  <h1>RevOnt: Reverse engineering of competency question from knowledge graphs</h1>
  
  <p>
    Extracting competency questions from the Wikidata knowledge graph
  </p>
  
## About the Project

The process of developing ontologies - a formal, explicit specification of a shared conceptualisation - is addressed by well-known methodologies. 
As for any engineering development, its fundamental basis is the collection of requirements, which includes the elicitation of competency questions. Competency questions are defined through interacting with domain and application experts or by investigating existing datasets that may be used to populate the ontology i.e. its knowledge graph. The rise in popularity and accessibility of knowledge graphs provides an opportunity to support this phase with automatic tools. In this work, we explore the possibility of extracting competency questions from a knowledge graph. We describe in detail RevOnt, an approach that extracts and abstracts triples from a knowledge graph, generates questions based on triple verbalisations, and filters the questions to guarantee that they are competency questions. This approach is implemented utilizing the Wikidata knowledge graph as a use case. The implementation results in a set of core competency questions from 20 domains present in the dataset presenting the knowledge graph, and their respective templates mapped to SPARQL query templates. We evaluate the resulting competency questions by calculating the BLEU score using human-annotated references. The results for the abstraction and question generation components of the approach show good to high quality. Meanwhile, the accuracy of the filtration component is above 86\%, which is comparable to the state-of-the-art classifications. 

![REVONT (3)](https://user-images.githubusercontent.com/12375920/210616161-9105a046-c809-4182-beb6-5ef4556ec101.png)

An overview of the RevOnt framework. The first stage, Verbalisation Abstraction, generates the abstraction of a triple verbalisation. The abstraction is used as input in the second stage, Question Generation, to generate three questions per triple and perform a grammar check. Lastly, the third stage, Question Filtration, filters the questions by performing different techniques.


<!-- Prerequisites -->
### Prerequisites

This project needs to have installed several packages for the usage of the language models and the Wikidata querying service.

```
pip install -U sentence-transformers
pip install -r requirements.txt
```
<!-- Usage -->
## Usage

In the repository, there are separate scripts (a floder "Script") for each of the components. This separation provides the possibility to opt out a using a component or interchanging the queue in which the components are executed. The scripts also allow to use a different language model that the default one. The language models used in the scripts are state-of-the-art models that have shown good to high results in the first evaluation of the method. GPU resources are recommended for encoding sentences into embedding vectors, but the code can also run using CPU resources.

You can use main.py to run the framework without executing the scripts separately.
```
python main.py
```

Alternatively, you can use the Quickstart_revont.ipynb file to run the framework on the Colab notebook.


## WDV-CQ Dataset

The **WDV-CQ** provides human-annotated and the RevOnt-generated verbalisations and competency questions from a sample of Wikidata triples (this extends the WDV collection by [Amaral et al.](https://arxiv.org/abs/2205.02627)).
Human annotations can be found in the WDV-CQ-HA subset, whereas their corresponding extractions via RevOnt are in the WDV-CQ-RO subset.
The WDV-CQ dataset can be accessed from [Zenodo](https://zenodo.org/records/10370725) and is made available under the Attribution 4.0 International CC-BY 4.0 license.

## Contact

Jongmo Kim - jongmo.kim@kcl.ac.uk <br>
Fiorela Ciroku - [@ciroku_fiorela](https://twitter.com/ciroku_fiorela) - fiorela.ciroku2@unibo.it 

## Acknowledgements

 - [Jacopo de Berardinis](https://www.kcl.ac.uk/people/jacopo-de-berardinis)
  - [Jongmo Kim](https://kr.linkedin.com/in/jongmo-kim-629995164)
 - [Albert Merono Penuela](https://www.kcl.ac.uk/people/albert-merono-penuela-1)
 - [Valentina Presutti](https://www.unibo.it/sitoweb/valentina.presutti/en)
 - [Elena Simperl](https://www.kcl.ac.uk/people/elena-simperl)

## License

RevOnt's code is released under the [MIT](https://opensource.org/license/mit/) liucense, whereas the dataset of extracted competency questions follows the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please, contact us if you have any doubt or issue concerning our licensing strategy.

---

## Citation
```
@article{RevOnt,
    title={RevOnt: Reverse Engineering of Competency Questions from Knowledge Graphs},
    author={Fiorela Ciroku, Jacopo de Berardinis, Jongmo Kim, Albert Meroño-Peñuela, and Valentina  Presutti and Elena Simperl},
    journal={Under revision}, 
    year={2023},
    keywords={knowledge engineering, knowledge graph, ontology development, competency question extraction}
}
```
