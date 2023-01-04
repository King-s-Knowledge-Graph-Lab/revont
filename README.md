<div align="center">

 <!-- <img src="Materials/Ciroku (2).png" alt="logo" width="200" height="auto" /> -->
  <h1>RevOnt: Reverse engineering of an ontology via competency question extraction from knowledge graphs</h1>
  
  <p>
    Extracting competency questions from the Wikidata knowledge graph
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/FiorelaCiroku/Ontology-Reverse-Engineering/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Louis3797/awesome-readme-template" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/FiorelaCiroku/Ontology-Reverse-Engineering" alt="last update" />
  </a>
  <a href="https://github.com/FiorelaCiroku/Ontology-Reverse-Engineering/network/members">
    <img src="https://img.shields.io/github/forks/FiorelaCiroku/Ontology-Reverse-Engineering" alt="forks" />
  </a>
  <a href="https://github.com/FiorelaCiroku/Ontology-Reverse-Engineering/stargazers">
    <img src="https://img.shields.io/github/stars/FiorelaCiroku/Ontology-Reverse-Engineering" alt="stars" />
  </a>
  <a href="https://github.com/FiorelaCiroku/Ontology-Reverse-Engineering/issues/">
    <img src="https://img.shields.io/github/issues/FiorelaCiroku/Ontology-Reverse-Engineering" alt="open issues" />
  </a>
  <!--<a href="https://github.com/FiorelaCiroku/Ontology-Reverse-Engineering/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Louis3797/awesome-readme-template.svg" alt="license" /> 
  </a> -->
</p>
   
<h4>
    <a href="https://github.com/FiorelaCiroku/RevOnt/tree/main/Scripts">Scripts</a>
  <span> · </span>
    <a href="https://github.com/FiorelaCiroku/RevOnt/blob/main/README.md">Documentation</a>
  <span> · </span>
    <a href="https://github.com/FiorelaCiroku/RevOnt/issues">Report Issue</a>
  </h4>
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)

- [Scripts](#toolbox-scripts)

  * [Prerequisites](#bangbang-prerequisites)
  * [Import](#gear-import)

- [Usage](#eyes-usage)
- [Roadmap](#compass-roadmap)
- [Contributing](#wave-contributing)
   <!--- * [Code of Conduct](#scroll-code-of-conduct)
- [FAQ](#grey_question-faq)
- [License](#warning-license)-->
- [Contact](#handshake-contact)
- [Acknowledgements](#gem-acknowledgements)

  

<!-- About the Project -->
## :star2: About the Project

The process of developing ontologies - a formal, explicit specification of a shared conceptualisation - is addressed by well-known methodologies. 
As for any engineering development, its fundamental basis is the collection of requirements, which includes the elicitation of competency questions. Competency questions are defined through interacting with domain and application experts or by investigating existing datasets that may be used to populate the ontology i.e. its knowledge graph. The rise in popularity and accessibility of knowledge graphs provides an opportunity to support this phase with automatic tools. In this work, we explore the possibility of extracting competency questions from a knowledge graph. We describe in detail RevOnt, an approach that extracts and abstracts triples from a knowledge graph, generates questions based on triple verbalisations, and filters the questions to guarantee that they are competency questions. This approach is implemented utilizing the Wikidata knowledge graph as a use case. The implementation results in a set of core competency questions from 20 domains present in the dataset presenting the knowledge graph, and their respective templates mapped to SPARQL query templates. We evaluate the resulting competency questions by calculating the BLEU score using human-annotated references. The results for the abstraction and question generation components of the approach show good to high quality. Meanwhile, the accuracy of the filtration component is above 86\%, which is comparable to the state-of-the-art classifications. 

![REVONT (3)](https://user-images.githubusercontent.com/12375920/210616161-9105a046-c809-4182-beb6-5ef4556ec101.png)

An overview of the RevOnt framework. The first stage, Verbalisation Abstraction, generates the abstraction of a triple verbalisation. The abstraction is used as input in the second stage, Question Generation, to generate three questions per triple and perform a grammar check. Lastly, the third stage, Question Filtration, filters the questions by performing different techniques.

<!-- Scripts -->
## 	:toolbox: Scripts

<!-- Prerequisites -->
### :bangbang: Prerequisites

This project needs to have installed several packages for the usage of the language models and the Wikidata querying service.

```
pip install -U sentence-transformers
pip install happytransformer
pip3 install qwikidata
```

<!-- Import -->
### :gear: Import

The functions import many packages and need to the downloading of wordnet and OMW-1.4 as shown below. 

```
from qwikidata.sparql import return_sparql_query_results
from IPython.core.debugger import skip_doctest
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
from happytransformer import HappyTextToText, TTSettings
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import time
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')
```

<!-- Usage -->
## :eyes: Usage

In the repository, there are separate scripts for each of the components. This separation provides the possibility to opt out a using a component or interchanging the queue in which the components are executed. The scripts also allow to use a different language model that the default one. The language models used in the scripts are state-of-the-art models that have shown good to high results in the first evaluation of the method. 

<!-- Roadmap -->
## :compass: Roadmap

* [x] First implementation of the RevOnt method using data from the Wikidata knowledge graph
* [ ] Second implementation of the RevOnt method using data from a AMR graph built from a textual corpus. 


<!-- Contributing -->
## :wave: Contributing

<a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Louis3797/awesome-readme-template" />
</a>
Contributions are always welcome!


<!-- Code of Conduct 
### :scroll: Code of Conduct

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)

<!-- FAQ 
## :grey_question: FAQ

- Question 1

  + Answer 1

- Question 2

  + Answer 2
-->

<!-- License 
## :warning: License

Distributed under the no License. See LICENSE.txt for more information.
-->

<!-- Contact -->
## :handshake: Contact

Fiorela Ciroku - [@ciroku_fiorela](https://twitter.com/ciroku_fiorela) - fiorela.ciroku2@unibo.it

Project Link: (https://github.com/FiorelaCiroku/RevOnt)[https://github.com/FiorelaCiroku/RevOnt]


<!-- Acknowledgments -->
## :gem: Acknowledgements

 - [Jacopo de Berardinis](https://www.kcl.ac.uk/people/jacopo-de-berardinis)
 - [Albert Merono Penuela](https://www.kcl.ac.uk/people/albert-merono-penuela-1)
 - [Valentina Presutti](https://www.unibo.it/sitoweb/valentina.presutti/en)
 - [Elena Simperl](https://www.kcl.ac.uk/people/elena-simperl)
