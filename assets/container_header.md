---
container-id: RevOnt
type: Project
name: "Revont: the RevOnt framework and validation dataset"
description: Extracting competency questions from the Wikidata knowledge graph and creating quality benchmarch datasets to support this task.
licence:
  - CC-BY_4.0
image: https://user-images.githubusercontent.com/12375920/210616161-9105a046-c809-4182-beb6-5ef4556ec101.png
logo: https://github.com/King-s-Knowledge-Graph-Lab/revont/blob/main/assets/RevOnt_logo.png
demo: https://github.com/King-s-Knowledge-Graph-Lab/revont/blob/main/Quickstart_revont.ipynb
release-date: 29-02-2024
release-number: v1.0.0
keywords:
  - RevOnt
  - WDV-CQ
work-package:
- WP2
pilot:
- Interlink
project: Polifonia Project
funder:
  - name: Polifonia Project
    url: "https://polifonia-project.eu"
    grant-agreement: "101004746"
credits: "This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement N. 101004746"
bibliography:
  - "Ciroku, Fiorela; de Berardinis, Jacopo; Kim, Jongmo; Meroño Peñuela, Albert; Presutti, Valentina; Simperl, Elena. RevOnt: Reverse Engineering of Competency Questions from Knowledge Graphs via Language Models"

has-part:
  - RevOnt software
  - WDV-CQ dataset
---

# RevOnt

The process of developing ontologies - a formal, explicit specification of a shared conceptualisation - is addressed by well-known methodologies. 
As for any engineering development, its fundamental basis is the collection of requirements, which includes the elicitation of competency questions. Competency questions are defined through interacting with domain and application experts or by investigating existing datasets that may be used to populate the ontology i.e. its knowledge graph. The rise in popularity and accessibility of knowledge graphs provides an opportunity to support this phase with automatic tools. In this work, we explore the possibility of extracting competency questions from a knowledge graph. We describe in detail RevOnt, an approach that extracts and abstracts triples from a knowledge graph, generates questions based on triple verbalisations, and filters the questions to guarantee that they are competency questions. This approach is implemented utilizing the Wikidata knowledge graph as a use case. The implementation results in a set of core competency questions from 20 domains present in the dataset presenting the knowledge graph, and their respective templates mapped to SPARQL query templates. We evaluate the resulting competency questions by calculating the BLEU score using human-annotated references. The results for the abstraction and question generation components of the approach show good to high quality. Meanwhile, the accuracy of the filtration component is above 86\%, which is comparable to the state-of-the-art classifications. 

![REVONT (3)](https://user-images.githubusercontent.com/12375920/210616161-9105a046-c809-4182-beb6-5ef4556ec101.png)

An overview of the RevOnt framework. The first stage, Verbalisation Abstraction, generates the abstraction of a triple verbalisation. The abstraction is used as input in the second stage, Question Generation, to generate three questions per triple and perform a grammar check. Lastly, the third stage, Question Filtration, filters the questions by performing different techniques.