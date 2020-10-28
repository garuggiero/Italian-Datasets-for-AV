# Italian Datasets for Authorship Verification

This repository contains the instructions to use and format two Italian datasets specifically designed for Authorship Verification. 

The datasets were collected for my Master Thesis ''Datasets and Models for Authorship Attribution on Italian Personal Writings'', written in collaboration with Professor Malvina Nissim from the University of Groningen, and Professor Albert Gatt, from the University of Malta. This work is being published at the Seventh Italian Conference on Computational Linguistics, CliC-it 2020. 

The datasets were used to run AV experiments using GLAD [1]. We investigated four dimensions: topic, length and genre of the texts and gender of the authors. Since the Diaries dataset does not contain any topic information, topic categories were only taken into account for the ForumFree dataset. 

## ForumFree  

The ForumFree dataset is a subset of a bigger dataset compiled by Maslennikova et al.[2]. The original dataset was meant for Age Identification experiments. 
Here, we reformat it according to the PAN 2015 format for AV [3]. The original dataset was a courtesy of the [Italian Institute of Computational Linguistics “Antonio Zampolli” (ILC) of Pisa](http://www.ilc.cnr.it/). The reformatted dataset is contained in `AV_ForumFree.zip`.

The ForumFree dataset contains forum comments taken from the [ForumFree](https://www.forumfree.it/) platform. 
It covers two topics, *Medicina Estetica* (Aesthetic Medicine) and *Programmi Tv* (Tv-programmes). A third topic, *Mixedtopics*, is the union of Medicina Estetica and Programmi Tv.

## Diaries 

The Diaries dataset is a collection of diary fragments included in the project [Italiani all’estero: i diari raccontano](https://www.idiariraccontano.org). These fragments are the diaries, letters and memoirs of Italian people who lived abroad between the beginning of the 19th century and the present day. 

The data from their website was used by permission of the data creators, and was automatically collected using software written for the purpose. We are happy to provide access to the scraping software, but permission from the data creators needs to be sought first. However, we provide here the code to reformat the data into AV problems, i.e. `make_Diaries_dataset.py`.
 

## Structure

The goal of the Authorship Verification task is to determine whether or not two documents are written by the same author. Therefore, the data is structured in problems (instances) made of two texts (*known* and *unknown* text) of equal length. 

ForumFree and Diaries are structured in the same way.

The folder names contain information about the type of data contained in them:

- *blogs* if they contain forum posts,  *diaries* if they contain diary fragments

- *train* or *test*, according to whether that subset is destined to be a training or a test set: training sets contain 70% of the data, test sets contain the remaining 30%

- *medicina10*, *tvprogrammes* or *mixedtopic* stands for the topic of the data contained, available only for the ForumFree dataset

- *400*, *1000*, *2000* or *3000* according to the number of words per problem

- *mixed*, *females* or *males* refers to the gender of the authors of known and unknown text: if *mixed*, the authors can be male and females, if *females* both the authors are female, if *males* both the authors are male

- *_downsized*: if added to the folder name, it means that the folders containing data from the same genre, topic and within the same authors' gender subset, correspond to data from the same pool of authors. For example, blogs_test_medicina10_400_mixed_downsized, blogs_test_medicina10_1000_mixed_downsized, blogs_test_medicina10_2000_mixed_downsized contain texts from the same authors as in blogs_test_medicina10_3000_mixed. The only factor changing is thus the number of words contained in the single documents.

Each folder contains a certain number of AV problems, labelled with a problem ID, which goes from IT001 to IT0...n. Each problem contains two .txt files, *known01.txt* and *unknown.txt*, representing the text pair. Rather than representing a single text, each of these documents contains a set of texts written by the same author.

Since we experimented with texts of different length, the known and unknown documents can be made of 200, 500, 1 000 and 1 500 words each. The text is already tokenized. 

Moreover, each folder contains 3 files:
- trutx.txt, in which we store the gold labels (Y o N) associated to each problem ID
         ```
         IT001 N
         IT002 Y
         IT003 N
         ...
         ```
         
- INFO.csv links the problem ID to the text pair associated with it, the gender of the authors and the gold label
          ```
          IT001	("@ Naturalia ... hai poi fatto la versione con il detergente bio ? ...")	Female	Female	N
          ```
- contents.json, a json file containing the language of the texts and a list of problem IDs

### References 

<a id="1">[1]</a>
Hürlimann, M., Weck, B., van den Berg, E., Suster, S., & Nissim, M. (2015). 
GLAD: Groningen Lightweight Authorship Detection. 
In CLEF (Working Notes).

<a id="2">[2]</a>
Maslennikova, A., Labruna, P., Cimino, A., & Dell'Orletta, F. (2019). 
Quanti anni hai? Age Identification for Italian. 
In CLiC-it.

<a id="3">[3]</a>
Stamatatos, E., Daelemans, W., Verhoeven, B., Juola, P., López, A., Potthast, M., & Stein, B. (2015). 
Overview of the author identification task at pan 2015. CLEF 2015 Evaluation Labs and Workshop, Online Working Notes, Toulouse, France. 
In CEUR Workshop Proceedings (pp. 1-17).




