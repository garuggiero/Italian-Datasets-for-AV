# Italian Datasets for Authorship Verification

This repository contains the instructions to use and format two Italian datasets specifically designed for the Authroship Verification Task. The datasets were used to run AV experiments using GLAD (https://www.researchgate.net/profile/Simon_Suster/publication/291346207_GLAD_Groningen_Lightweight_Authorship_Detection/links/56a1413608ae24f62701f9c5.pdf). We investigated four dimensions: topic, length and genre of the texts and gender of the authors. Since the Diaries dataset does not contain any topic information, topic categories were only taken into account for the ForumFree dataset. 

# ForumFree Dataset 

The ForumFree dataset is a subset of a bigger dataset compiled by Maslennikova et al. (2019) (http://ceur-ws.org/Vol-2481/paper43.pdf). 
The original dataset was meant for Age Identification experiments. Here, we reformat it according to the PAN 2015 format for AV (https://pan.webis.de/clef15/pan15-web/authorship-verification.html). The original dataset was a courtesy of the Italian Institute of Computational Linguistics “Antonio Zampolli” (ILC) of Pisa (http://www.ilc.cnr.it/). 

The ForumFree dataset contains forum comments taken from the ForumFree platform (https://www.forumfree.it/). 
It covers two topics, Medicina Estetica (Aesthetic Medicine) and Programmi Tv (Tv-programmes). A third topic, Mixedtopics, is simply the union of Medicina Estetica and Programmi Tv. Mixedtopics was created to run experiments where the known-unknown text pairs can be either same- or different-topic.

# Diaries Dataset

The Diaries dataset is a collection of diary fragments included in the project Italiani all’estero: i diari raccontano (Italians abroad: the diaries narrate) (https://www.idiariraccontano.org). These fragments are the diaries, letters and memoirs of Italian people who lived abroad between the beginning of the 19th century and the present day. 

The data from their website was used by permission of the data creators, and was automatically collected using software written for the purpose. We are happy to provide access to the scraping software, but permission from the data creators needs to be sought first. However, we provide here the code to reformat the data into AV problems.
 

# Structure

The goal of the Authorship Verification task is to determine whether or not two documents are written by the same author. Therefore, the data is structured in problems (instances) made of two texts (known and unknown text) of equal length. The structure of the ForumFree and Diaries datasets is exactly the same. While ForumFree is already available in the AV format, the Diaries dataset needs to be compiled. Further information about compiling this dataset is offered in the next section. 

The folder names contain information about the type of data contained in them ('blogs' or 'diaries'), the subset for which the data is destined ('train' or 'test' set), the topic (only available for ForumFree: 'medicina10', 'tvprogrammes' or 'mixedtopic'), the number of words per instance ('400', '1000', '2000', '3000') and the gender of the authors ('mixed' for text pairs that can be written by both female and male authors, 'females' for only female authors, 'males' for only male authors.
When '_downsized'_ is added to the folder name, it means that the folders containing data from the same genre, topic and within the same authors' gender subset, correspond to data from the same pool of authors. For example, blogs_test_medicina10_400_mixed_downsized, blogs_test_medicina10_1000_mixed_downsized, blogs_test_medicina10_2000_mixed_downsized contain texts from the same authors as in blogs_test_medicina10_3000_mixed. The only factor changing is thus the number of words contained in the single documents. 

Each folder contains a certain number of AV problems, labelled with a problem ID, which goes from IT001 to IT0...n. Each problem contains two .txt files, known01.txt and unknown.txt, representing the text pair. Rather than representing a single text, each of these documents contains a set of texts written by the same author.
Since we experimented with texts of different length, the known and unknown documents can be made of 200, 500, 1 000 and 1 500 words each. The text is already tokenized. Moreover, each folder contains 3 files:
- trutx.txt, in which we store the gold labels (Y o N) associated to each problem ID
          IT001 N
          IT002 Y
          IT003 N
          ...
- INFO.csv links the problem ID to the text pair associated with it, the gender of the authors and the gold label
          IT001	("@ Naturalia ... hai poi fatto la versione con il detergente bio ? ...")	Female	Female	N

- contents.json, a json file containing the language of the texts and a list of problem IDs

