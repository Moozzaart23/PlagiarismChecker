
# Plagiarism Checker
**A Plagiarism Checker for text documents**
This is mainly designed to go through all the files in the corpus present and check the similarity of the input document based on the documents present in the corpus. Firstly vector space model built with the help of tf-idf and then KL Divergence is used to find the similarity of the query document with other documents present in the corpus

## Getting Started

 - Python (Version 3.7 and above)
 - Pip (Latest Version recommended)
 - Git
 
 ## Installation
 - Clone the repository in your preferred directory using the following command
```
	git clone https://github.com/Moozzaart23/PlagiarismChecker.git
```
 - Change your working directory to Plagarism_Checker
```
	cd Plagarism_Checker 
```
  - Create a python terminal to install **nltk** dependencies
```
	>>> import nltk
	>>> nltk.download('pukt')
	>>> nltk.download('stopwords')
	>>> exit()
```
- Copy the documents which you want to check for along with the query document in the same directory
- Run the following command to check for query
```
	python check.py 
```
 
 ## Team
 - [Anish Dey](https://github.com/Moozzaart23)
 - [Anurag Behera](https://github.com/19981999ab)
 - [Shuvam Banerjee](https://github.com/player1798)
