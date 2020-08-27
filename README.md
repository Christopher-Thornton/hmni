<p align="center">
  <img src="https://github.com/Christopher-Thornton/hmni/blob/master/nametag.png?raw=true" alt="logo" />
</p>

# HMNI
Fuzzy name matching with machine learning. Perform common fuzzy name matching tasks including similarity scoring, record linkage, deduplication and normalization.

HMNI is trained on an internationally-transliterated Latin firstname dataset, where precision is afforded priority.

|    Model    |  Accuracy | Precision |   Recall  |  F1-Score 
|-------------|-----------|-----------|-----------|-----------
| HMNI-Latin  | 0.9393    | 0.9255    | 0.7548    | 0.8315    

For an introduction to the methodology and research behind HMNI, please refer to my [blog post](https://towardsdatascience.com/fuzzy-name-matching-with-machine-learning-f09895dce7b4).

## Requirements
### Python 3.5–3.8
-  tensorflow
-  scikit-learn
-  fuzzywuzzy
-  abydos
-  unidecode

## QUICK USAGE GUIDE
## Installation
Using PIP via PyPI
```bash
pip install hmni
```
#### Initialize a Matcher Object
```python
import hmni
matcher = hmni.Matcher(model='latin')
```
#### Single Pair Similarity
```python
matcher.similarity('Alan', 'Al')
# 0.6838303319889133

matcher.similarity('Alan', 'Al', prob=False)
# 1

matcher.similarity('Alan Turing', 'Al Turing', surname_first=False)
# 0.6838303319889133
```
#### Record Linkage
```python
import pandas as pd

df1 = pd.DataFrame({'name': ['Al', 'Mark', 'James', 'Harold']})
df2 = pd.DataFrame({'name': ['Mark', 'Alan', 'James', 'Harold']})

merged = matcher.fuzzymerge(df1, df2, how='left', on='name')
```
#### Name Deduplication and Normalization
```python
names_list = ['Alan', 'Al', 'Al', 'James']

matcher.dedupe(names_list, keep='longest')
# ['Alan', 'James']

matcher.dedupe(names_list, keep='frequent')
# ['Al, 'James']

matcher.dedupe(names_list, keep='longest', replace=True)
# ['Alan, 'Alan', 'Alan', 'James']
```
## Matcher Methods and Parameters
> **similarity**(name_a, name_b, prob=True, surname_first=False)
* **name_a** *(str)* -- First name for comparison
* **name_b** *(str)* -- Second name for comparison
* **prob** *(bool)* -- If True return a predicted probability, else binary class label
* **threshold** *(float)* -- Prediction probability threshold for positive match (0.5 by default)
* **surname_first** *(bool)* -- If name strings start with surname (False by default)

> **fuzzymerge**(df1, df2, how='inner', on=None, left_on=None, right_on=None, indicator=False, limit=1, threshold=0.5, allow_exact_matches=True, surname_first=False)
* **df1** *(pandas DataFrame or named Series)* -- First/Left object to merge with
* **df2** *(pandas DataFrame or named Series)* -- Second/Right object to merge with
* **how** *(str)* -- Type of merge to be performed
    * `inner` (default): Use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys
    * `left`: Use only keys from left frame, similar to a SQL left outer join; preserve key order
    * `right`: Use only keys from right frame, similar to a SQL right outer join; preserve key order
    * `outer`: Use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically
* **on** *(label or list)* -- Column or index level names to join on. These must be found in both DataFrames
* **left_on** *(label or list)* -- Column or index level names to join on in the left DataFrame.
* **right_on** *(label or list)* -- Column or index level names to join on in the right DataFrame.
* **indicator** *(bool)* -- If True, adds a column to output DataFrame called “_merge” with information on the source of each row (False by default)
* **limit** *(int)* -- Top number of name matches to consider (1 by default)     
* **threshold** *(float)* -- Prediction probability threshold for positive match (0.5 by default)       
* **allow_exact_matches** *(bool)* -- If True allow merging on exact name matches, else do not consider exact matches (True by default)
* **surname_first** *(bool)* -- If name strings start with surname (False by default)

> **dedupe**(names, threshold=0.5, keep='longest', reverse=True, limit=3, replace=False, surname_first=False)
* **names** *(list)* -- List of names to dedupe
* **threshold** *(float)* -- Prediction probability threshold for positive match (0.5 by default)
* **keep** *(str)* -- Specifies method for keeping one of multiple alternative names 
    * `longest` (default): Keeps longest name
    * `frequent`: Keeps most frequent name in names list
* **reverse** *(bool)* -- If True will sort matches descending order, else ascending (True by default)
* **limit** *(int)* -- Top number of name matches to consider (3 by default)
* **replace** *(bool)* -- If True return normalized name list, else return deduplicated name list (False by default) 
* **surname_first** *(bool)* -- If name strings start with surname (False by default)

## Contributing
Pull requests are welcome. 
For developers wishing to build a model using Latin or non-Latin writing systems (Chinese, Cyrillic, Arabic), 
jupyter notebooks are shared in the `dev` folder to build models using similar methods. 

## License
[MIT](https://choosealicense.com/licenses/mit/)
