# Spam Filter with Naive Bayes

In this study, we will apply the Naive Bayes algorithm to a database of SMS messages which was already classified as spam or not by humans. The goal is to reach 80% of accuracy in this classification.

The dataset was put together by Tiago A. Almeida and José María Gómez Hidalgo, and it can be downloaded from <a href="https://archive.ics.uci.edu/ml/datasets/sms+spam+collection" target="_blank">The UCI Machine Learning Repository</a>. The data collection process is described in more detail <a href="http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/#composition" target="_blank">on this page</a>, where you can also find some of the authors' papers.

## Exploring the Dataset


```python
# Import libraries
import pandas as pd
import re
```


```python
# Load data
sms = pd.read_csv('SMSSpamCollection', sep = '\t', header = None, names = ['Label', 'SMS'])
sms.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
sms.shape
```




    (5572, 2)




```python
# Checking the percentage of spams in the data set
round(sms['Label'].value_counts() / sms.shape[0] * 100, 2)
```




    ham     86.59
    spam    13.41
    Name: Label, dtype: float64



From the whole dataset, only 13.41% of the messages are spam.

## Training and testing split

Dividing the dataset in 50/50 to training it using multinomial Naive Bayes algorithm


```python
# Divide the dataset between training and testing
sms = sms.sample(frac = 1, random_state = 42)
sms.reset_index(drop = True, inplace = True)

size = sms.shape[0] / 2

training = sms.loc[:size,]
testing = sms.loc[size:,]
testing.reset_index(drop = True, inplace = True)

print(training.shape)
print(testing.shape)
```

    (2787, 2)
    (2786, 2)


Checking the percentage of spam in both new datasets


```python
# Checking the percentage of spam in the train data sets
round(training['Label'].value_counts(normalize = True) * 100, 2)
```




    ham     86.72
    spam    13.28
    Name: Label, dtype: float64




```python
# Checking the percentage of spam in the test data sets
round(testing['Label'].value_counts(normalize = True) * 100, 2)
```




    ham     86.47
    spam    13.53
    Name: Label, dtype: float64



## Transforming training dataset


```python
# Removing pontuaction and change letters to lowercase
pd.options.mode.chained_assignment = None
training['SMS'] = training['SMS'].apply(lambda sms: re.sub('\W', ' ', sms))
training['SMS'] = training['SMS'].str.lower()
pd.options.mode.chained_assignment = 'warn'
training.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>squeeeeeze   this is christmas hug   if u lik ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>and also i ve sorta blown him off a couple tim...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>mmm thats better now i got a roast down me  i ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>mm have some kanji dont eat anything heavy ok</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>so there s a ring that comes with the guys cos...</td>
    </tr>
  </tbody>
</table>
</div>



Get all unique words in the training dataset and creating a vocabulary


```python
# Get the vocabulary
vocabulary = training['SMS'].str.split().values.tolist()
vocabulary = list(set(sum(vocabulary, [])))
```

## Preparing dataset

Creating a new dataset with words frequency per message


```python
# Create a dictionary with words frequency
word_counts_per_sms = {unique_word: [0] * training.shape[0] for unique_word in vocabulary}

for index, sms in enumerate(training['SMS']):
    for word in sms.split():
        word_counts_per_sms[word][index] += 1
```


```python
# Convert the dictionary in DataFrame
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shit</th>
      <th>cutter</th>
      <th>starting</th>
      <th>red</th>
      <th>somethin</th>
      <th>atlast</th>
      <th>transaction</th>
      <th>waste</th>
      <th>08712402902</th>
      <th>08718725756</th>
      <th>...</th>
      <th>shove</th>
      <th>demand</th>
      <th>300603</th>
      <th>night</th>
      <th>09065394514</th>
      <th>christmas</th>
      <th>enjoying</th>
      <th>skye</th>
      <th>pizza</th>
      <th>indeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5976 columns</p>
</div>




```python
# Concatenating the 2 dataframes
training_ = pd.concat([training, word_counts], axis = 1)
training_.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>shit</th>
      <th>cutter</th>
      <th>starting</th>
      <th>red</th>
      <th>somethin</th>
      <th>atlast</th>
      <th>transaction</th>
      <th>waste</th>
      <th>...</th>
      <th>shove</th>
      <th>demand</th>
      <th>300603</th>
      <th>night</th>
      <th>09065394514</th>
      <th>christmas</th>
      <th>enjoying</th>
      <th>skye</th>
      <th>pizza</th>
      <th>indeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>squeeeeeze   this is christmas hug   if u lik ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>and also i ve sorta blown him off a couple tim...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>mmm thats better now i got a roast down me  i ...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>mm have some kanji dont eat anything heavy ok</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>so there s a ring that comes with the guys cos...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5978 columns</p>
</div>



This is the final dataset with the words frequency and the target which is the Label

## Creating the constants to training

Creating the constant values that don't need to be calculated in every new message that has to be classified


```python
# Create the n_spam
n_spam = len(training_[training_.Label == 'spam']['SMS'].str.split().sum())
n_spam
```




    9343




```python
# Create the n_ham
n_ham = len(training_[training_.Label == 'ham']['SMS'].str.split().sum())
n_ham
```




    35228




```python
# Create the n_vocabulary
n_vocabulary = len(vocabulary)
n_vocabulary
```




    5976




```python
# Create the p_spam
spam_messages = training_[training_['Label'] == 'spam']
p_spam = len(spam_messages) / len(training_)
```


```python
# Create the p_ham
ham_messages = training_[training_['Label'] == 'ham']
p_ham = len(spam_messages) / len(training_)
```


```python
# Create alpha
alpha = 1
```

## Calculating probabilities for every word in the vocabulary

For every word in the vocabulary, it will calculate the probability that it is or is not a spam


```python
# Calculating the probability of a word be or not be a spam
p_spam_w = {word: 0 for word in vocabulary}
p_ham_w = p_spam_w.copy()

training_spam = training_[training_.Label == 'spam']
training_spam = training_[training_.Label == 'ham']

for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()
    n_word_given_ham = ham_messages[word].sum()
    
    p_word_given_spam = (
        (n_word_given_spam + alpha) / (n_spam + alpha * n_vocabulary)
    )
    p_spam_w[word] = p_word_given_spam
    
    p_word_given_ham = (
        (n_word_given_ham + alpha) / (n_ham + alpha * n_vocabulary)
    )
    p_ham_w[word] = p_word_given_ham
```

## Classifing messages

Creating a function that receives a message and classifies it as spam or not


```python
# Creating function to classify messages
def classify(message, p_spam, p_ham, p_spam_w, p_ham_w):
    message = re.sub('\W', ' ', message).lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in p_spam_w:
            p_spam_given_message *= p_spam_w[word]
        if word in p_ham_w:
            p_ham_given_message *= p_ham_w[word]
        
    
    if p_spam_given_message > p_ham_given_message:
        return 'spam'
    elif p_spam_given_message < p_ham_given_message:
        return 'ham'
    else:
        return None

print(classify('Test the classify function!!!', p_spam, p_ham, p_spam_w, p_ham_w))
print(classify('Secrety classify function!!! Winner!', p_spam, p_ham, p_spam_w, p_ham_w))
```

    ham
    spam


## Predicting the messages in the Testing data set

Using the classification function created above to classify the testing dataset and verify its accuracy


```python
testing['Predicted'] = testing['SMS'].apply(classify, args = (p_spam, p_ham, p_spam_w, p_ham_w))
testing.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>jus chillaxin  what up</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Hey leave it. not a big deal:-) take care.</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>I am real, baby! I want to bring out your inne...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam</td>
      <td>U have a secret admirer who is looking 2 make ...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Cool, text me when you're ready</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ham</td>
      <td>Ok then i will come to ur home after half an hour</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spam</td>
      <td>We tried to contact you re your reply to our o...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ham</td>
      <td>Honey, can you pls find out how much they sell...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spam</td>
      <td>Message Important information for O2 user. Tod...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ham</td>
      <td>Wat time ü wan today?</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Measuring the accuracy
testing[testing.Label == testing.Predicted].shape[0] / testing.shape[0]
```




    0.9763101220387652



The classification has an accuracy of 97.63%

## Modifying the classification function

To try to improve the accuracy it will change the classification function to consider the punctuation and sensitive case


```python
# Mofying the classification function
def classify(message, p_spam, p_ham, p_spam_w, p_ham_w):
    message = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', r' \g<0> ', message).strip().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in p_spam_w:
            p_spam_given_message *= p_spam_w[word]
        if word in p_ham_w:
            p_ham_given_message *= p_ham_w[word]
        
    
    if p_spam_given_message > p_ham_given_message:
        return 'spam'
    elif p_spam_given_message < p_ham_given_message:
        return 'ham'
    else:
        return None

print(classify('Test the classify function!!!', p_spam, p_ham, p_spam_w, p_ham_w))
print(classify('Secrety classify function!!! Winner!', p_spam, p_ham, p_spam_w, p_ham_w))
```

    ham
    None


## Re-testing the prediction

With the new function, it can try to predict again and evaluate the new accuracy


```python
testing['Predicted'] = testing['SMS'].apply(classify, args = (p_spam, p_ham, p_spam_w, p_ham_w))
testing.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
      <th>SMS</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>jus chillaxin  what up</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Hey leave it. not a big deal:-) take care.</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ham</td>
      <td>I am real, baby! I want to bring out your inne...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spam</td>
      <td>U have a secret admirer who is looking 2 make ...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Cool, text me when you're ready</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ham</td>
      <td>Ok then i will come to ur home after half an hour</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spam</td>
      <td>We tried to contact you re your reply to our o...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ham</td>
      <td>Honey, can you pls find out how much they sell...</td>
      <td>ham</td>
    </tr>
    <tr>
      <th>8</th>
      <td>spam</td>
      <td>Message Important information for O2 user. Tod...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ham</td>
      <td>Wat time ü wan today?</td>
      <td>ham</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Measuring the accuracy
testing[testing.Label == testing.Predicted].shape[0] / testing.shape[0]
```




    0.9422110552763819



Consider the punctuation and the sensitive case didn't help to increase the accuracy
