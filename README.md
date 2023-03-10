# TextClassification

# dataset https://www.kaggle.com/datasets/crawford/20-newsgroups
# The objective of the project is to build a classifier for multi-classification. There are 20 classes (headings), with 1000 training points(documents) each. The task 
is to classify each of the documents as belonging to one of the headings(classes). To do this, we read the documents and choose the top 1000 most ocuuring key-words(we 
find the value counts for all words and remove stop-words). we save these top 1000 key-words. We don't have to worry about data leakage as these manipulations are 
based on the data in the document itself, and don't take into account the headings(classes).

#text_classification2.py
we create a dataframe with each of the 1000 key words as features, and each document as a data point (shape=(20000,1000)). Next, we find out the number of occurances of 
each key-word(feature) in each document, and fill in the values in the dataframe. we save the resulting dataframe(not included as size exceeds 25 mb).

#text_classification3.py
we implement the built-in multinomial naive bayes in sklearn and print the classification accuracy,confusion matrix, classification report.

#text_classification4.py
we build our own naive bayes classifier from scratch. 
we use the bayes theorem and assume all features to be independent.

p(class|data)=p(data|class)*p(class)/p(data)

p(data) can be ignored 
as it will be same for all classes, and we are interested in fiinding the class with the highest probability, and not the actual probabilities.

hence we need to find
  p(data|class)
 
to find this, we group the data by class.
Essentially, we need to find the probability that a key-word(feature) is present in a document given that it belongs to a given group(class).

we repeat this for all the features to find individual probabilities, then simply multiply all 1000 probabilities (since we are assuming features to be independent).

the complete procedure is as given below

#steps:-
# 1. Create a sum column containing sum for each row excluding target_y column
# 2. Group by target_y column
# 3. For 1st group, sum each column and find product of all sums for that group
# 4. divide by (sum of all (sum of each column))**(number of columns)
# 5. multiply by (length of group/total length of dataframe)
# 6. This is the probability for that particular class(group)
# 7. repeat for each group
# 8. predict group with highest probability as the answer

Finally, we print the classification accuracy, confusion matrix, classification report.

We compare the performance of our own classifier with that of the in-built classifier in sklearn.


The results are as follows:

classification accuracy on testing data:
#   built-in classifier:  0.8056
#   our own classifier:   0.7758

