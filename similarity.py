# -------------------------------------------------------------------------
# AUTHOR: Andrew Lau
# FILENAME: similarity.py
# SPECIFICATION: Find the documents that are the most similiar using cosine similarity
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: 2 hrs
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
        documents.append(row)
        #  print(row)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
docTermMatrix = []
terms = []

for x in documents:
  for word in x[1].split():
    if word not in terms:
        terms.append(word)

print(terms)
print(len(terms))

for x in documents:
  row = [0]*len(terms)

  for term in range(len(terms)):
    if terms[term] in x[1].split():
      row[term] = 1
  
  docTermMatrix.append(row)

print(docTermMatrix)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
maxSimilarity = float("-inf")
similarDocs = []

for i in range(len(docTermMatrix)):
  for j in range(i+1, len(docTermMatrix)):
    similarity = cosine_similarity([docTermMatrix[i]], [docTermMatrix[j]])[0][0]
    print(similarity)
    if similarity > maxSimilarity:
      maxSimilarity = similarity
      similarDocs = [i+1, j+1]

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print("The most similar documents are document", similarDocs[0], "and document", similarDocs[1], "with cosine similarity =", str(maxSimilarity))
