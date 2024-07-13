# Step 1 - Experimenting with Text files

The model we will be using for this 1st section is called a [Bigram](https://web.stanford.edu/~jurafsky/slp3/3.pdf) model which is a type of Natural language processing (NLP) model that predicts a word based on the immediately preceding word.

Text file used is the book Wizard of OZ which you can download from Gutenberg library for free.
<br>Click link and make sure you select "Plain Text UTF-8"
<br>https://www.gutenberg.org/ebooks/22566

```python
# Bring in text file "Wizard of OZ"
with open('data/wizard_of_oz.txt', 'r', encoding='utf=8') as f:
  text = f.read()
# print(text[:200])
# bring in all our uniqye text characters as a set and sort
chars = sorted(set(text))
print(chars)
print(len(chars)) # 81 unique character values
```
