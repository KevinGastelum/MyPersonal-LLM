# Building a GPT-like LLM from Scratch step by step

## [Step 1 - Experimenting with Text files](1.bigramModel.ipynb)

The model we will be using for this 1st section is called a [Bigram](https://web.stanford.edu/~jurafsky/slp3/3.pdf) model. Which is a type of Natural language processing (NLP) model that predicts a word based on the immediately preceding word.

Text file used is the book Wizard of OZ which you can download from Gutenberg library for free.
<br>Make sure to select "Plain Text UTF-8"
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

OUTPUT:
['\n', ' ', '!', '"', '&', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\ufeff']
81

```

### Pytorch - Deep Learning framework used to train neural networks

Pytorch requires Nvidia GPU with CUDA enabled or use one free with [Google Colab](https://colab.research.google.com/)<br>
Download Pytorch - https://pytorch.org/get-started/locally/<br>
Enable GPU CUDA tensors if available for faster training times<br>
https://pytorch.org/tutorials/beginner/basics/intro.html

```python
# Check if GPU CUDA tensors are available otherwise use CPU
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

```

### Comparing speeds: Pytorch (GPU) vs Numpy (CPU) for multidimensional array computations

```python
# Speed Check - Pytorch (GPU) vs Numpy (CPU)
torch_rand1 = torch.rand(100, 100, 100, 100).to(device)
torch_rand2 = torch.rand(100, 100, 100, 100).to(device)
np_rand1 = torch.rand(100, 100, 100, 100)
np_rand2 = torch.rand(100, 100, 100, 100)

start_time = time.time()

rand = (torch_rand1 @ torch_rand2)

end_time = time.time()

elapsed_time = end_time - start_time
print(f'Pytorch CUDA: {elapsed_time:.8f}s')

start_time = time.time()

rand = np.multiply(np_rand1, np_rand2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Numpy: {elapsed_time:.8f}s')

OUTPUT:
Pytorch CUDA: 0.00800133s
Numpy: 0.19042158s

```
