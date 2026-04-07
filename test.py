import torch
import nltk

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))

nltk.download("punkt")  # for BLEU tokenization