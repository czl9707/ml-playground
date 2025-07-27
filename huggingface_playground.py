from transformers import DistilBertModel, DistilBertTokenizer

checkpoint = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer: DistilBertTokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model: DistilBertModel = DistilBertModel.from_pretrained(checkpoint)



raw_inputs = "I've been waiting for a HuggingFace course my whole life."


model_inputs = tokenizer(raw_inputs, return_tensors="pt")
print(model_inputs["input_ids"].size())
# torch.Size([1, 16])

model_inputs = tokenizer(raw_inputs, padding="longest", return_tensors="pt")
print(model_inputs["input_ids"].size())
# torch.Size([1, 16])

model_inputs = tokenizer(raw_inputs, padding="max_length", return_tensors="pt")
print(model_inputs["input_ids"].size())
# torch.Size([1, 512])

model_inputs = tokenizer(raw_inputs, padding="max_length", max_length=8, return_tensors="pt")
print(model_inputs["input_ids"].size())
# torch.Size([1, 16])

tokens = tokenizer.tokenize(raw_inputs)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
