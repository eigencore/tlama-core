# **Run Tlama 124M Model 🏃‍♂️**

We have a small 124M model available for quick testing. Here’s how to run it:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("eigencore/tlama-124M", trust_remote_code=True)

prompt = "Once upon a time in a distant kingdom..."
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

You can explore more on the [Hugging Face page](https://huggingface.co/eigencore/tlama-124M).