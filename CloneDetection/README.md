# Clone Detection

> [!NOTE]
> Following the instructions are from the [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder#2-similarity-between-code-and-nl) repository

### Recommendation
Using UniXCode without fine-tuning: https://github.com/microsoft/CodeBERT/issues/196


### No fine-tuned model example
```bash
poetry run python example_no_finetune.py
```
```
Similarity between first function and nl: tensor([[0.9183]], grad_fn=<ViewBackward0>)
Similarity between second function and nl: tensor([[0.7548]], grad_fn=<ViewBackward0>)
```
