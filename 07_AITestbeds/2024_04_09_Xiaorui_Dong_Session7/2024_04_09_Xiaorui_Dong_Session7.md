# Homework
author: Xiaorui Dong

## SambaNova

| ntasks | Final Loss | Training Samples per Second |
|-------|------------|-----------------------------|
| 4     | 8.350 | 1191.7 |
| 8     | 8.309 | 2348.6 |
| 16    | 8.255 | 4605.5 |

The training speed is almost proportional to the the value of `ntasks`, and the model trained with more allowed tasks has a smaller total loss as a result of a more efficient distributed training.

\* ntask and ntasks-per-node are modified simultaneously. With --cpus-per-task=8, `16` is largest ntask can be used, given 128 cpus are available on a single node.

## Graphcore
Batch size and number of epochs are varied in my trials.
### Campaign 1
Vary batch size to [8, 16, 32] while fixing number epochs = 10 and learning rate = 0.03.

| Batch Size | Test Accuracy |
|------------|-----------------------------|
| 8          | 98.05% |
| 16         | 97.61% |
| 32         | 98.59% |

Within the above trials, the one with batch size = 32 gives the best results.

### Campaign 2
Vary the number of epochs to [10, 20, 40] while fixing batch size = 8 and learning rate = 0.03.

| N Epochs | Test Accuracy |
|------------|-----------------------------|
| 10         | 98.05% |
| 20         | 98.70% |
| 40         | 98.02% |

Within the above trials, the one with the number of epochs = 20 gives the best results.

There isn't a clear trend from the above experiments. With this naive 1D grid search, it seems like the next trial worth conducting for better performance may be batch size = 32 and n epochs = 20. However, given the randomness in the model and training process and how close the model performances are, more repeats are needed to obtain a more statistically significant result.


## Cerebras
The BERT example was run with different batch sizes of 512, 1024, and 2048.
| Batch Size | Number of samples processed | Processing Time (s) | Processing Rate (samples/s)|
|------------|-----------------------------|---------------------|----------------------------|
| 512        | 512000       | 174.285041622| 2937.72 |
| 1024       | 1024000      | 211.352523348| 4844.99 |
| 2048       | 2048000      | 304.558942337| 6724.48 |

As can be seen, the larger the batch size, the faster the sample processing rate we can have.

## Groq
Since the tested model is pretrained on the `sst2` dataset consisting of movie reviews and labels of being positive or negative, therefore I replace the dummy inputs with an input from the text `"Dune is a great movie!"`. The corresponding input generated from the tokenizer is
```
{
    input_ids: tensor(
        [[  101, 21643,  2003,  1037,  2307,  3185,   999,   102,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0]]),
    attention_mask: tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0]])
}
```
The output of the model is
```
logits: tensor([[-1.9072,  2.1699]], dtype=torch.float16)
probabilities: tensor([[0.0167, 0.9833]])
class label: 1 (positive)
```
Therefore the model predicts my "review" to be positive which is accurate.

The modified python script and the log files are distributed along with this document under `./Groq`.