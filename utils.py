import torch
import pandas as pd
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def create_lookup_table():
    if not hasattr(create_lookup_table, "cache"):
        idx_to_vocab_df = pd.read_csv('captions_idx_to_vocab.csv')
        IDs = idx_to_vocab_df['ID'].to_list()
        words = idx_to_vocab_df['token'].to_list()

        create_lookup_table.idx_to_vocab = { int(ID): word for ID, word in zip(IDs, words) }
        create_lookup_table.vocab_to_idx = { word: int(ID) for ID, word in zip(IDs, words) }


    return create_lookup_table.idx_to_vocab, create_lookup_table.vocab_to_idx

def bleu_score_sum(outputs, targets):
    sum = 0.0

    idx_to_vocab, _ = create_lookup_table()

    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()

    for output, target in zip(outputs, targets):
        output = [ idx_to_vocab[idx] for idx in output if idx not in [0,1,2,3] ] # exclude <pad>, <start>, <end>, <unk>
        target = [ idx_to_vocab[idx] for idx in target if idx not in [0,1,2,3] ] # exclude <pad>, <start>, <end>, <unk>

        smooth_fn = SmoothingFunction()

        sum += sentence_bleu(references=[ target ], hypothesis=output, smoothing_function=smooth_fn.method1)

    return sum

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

# First layer -> (input_size, hidden_size)
# Subsequent layers -> (hidden_size, hidden_size)
def stack_rnn_cells(cell_class, input_size, hidden_size, num_layers):
    rnn_cells = nn.ModuleList()

    for l in range(num_layers):
        if l == 0:
            rnn_cells.append(cell_class(input_size, hidden_size))
        else:
            # The input is the hidden state of the previous layer
            rnn_cells.append(cell_class(hidden_size, hidden_size))

    return rnn_cells


if __name__ == "__main__":
    # Example usage
    outputs = torch.tensor([[5,6,7,5,6,7,5,6,7,5,6,7], [5,6,7,5,6,7,5,6,7,5,6,7]])
    targets = torch.tensor([[5,6,7,5,6,7,5,6,7,5,6,7], [5,6,7,5,6,7,5,6,7,5,6,7]])

    bleu_score = bleu_score_sum(outputs, targets)
    print(f"BLEU Score: {bleu_score:.4f}")