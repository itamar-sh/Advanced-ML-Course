import torch
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mingpt.trainer import Trainer
from mingpt import bpe as bpe
from mingpt.model import GPT


class AliceDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def get_vocab(self):
        return len(np.unique(self.tokens))

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        return torch.tensor(self.tokens[i: self.block_size + i]), \
               torch.tensor(self.tokens[i + 1: self.block_size + i + 1])


class AliceDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def get_vocab(self):
        return len(np.unique(self.tokens))

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, i):
        return torch.tensor(self.tokens[i: self.block_size + i]), \
               torch.tensor(self.tokens[i + 1: self.block_size + i + 1])


def getDataSet(alice_path):
    with open(alice_path, 'r') as f:
        alice_text = f.read()
    encoder = bpe.get_encoder()
    tokens = encoder.encode(alice_text)  # np.array(e.encode(alice_text), dtype=np.int64)

    block_size = 64
    dataset = AliceDataset(tokens, block_size)
    return dataset


def Q1_Train_AR_model(alice_path):
    print("Q1")
    dataset = getDataSet(alice_path)

    config = GPT.get_default_config()
    config.model_type = 'gpt2'  # 127M params
    config.vocab_size = 50257
    config.block_size = 64
    model = GPT(config)
    config_train = Trainer.get_default_config()
    config_train.learning_rate = 5e-4
    config_train.max_iters = 1000   # how many iteration for training
    config_train.num_workers = 1
    trainer = Trainer(config_train, model, dataset)

    losses = []
    iterations = []
    # iteration callback
    def batch_end_callback(trainer):  # Run of end of each callback by the model
        losses.append(trainer.loss.item())
        iterations.append(trainer.iter_num)

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    plt.plot(iterations, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.show()
    return model


def Q2_Inversion(model):
    print("Q2")
    model.eval()
    input_vector = torch.nn.Parameter(torch.randn((1, 9, 768)), requires_grad=True).to(device)
    input_vector.retain_grad()
    sentence_to_learn = "I am a little squirrel holding a walnut"
    tokenized_sentence = torch.tensor(bpe.get_encoder().encode(sentence_to_learn), dtype=torch.long).to(device)
    optimizer = optim.Adam([input_vector], lr=0.01)
    iterations = []
    losses = []
    for i in range(1000):
        for i, token in enumerate(tokenized_sentence):
            optimizer.zero_grad()
            logits, loss = model.forward(torch.cat((input_vector, model.transformer.wte(tokenized_sentence[:i]).unsqueeze(0)), dim=1), token, embedded=True)
            loss.backward()
            optimizer.step()
        if (i + 1) % 100 == 0:
            iterations.append(i+1)
            losses.append(loss.item())
            sent = model.generate(input_vector, 9, embedded=True)
            sent = [int(s) for s in sent]
            print(f"The current result is:\n{''.join(bpe.get_encoder().decode(sent))}")
    print("The final result is:\n")
    sent = model.generate(input_vector, 9, embedded=True)
    sent = [int(s) for s in sent]
    print(''.join(bpe.get_encoder().decode(sent)))
    plt.plot(iterations, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Losses')
    plt.show()
    return model


def Q_3_attention(model, device):
    print("Q3")
    prefix = "She has to go to look for"
    prefix_par = torch.tensor(bpe.get_encoder().encode(prefix), dtype=torch.long, device=device).unsqueeze(0)
    sent_tokens, attention_result = model.generate(prefix_par, 7, attention=6)
    sent_tokens = sent_tokens.squeeze().tolist()
    str_sent = ''.join(bpe.get_encoder().decode(sent_tokens))
    print(f"full sentence:\n{str_sent}")
    mean = attention_result[:, :, 10, :].mean(dim=1)
    print(f"mean value of the scores: {mean}")
    scores_sum = 0
    scores = []
    tokens_list = []
    for i in range(len(sent_tokens)-1):
        token_to_add = bpe.get_encoder().decode([sent_tokens[i]]) if i < 11 else bpe.get_encoder().decode([sent_tokens[i+1]])
        tokens_list.append(token_to_add)
        scores.append(mean[0, i].item())
        scores_sum += mean[0, i].item()
    print("sum of scores:", scores_sum)
    plt.figure(figsize=(8, 8))
    plt.bar(tokens_list, scores, width=0.4)

    plt.xlabel("Words")
    plt.ylabel("Score")
    plt.title("Attention")
    plt.show()


def Q_4_attention(model, device):
    print("Q4")
    prefix = "She has to go to look for"
    prefix_par = torch.tensor(bpe.get_encoder().encode(prefix), dtype=torch.long, device=device).unsqueeze(0)
    print(prefix_par.shape)
    sent_tokens, attention_result = model.generate(prefix_par, 7, first_attention=6)
    print(attention_result.shape)
    sent_tokens = sent_tokens.squeeze().tolist()
    str_sent = ''.join(bpe.get_encoder().decode(sent_tokens))
    print(f"full sentence:\n{str_sent}")
    mean = attention_result[:, :, 10, :].mean(dim=1)
    print(f"mean value of the scores: {mean}")
    scores_sum = 0
    scores = []
    tokens_list = []
    for i in range(len(sent_tokens)-1):
        token_to_add = bpe.get_encoder().decode([sent_tokens[i]]) if i < 11 else bpe.get_encoder().decode([sent_tokens[i+1]])
        tokens_list.append(token_to_add)
        scores.append(mean[0, i].item())
        scores_sum += mean[0, i].item()
    print("sum of scores:", scores_sum)
    plt.figure(figsize=(8, 8))
    plt.bar(tokens_list, scores, width=0.4)

    plt.xlabel("Words")
    plt.ylabel("Score")
    plt.title("Attention")
    plt.show()


def Q_5_probs(model, device):
    print("Q5")
    sentence_tokens = model.generate(torch.tensor(bpe.get_encoder().encode("Beginning of a journey is"), dtype=torch.long, device=device).unsqueeze(0), 10)[0].squeeze().tolist()
    sentence_str = ''.join(bpe.get_encoder().decode(sentence_tokens))
    sentence_par = torch.tensor(bpe.get_encoder().encode(sentence_str), dtype=torch.long, device=device).unsqueeze(0)
    logits = model(sentence_par)
    sentence_probs = torch.softmax(logits[0], dim=-1)[0]

    print('Generated Sentence:')
    print(sentence_str)
    print('Score:', sum([sentence_probs[i-1][sentence_par[0][i]] for i in range(1, sentence_probs.shape[0])]))


def main_function():
    alice_path = "alice_in_wonderland.txt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Q1_Train_AR_model(alice_path)
    Q2_Inversion(model, device)
    Q_3_attention(model, device)
    Q_4_attention(model, device)
    Q_5_probs(model, device)


if __name__ == '__main__':
    main_function()