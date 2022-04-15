

#基于sentence_transformers快速实现无监督学习
import logging
import math

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers import models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from torch.utils.data import DataLoader

def load_STS_data(path):
    with open(path, 'r', encoding='utf8') as f:
        return [(line.split("||")[1], line.split("||")[2], int(line.split("||")[3])) for line in f]

# 训练参数
model_name = 'hfl/rbt6'
train_batch_size = 6
num_epochs = 2
max_seq_length = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 模型保存路径
model_save_path = './saved_unsup/best_model.pkl'

# 建立模型
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode="cls",
                               pooling_mode_cls_token=True)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
model[0].auto_model.attention_probs_dropout_prob = 0.1 #dropout概率
model[0].auto_model.hidden_dropout_prob = 0.1  #

# 准备训练集
sts_vocab = load_STS_data("./datasets/STS-B/cnsd-sts-train.txt")[:100]
all_vocab = [x[0] for x in sts_vocab] + [x[1] for x in sts_vocab]
simCSE_data = all_vocab
print("The len of SimCSE unsupervised data is {}".format(len(simCSE_data)))
train_samples = []
for data in all_vocab:
    train_samples.append(InputExample(texts=[data, data]))

# 准备验证集和测试集
dev_data = load_STS_data("./datasets/STS-B/cnsd-sts-dev.txt")[:100]
test_data = load_STS_data("./datasets/STS-B/cnsd-sts-test.txt")[:100]
dev_samples = []
test_samples = []
for data in dev_data:
    dev_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))
for data in test_data:
    test_samples.append(InputExample(texts=[data[0], data[1]], label=data[2] / 5.0))

# 初始化评估器,EmbeddingSimilarityEvaluator相似度评估器,SimilarityFunction.COSINE余弦相似度
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='sts-dev',
                                                                 main_similarity=SimilarityFunction.COSINE)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size,
                                                                  name='sts-test',
                                                                  main_similarity=SimilarityFunction.COSINE)

# We train our model using the MultipleNegativesRankingLoss，即对比损失
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1)  # Evaluate every 10% of the data
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# 模型训练
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          show_progress_bar=False,
          output_path=model_save_path,
          optimizer_params={'lr': 2e-5},
          use_amp=False  # Set to True, if your GPU supports FP16 cores
          )

# 测试集上的表现 看了下源码,res是SimilarityFunction中距离值(余弦,,欧几里德,曼哈顿,点积）最大的一个的是斯皮尔曼得分
model = SentenceTransformer(model_save_path)
res=test_evaluator(model)
print(res)