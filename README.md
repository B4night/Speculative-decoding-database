# Speculative decoding with database
Using database instead of small model for speculative decoding.

Using this database:https://arxiv.org/abs/2303.00595

Using this technique for dealing with tokens returned by database:https://arxiv.org/abs/2312.12728

# verification used in lookahead
## 概率验证策略
基于概率评分来得到概率较高的分支
$$
P(C_i|S)=\prod_{j=1}^kp(t_{i,j}|S,t..) \\
log P(C_i|S)=\prod_{j=1}^k log~p(t_{i,j}|S,t..)
$$

## 加权概率验证策略
对分支中的每个token使用加权概率，比如说更重视序列前半部分或后半部分的准确性。

# Metheology
计算db-answer每个token的prob和，然后除以token数，得到avg_prob

1. 以avg_prob的概率接收db_answer
2. 比较avg_prob和threshold

# Database choice
- bio2rdf

# Dataset choice
https://github.com/ad-freiburg/large-qa-datasets?tab=readme-ov-file

# work flow
1. Original prompt
2. Seq2seq model, convert original prompt to SPARQL
3. Query from RDF database, return answer
4. 