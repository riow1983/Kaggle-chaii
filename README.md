# Kaggle-chaii
![input file image]()<br>
https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering<br>
どんなコンペ?:<br>
開催期間: 2021-08-11 ~ 2021-11-15<br>
[結果]()<br>  
<br>
<br>
<br>
***
## 実験管理テーブル
|commitSHA|comment|Local CV|Public LB|
|----|----|----|----|
|-|-|-|0.0010|
<br>

## Late Submissions
|commitSHA|comment|Local CV|Private LB|Public LB|
|----|----|----|----|----|
<br>


## My Assets
[notebook命名規則]  
- kagglenb001-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- nb001-hoge.ipynb: kagglenb001-hoge.ipynbをlocalにpullしlocalで変更を加えるもの. 番号はkagglenb001-hoge.ipynbと共通.
- localnb001-hoge.ipynb: localで新規作成されたnotebook. 
- l2knb001-hoge.ipynb: localnb001-hoge.ipynbをKaggle platformにpushしたもの. 番号はlocalnb001-hoge.ipynbと共通.

#### Code
作成したnotebook等の説明  
|name|url|input|output|status|comment|
|----|----|----|----|----|----|
|localnb001-export-transformers|-|-|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)|Done|`bert-base-multilingual-cased`のファイルをKaggle Datasetとしてexport|
|kagglenb001-chaii-eda|-|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)|submission.csv|Done|`bert-base-multilingual-cased`による予測 (w/o fine-tuning)|
<br>





***
## 参考資料
#### Snipets
<br>

```python
# オーバーラップを確保しながら特定長のシーケンスを得るためのポジションを取得するループ

max_len = 512
context_len = 5121
overlap=60

for i in range(0, context_len, max_len-overlap):
    print("start_position:", i)
    print("end_position:", min(i + max_len, context_len-1))
    print()
```



#### Papers
|name|url|status|comment|
|----|----|----|----|
|BERT Based Multilingual Machine Comprehension in English and Hindi|[URL](https://arxiv.org/pdf/2006.01432.pdf)|未読|-|
<br>


#### Blogs / Qiita / etc.
|name|url|status|comment|
|----|----|----|----|
|BERT入門|[URL](https://www.slideshare.net/matsukenbook/bert-217710964)|Done|Kaggle inference時 (=internetオフ時),<br>:hugs: transformersのメタファイルを取得する方針が参考になった|
|Unicode block|[URL](https://en.wikipedia.org/wiki/Unicode_block)|Done|各言語文字のblock rangeが示されている|
|iHateRegex|[URL](https://ihateregex.io/expr/hyphen-word-break/)|Done|正規表現の動作確認がブラウザ上でできる|
|The Unicode Standard, Version 13.0 > Devanagari|[URL](https://unicode.org/charts/PDF/U0900.pdf)|Done|Devanagariのunicodeの解説書|
|Adapting BERT question answering for the medical domain|[URL](https://medium.com/analytics-vidhya/adapting-bert-question-answering-for-the-medical-domain-2085ada8ceb1)|Done|Dealing with a long contextセクションに512トークン長以内にcontextを分割する手法が言葉で説明されている|
|Question Answering with a fine-tuned BERT|[URL](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626)|Done|fine-tunedということでfine-tuneする実装例ではなく, fine-tune済みのモデルの利用例になっていることに注意|
|【突然GitHubにpushできなくなった】 トークン生成で解決：The requested URL returned error: 403|[URL](https://zenn.dev/yuri0427/articles/9587ae6a578ee9)|Done|GitHub連携にパスワード方式が使えなくなった2021年8月13日以降はこのトークン方式になる|
|(:hugs:) ValueError: char_to_token() is not available when using Python based tokenizers|[URL](https://www.gitmemory.com/issue/huggingface/transformers/12201/862549850)|Done|:hugs: `char_to_token`メソッドを使うならfast tokenizerを使うべし|
<br>


#### Official Documentation or Tutorial
|name|url|status|comment|
|----|----|----|----|
|(:hugs:) Fine-tuning a pretrained model|[URL](https://huggingface.co/transformers/training.html)|参照中|Trainer APIや素のPyTorchによる:hugs: pretrained modelのfine-tuning実装例|
|(:hugs:) Fine-tuning with custom datasets|[URL](https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-custom-datasets)|参照中|IMDb(Sequence Classification), W-NUT(Token Classification = NER), SQuAD2.0(QA) それぞれについてのfine-tuning実装例|
|(PyTorch) DATASETS & DATALOADERS|[URL](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)|参照中|PyTorch Dataset, DataLoaderの公式実装解説|
|(PyTorch) Custom loss functions|[URL](https://discuss.pytorch.org/t/custom-loss-functions/29387)|Done|PyTorchでCustom loss functionを実装する方法|
|(:hugs:) BERT|[URL](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforquestionanswering#berttokenizer)|参照中|:hugs:BERTメインページ|
|(:hugs:) Tokenizer|[URL](https://huggingface.co/transformers/main_classes/tokenizer.html#tokenizer)|参照中|:hugs:Tokenizerメインページ|
|(:hugs:) Preprocessing data|[URL](https://huggingface.co/transformers/preprocessing.html)|参照中|tokenizerにbatch sentence Aとbatch sentence Bを入力した場合に出力されるものを確認できる|
<br>

#### StackOverflow
|name|url|status|comment|
|----|----|----|----|
|BERT tokenizer & model download|[URL](https://stackoverflow.com/questions/59701981/bert-tokenizer-model-download)|Done|:hugs: transformersのメタファイルをダウンロードする例.<br>ただし最近は格納先がHugging Face Platformに変更になった模様.|
|python pandas: apply a function with arguments to a series|[URL](https://stackoverflow.com/questions/12182744/python-pandas-apply-a-function-with-arguments-to-a-series)|Done|apply対象の関数への引数の渡し方|
|Python: Find a substring in a string and returning the index of the substring|[URL](https://stackoverflow.com/questions/21842885/python-find-a-substring-in-a-string-and-returning-the-index-of-the-substring)|Done|`string`型のfindメソッドでできる|


#### GitHub
|name|url|status|comment|
|----|----|----|----|
|個人アクセストークンを使用する|[URL](https://docs.github.com/ja/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)|Done|GitHub連携にパスワード方式が使えなくなった2021年8月13日以降はこのトークン方式になる|
|(PyTorch) How to slice a BatchEncoding object into desired batch sizes?|[URL](https://github.com/huggingface/tokenizers/issues/577)|Done|`tokenizer`はbatchにしたtextを受け取れる仕様|
<br>

#### Hugging Face Platform
|name|url|status|comment|
|----|----|----|----|
|bert-base-multilingual-cased|[URL](https://huggingface.co/bert-base-multilingual-cased)|Done|`config.json`, `pytorch_model.bin`, `vocab.txt`を取得|
<br>

#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
|coleridge_regex_electra|[URL](https://www.kaggle.com/nbroad/no-training-question-answering-model/data?scriptVersionId=66240356)|参考|Coleridgeコンペの47th solution.<br>transformers pipelineを使ったQA実装例が非常に分かりやすく, 本コンペでも参考とした.|
<br>


#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
<br>

#### Kaggle Discussion
|name|url|status|comment|
|----|----|----|----|
<br>



***
## Diary

#### 2021-08-21  
コンペ参加. 
<br>
<br>
<br>

#### 2021-08-22
kagglenb001-chaii-edaを作成.
<br>
<br>
<br>

#### 2021-08-24
kagglenb001-chaii-edaによるsubmit完了.






