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
|-|-|
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


#### Papers
|name|url|status|comment|
|----|----|----|----|
|BERT Based Multilingual Machine Comprehension in English and Hindi|[URL](https://arxiv.org/pdf/2006.01432.pdf)|未読|-|
<br>


#### Blogs / Qiita / etc.
|name|url|status|comment|
|----|----|----|----|
|BERT入門|[URL](https://www.slideshare.net/matsukenbook/bert-217710964)|Done|Kaggle inference時 (=internetオフ時),<br>:hugs: transformersのメタファイルを取得する方針が参考になった|
<br>


#### Official Documentation or Tutorial
|name|url|status|comment|
|----|----|----|----|
<br>

#### StackOverflow
|name|url|status|comment|
|----|----|----|----|
|BERT tokenizer & model download|[URL](https://stackoverflow.com/questions/59701981/bert-tokenizer-model-download)|Done|:hugs: transformersのメタファイルをダウンロードする例.<br>ただし最近は格納先がHugging Face Platformに変更になった模様.|


#### GitHub
|name|url|status|comment|
|----|----|----|----|
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






