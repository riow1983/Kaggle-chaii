# Kaggle-chaii
![input file image](https://github.com/riow1983/Kaggle-chaii/blob/master/png/20210823.png)<br>
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
|6bb70140768d5dec90205db8b0568746124f5568|for loop incorporated in the function for memory efficiency|-|Error|
|8c807fec919722adf8eacb7822ddbfd3c0627487|sentence seperation deprecated|-|0.002|
|aa0b23e093ebbc085a56e354e764d95c3b31bb9f|replaced pipeline w/ torch-native way in the inference loop|-|0.005|
|9c381dbda9b2e142ab2ea1f32fcda596f4eb28d0|replaced pipeline w/ torch-native way in the inference loop|-|0.005|
|c85cec65d19ea00a6942e63934cd7f1288bc2460|pre-trained model (mBERT) w/ fine-tuning being done on this notebook|-|0.005|
<br>

## Late Submissions
|commitSHA|comment|Local CV|Private LB|Public LB|
|----|----|----|----|----|
<br>


## My Assets
[notebook命名規則]  
- kagglenb001-hoge.ipynb: Kaggle platform上で新規作成されたKaggle notebook (kernel).
- localnb001-hoge.ipynb: localで新規作成されたnotebook. 
- k2lnb001-hoge.ipynb: kagglenb001-hoge.ipynbをlocalにpullしlocalで変更を加えるもの.
- l2knb001-hoge.ipynb: localnb001-hoge.ipynbをKaggle platformにpushしたもの.

#### Code
作成したnotebook等の説明  
|name|url|input|output|status|comment|
|----|----|----|----|----|----|
|localnb001-export-transformers|-|-|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)|Done|`bert-base-multilingual-cased`のファイルをKaggle Datasetとしてexport|
|kagglenb001-chaii-eda|-|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)|submission.csv|Done|`bert-base-multilingual-cased`による予測 (w/o fine-tuning)|
|localnb002-fine-tune|[URL](https://github.com/riow1983/Kaggle-chaii/blob/master/notebooks/localnb002-fine-tune.ipynb)|`../input/chaii-hindi-and-tamil-question-answering/train.csv`|localnb002|Done|`bert-base-multilingual-cased`のfine-tuning|
|l2knb001-fine-tune|[URL](https://www.kaggle.com/riow1983/l2knb001-fine-tune)|localnb001, localnb002|submission.csv|作成中|fine-tuned `bert-base-multilingual-cased`によるinference|
|kagglenb002-fine-tune|[URL](https://www.kaggle.com/riow1983/kagglenb002-fine-tune)|localnbf001, localnb002|submission.csv|作成中|kagglenb001をベースにしたfine-tuned `bert-base-multilingual-cased`によるinference|
<br>





***
## 参考資料
#### Snipets
```python
# huggingface modelをPyTorch nn.Moduleで訓練した後save (& load) する方法:
# reference: https://github.com/riow1983/Kaggle-Coleridge-Initiative#2021-04-25
model = MyModel(num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
output_model = './models/model_hoge.pth'

# save
def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

save(model, optimizer)

# load
checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```
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
|Unicode block|[URL](https://en.wikipedia.org/wiki/Unicode_block)|Done|各言語文字のblock rangeが示されている|
|(正規表現) iHateRegex|[URL](https://ihateregex.io/expr/hyphen-word-break/)|Done|正規表現の動作確認がブラウザ上でできる|
|The Unicode Standard, Version 13.0 > Devanagari|[URL](https://unicode.org/charts/PDF/U0900.pdf)|Done|Devanagariのunicodeの解説書|
|Adapting BERT question answering for the medical domain|[URL](https://medium.com/analytics-vidhya/adapting-bert-question-answering-for-the-medical-domain-2085ada8ceb1)|Done|Dealing with a long contextセクションに512トークン長以内にcontextを分割する手法が言葉で説明されている|
|Question Answering with a fine-tuned BERT|[URL](https://towardsdatascience.com/question-answering-with-a-fine-tuned-bert-bc4dafd45626)|Done|fine-tunedということでfine-tuneする実装例ではなく, fine-tune済みのモデルの利用例になっていることに注意|
|【突然GitHubにpushできなくなった】 トークン生成で解決：The requested URL returned error: 403|[URL](https://zenn.dev/yuri0427/articles/9587ae6a578ee9)|Done|GitHub連携にパスワード方式が使えなくなった2021年8月13日以降はこのトークン方式になる|
|(:hugs:) ValueError: char_to_token() is not available when using Python based tokenizers|[URL](https://www.gitmemory.com/issue/huggingface/transformers/12201/862549850)|Done|:hugs: `char_to_token`メソッドを使うならfast tokenizerを使うべし|
|(pandas) pandasでDataFrameのセルにlistを代入する|[URL](https://linus-mk.hatenablog.com/entry/pandas_insert_list_into_cell)|Done|`object`型のSeriesに対して`.at`メソッドを使うべし|
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
|(pandas) Pandas drop duplicates on one column and keep only rows with the most frequent value in another column|[URL](https://stackoverflow.com/questions/63319148/pandas-drop-duplicates-on-one-column-and-keep-only-rows-with-the-most-frequent-v)|Done|出現回数が最頻のものを残してdrop duplicatesする方法|


#### GitHub
|name|url|status|comment|
|----|----|----|----|
|Kaggle-Coleridge-Initiative|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative)|Done|Coleridgeコンペ時につけていたKagge日記|
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
<br>
<br>
<br>

#### 2021-09-13
localnb002-fine-tuneのpreprocessingコードがようやくひと段落し, train開始.
<br>
<br>
<br>

#### 2021-09-14
fine-tuned mBERTでsubmitするもzero-shotよりもLB悪化(0.010 -> 0.002).<br>
inferenceをpipelineで実施していたが, [torch-nativeな方法](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)も試したい.<br>
その次何するか:
- pipeline方式でtrainする際, HindiモデルとTamilモデルに分けてそれぞれでinferenceする
- Jaccard関数などloss関数に組み込んでちゃんとtrainする
- mBERTの枠組みで, Hindi Question - Tamil Answer など自動翻訳などを利用してdata augmentationしてtrainする
- Hindi - Tamil に特化したpre-trainedモデルが利用できる形でどこかに落ちてないか
- 諦めてホストのtutorial notebookの軍門に下るか
<br>
<br>
<br>

#### 2021-09-15
>inferenceをpipelineで実施していたが, [torch-nativeな方法](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)も試したい.

について実行した. 結果は以下の通り:
|Notebook|fine-tuning方式|inference時コンテキスト分割してるか|inference方式|Public LB|
|----|----|----|----|----|
|kagglenb002-fine-tune|torch-native(コンテキスト分割)|No|pipeline|0.002|
|kagglenb002-fine-tune|torch-native(コンテキスト分割)|No|torch-native|0.005|
|l2knb001-fine-tune|torch-native(コンテキスト分割)|Yes|pipeline|Error|
|l2knb001-fine-tune|torch-native(コンテキスト分割)|Yes|torch-native|0.005|
|kagglenb001-chaii-eda|-|No|pipeline|0.010|
|kagglenb001-chaii-eda|pipeline(コンテキストそのまま)|No|pipeline|0.005|





