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
|-|-|-|0.010|
|6bb70140768d5dec90205db8b0568746124f5568|for loop incorporated in the function for memory efficiency|-|Error|
|8c807fec919722adf8eacb7822ddbfd3c0627487|sentence seperation deprecated|-|0.002|
|aa0b23e093ebbc085a56e354e764d95c3b31bb9f|replaced pipeline w/ torch-native way in the inference loop|-|0.005|
|9c381dbda9b2e142ab2ea1f32fcda596f4eb28d0|replaced pipeline w/ torch-native way in the inference loop|-|0.005|
|c85cec65d19ea00a6942e63934cd7f1288bc2460|pre-trained model (mBERT) w/ fine-tuning being done on this notebook|-|0.005|
|f4d2f7179bc1bc0c11bb03558ccf19a6a0ee3a00|indi-bert w/o fine-tuning|-|0.006|
|d421e45e0d00c77123ed61e7fef6b680aae9f19d|xlm-roberta-large-squad2 w/o fine-tuning|-|0.571|
|652b874625c0200035e39a79ab8144469b174a58|xlm-roberta-large-squad2 w/ fine-tuning (pipeline inference)|-|0.008|
|495c5de2a4b7582eccc3da4e1831572255e11004|xlm-roberta-large-squad2 w/ fine-tuning (naive torch inference)|-|Error|
|30955f14a9a0da8823b9e68df53443505ccacd96|copy & paste|-|0.792|
|5097e95f17ac09bb806088ba626ddb33f180ce5e|huggingface trainer API example reproduced faithfully|0.654|0.728|
|e316f40c44c52d204978b2c34a7b66320cb7687e|inference using fine-tuned model trained on Colab w/ 3 epochs|0.655|0.696|
|89a9c10ef578370e83f6ed05d86c24fa9fd4b262|--max_length 512 --doc_stride 80 --num_train_epochs 1|0.650|0.719|
|3c37a320af36ffe503d4839805c0cb7435601946|--drop_tamil_train true --CV 0.685 --LB 0.708|0.685|0.708|
|8c3d0c8487e6e7f99dc2277b7cc120259acb80a2|--max_seq_length 512 --doc_stride 80|-|0.775|
|7a15ddf4444426ad241d591b48610ec83311674a|--ensemble [2,1,1,1,1]|-|0.789|
|84a040e7d17bb3831a5f9ca5a7d9aa8081352043|--ensemble [1,2,1,1,1]|-|0.787|
|7495f69317823feeb8824da564a2d3bafad89479|riow1983/chaii-qa-5-fold-xlmroberta-torch-fit-7 used|-|0.770|
|1fae73a8fca470057b8650a6dbc8cd79fbbc2fc0|external data concatenated before 5-fold splitting|-|0.779|
|5f4c5aa7bb7bfe98f17a61f8ec1704d133a7987d|Models updated (tamil from train dropped)|-|0.771|
|356b83219e70108a9f6b47e850d640ac4c47e19b|Models updated (2 epochs for all folds)|-|0.783|
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
|localnb001-export-transformers|[URL](https://github.com/riow1983/Kaggle-chaii/blob/master/notebooks/localnb001-export-transformers.ipynb)|-|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)|Done|`bert-base-multilingual-cased`のファイルをKaggle Datasetとしてexport|
|kagglenb001-chaii-eda|[URL](https://www.kaggle.com/riow1983/kagglenb001-chaii-eda)|[localnb001-export-transformers](https://www.kaggle.com/riow1983/localnb001-export-transformers)<br>[indic-bert](https://www.kaggle.com/ajax0564/indicbert)|submission.csv|Done|`bert-base-multilingual-cased`による予測 (w/o fine-tuning)|
|localnb002-fine-tune|[URL](https://github.com/riow1983/Kaggle-chaii/blob/master/notebooks/localnb002-fine-tune.ipynb)|`../input/chaii-hindi-and-tamil-question-answering/train.csv`|localnb002|Done|`bert-base-multilingual-cased`のfine-tuning|
|l2knb001-fine-tune|[URL](https://www.kaggle.com/riow1983/l2knb001-fine-tune)|localnb001, localnb002|submission.csv|作成中|fine-tuned `bert-base-multilingual-cased`によるinference|
|kagglenb002-fine-tune|[URL](https://www.kaggle.com/riow1983/kagglenb002-fine-tune)|localnbf001, localnb002|submission.csv|作成中|kagglenb001をベースにしたfine-tuned `bert-base-multilingual-cased`によるinference|
|reproduction-of-0-792-notebook|[URL](https://www.kaggle.com/riow1983/reproduction-of-0-792-notebook)|[kishalmandal/5foldsroberta](https://www.kaggle.com/kishalmandal/5foldsroberta), [nguyenduongthanh/xlm-roberta-large-squad-v2](https://www.kaggle.com/nguyenduongthanh/xlm-roberta-large-squad-v2)|submission.csv|Done|[Reproduction of 0.792 notebook](https://www.kaggle.com/tkm2261/reproduction-of-0-792-notebook)のコピー|
|ChAII - EDA & Baseline|[URL](https://www.kaggle.com/riow1983/chaii-eda-baseline)|[thedrcat/hf-datasets](https://www.kaggle.com/thedrcat/hf-datasets), [nbroad/xlm-roberta-squad2](https://www.kaggle.com/nbroad/xlm-roberta-squad2)|chaii-bert-trained, chaii-qa, runs, submission.csv|Done|[ChAII - EDA & Baseline](https://www.kaggle.com/thedrcat/chaii-eda-baseline)のコピー|
|k2lnb001-chaii-eda-baseline-train|-|./input/hf-datasets, ./input/xlm-roberta-squad2|./notebooks/k2lnb001-chaii-eda-baseline-train/chaii-bert-trained, ./notebooks/k2lnb001-chaii-eda-baseline-train/chaii-qa, ./notebooks/k2lnb001-chaii-eda-baseline-train/runs|Done|[ChAII - EDA & Baseline](https://www.kaggle.com/riow1983/chaii-eda-baseline)からinference部分を除外したもの|
|kagglenb003-chaii-eda-baseline-inference|-|[thedrcat/hf-datasets](https://www.kaggle.com/thedrcat/hf-datasets), [nbroad/xlm-roberta-squad2](https://www.kaggle.com/nbroad/xlm-roberta-squad2), [riow1983/k2lnb001-chaii-eda-baseline-train](https://www.kaggle.com/riow1983/k2lnb001-chaii-eda-baseline-train)|submission.csv|Done|[ChAII - EDA & Baseline](https://www.kaggle.com/riow1983/chaii-eda-baseline)からtrain部分を除外したもの|
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
```python
# ip_addressごとの最頻出malware_typeを表示するSeriesを取得する方法

def md(s):
    c = Counter(s)
    return c.most_common(1)[0][0]

df.groupby('ip_address')['malware_type'].agg(md)
```
<br>


#### Papers
|name|url|status|comment|
|----|----|----|----|
|BERT Based Multilingual Machine Comprehension in English and Hindi|[URL](https://arxiv.org/pdf/2006.01432.pdf)|未読|-|
|Unsupervised Cross-lingual Representation Learning at Scale|[URL](https://arxiv.org/pdf/1911.02116.pdf)|未読|XLM-RoBERTaの論文|
|RETHINKING EMBEDDING COUPLING IN PRE-TRAINED LANGUAGE MODELS|[URL](https://openreview.net/pdf?id=xpFFI_NtgpW)|未読|mBERTの改良版"RemBERT"の論文.<br>XLM-RoBERTaを凌駕. [ディスカッション](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/267827)で取り上げられている.|
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
|HTML 特殊文字|[URL](https://qiita.com/inabe49/items/303afa114b0204da8a24)|Done|矢印やギリシア文字などの記法集|
|pandasでjsonlを扱いたい時|[URL](https://qiita.com/meshidenn/items/3ff72396fe85044bc74f)|Done|jsonlとは|
|エラーを出さずに最頻値を得たいとき|[URL](https://qiita.com/tmitani/items/bd77eb08f1da7c283fed)|Done|statistics.modeではエラーになる局面もcollections.Counterで解決可能|
<br>


#### Official Documentation or Tutorial
|name|url|status|comment|
|----|----|----|----|
|(:hugs:) Fine-tuning a pretrained model|[URL](https://huggingface.co/transformers/training.html)|参照中|Trainer APIや素のPyTorchによる:hugs: pretrained modelのfine-tuning実装例|
|(:hugs:) Fine-tuning with custom datasets|[URL](https://huggingface.co/transformers/custom_datasets.html#fine-tuning-with-custom-datasets)|参照中|IMDb(Sequence Classification), W-NUT(Token Classification = NER), SQuAD2.0(QA) それぞれについてのfine-tuning実装例|
|(PyTorch) DATASETS & DATALOADERS|[URL](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)|参照中|PyTorch Dataset, DataLoaderの公式実装解説|
|(PyTorch) Custom loss functions|[URL](https://discuss.pytorch.org/t/custom-loss-functions/29387)|Done|PyTorchでCustom loss functionを実装する方法|
|(:hugs:) BERT|[URL](https://huggingface.co/transformers/model_doc/bert.html?highlight=bertforquestionanswering#berttokenizer)|参照中|:hugs:BERTメインページ|
|(:hugs:) RoBERTa|[URL](https://huggingface.co/transformers/model_doc/roberta.html)|参照中|:hugs:RoBERTaメインページ|
|(:hugs:) XLM-RoBERTa|[URL](https://huggingface.co/transformers/model_doc/xlmroberta.html)|参照中|:hugs:XLM-RoBERTaメインページ|
|(:hugs:) Tokenizer|[URL](https://huggingface.co/transformers/main_classes/tokenizer.html#tokenizer)|参照中|:hugs:Tokenizerメインページ|
|(:hugs:) Preprocessing data|[URL](https://huggingface.co/transformers/preprocessing.html)|参照中|tokenizerにbatch sentence Aとbatch sentence Bを入力した場合に出力されるものを確認できる|
|(:hugs:) Extractive Question Answering|[URL](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)|Done|`pipeline`によるinferenceと`torch`によるinference両者の例がシンプルにまとまっている|
<br>

#### StackOverflow
|name|url|status|comment|
|----|----|----|----|
|BERT tokenizer & model download|[URL](https://stackoverflow.com/questions/59701981/bert-tokenizer-model-download)|Done|:hugs: transformersのメタファイルをダウンロードする例.<br>ただし最近は格納先がHugging Face Platformに変更になった模様.|
|python pandas: apply a function with arguments to a series|[URL](https://stackoverflow.com/questions/12182744/python-pandas-apply-a-function-with-arguments-to-a-series)|Done|apply対象の関数への引数の渡し方|
|Python: Find a substring in a string and returning the index of the substring|[URL](https://stackoverflow.com/questions/21842885/python-find-a-substring-in-a-string-and-returning-the-index-of-the-substring)|Done|`string`型のfindメソッドでできる|
|(pandas) Pandas drop duplicates on one column and keep only rows with the most frequent value in another column|[URL](https://stackoverflow.com/questions/63319148/pandas-drop-duplicates-on-one-column-and-keep-only-rows-with-the-most-frequent-v)|Done|出現回数が最頻のものを残してdrop duplicatesする方法|
|(:hugs:) Transformers v4.x: Convert slow tokenizer to fast tokenizer|[URL](https://stackoverflow.com/questions/65431837/transformers-v4-x-convert-slow-tokenizer-to-fast-tokenizer)|Done|`XLMRobertaTokenizer`を使用するためには`pip install transformers[sentencepiece]`としてやる必要あり|


#### GitHub
|name|url|status|comment|
|----|----|----|----|
|Kaggle-Coleridge-Initiative|[URL](https://github.com/riow1983/Kaggle-Coleridge-Initiative)|Done|Coleridgeコンペ時につけていたKagge日記|
|個人アクセストークンを使用する|[URL](https://docs.github.com/ja/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)|Done|GitHub連携にパスワード方式が使えなくなった2021年8月13日以降はこのトークン方式になる|
|(PyTorch) How to slice a BatchEncoding object into desired batch sizes?|[URL](https://github.com/huggingface/tokenizers/issues/577)|Done|`tokenizer`はbatchにしたtextを受け取れる仕様|
|IndicBERT|[URL](https://github.com/AI4Bharat/indic-bert)|未確認|[ai4bharat/indic-bert](https://huggingface.co/ai4bharat/indic-bert)にfine-tuneing cliが付いたもの(?)|
|Visualize your 🤗 Hugging Face data with 🏋‍♀️ Weights & Biases|[URL](https://github.com/wandb/examples/blob/master/colabs/huggingface/Visualize_your_Hugging_Face_data_with_Weights_%26_Biases.ipynb)|実行エラー|Colabで実行してみたがデータのロードでエラーが出る|
<br>

#### Hugging Face Platform
|name|url|status|comment|
|----|----|----|----|
|bert-base-multilingual-cased|[URL](https://huggingface.co/bert-base-multilingual-cased)|Done|`config.json`, `pytorch_model.bin`, `vocab.txt`を取得<br>kagglenb001-chaii-edaにて使用|
|ai4bharat/indic-bert|[URL](https://huggingface.co/ai4bharat/indic-bert)|Done|kagglenb001-chaii-edaにて使用|
|deepset/xlm-roberta-large-squad2|[URL](https://huggingface.co/deepset/xlm-roberta-large-squad2)|Done|kagglenb001-chaii-edaにて使用.<br>tokenizerには`XLMRobertaTokenizer`を使用していることが[こちら](https://public-mlflow.deepset.ai/#/experiments/124/runs/3a540e3f3ecf4dd98eae8fc6d457ff20)で確認できる<br>|
<br>

#### Kaggle Notebooks
|name|url|status|comment|
|----|----|----|----|
|coleridge_regex_electra|[URL](https://www.kaggle.com/nbroad/no-training-question-answering-model/data?scriptVersionId=66240356)|参考|Coleridgeコンペの47th solution.<br>transformers pipelineを使ったQA実装例が非常に分かりやすく, 本コンペでも参考とした.|
|chaii-QA: multi-lingual pretrained baseline|[URL](https://www.kaggle.com/nbroad/chaii-qa-multi-lingual-pretrained-baseline)|参照中|:hugs:製マルチリンガル系pre-trainedモデルをpipelineメソッドで実行のうえsubmitしてみた結果`xlm-roberta-large-squad2`が最良とのこと|
|Intro to Hugging Face datasets :hugs:|[URL](https://www.kaggle.com/nbroad/intro-to-hugging-face-datasets/notebook)|読了|入力データはtensorを格納した辞書({'input_ids':tensor(), 'attention_mask':tensor(), 'offset_mapping':tensor(), start_positions':tensor(), 'end_positions':tensor()})であれば良い.<br>:hugs: `datasets`はデータ加工にpandasを必要としないほど柔軟性がありそう.|
<br>


#### Kaggle Datasets
|name|url|status|comment|
|----|----|----|----|
|xlm roberta squad2|[URL](https://www.kaggle.com/nbroad/xlm-roberta-squad2)|Done|[deepset/xlm-roberta-large-squad2](deepset/xlm-roberta-large-squad2)をダウンロードしたもの|
<br>

#### Kaggle Discussion
|name|url|status|comment|
|----|----|----|----|
|Recipe for winning?|[URL](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/264917#1482290)|Done|学習データのラベルが適当(特にtタミル語)かつデータ量が少ない件について手動アノテーションの有利性に言及したもの.<br>もしくは学習データを一切使わず外部データだけでfine-tuneしたほうがいいという意見も.|
|Hindi & Tamil QA papers / datasets|[URL](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/264344)|未確認|使えそうな外部データやpre-trainedモデルの紹介|
|Useful Resources for the competition|[URL](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/264795)|Done|タイトルとは裏腹に学習データのラベルがいい加減だという指摘まとめとそこに起因する倫理的問題についてのディスカッションまとめが秀逸|
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
fine-tuned mBERTでsubmitするもzero-shotよりもLB悪化(0.010 &rArr; 0.002).<br>
inferenceをpipelineで実施していたが, [torch-nativeな方法](https://huggingface.co/transformers/task_summary.html#extractive-question-answering)も試したい.<br>
その次何するか:
- pipeline方式でtrainする際, HindiモデルとTamilモデルに分けてそれぞれでinferenceする &rArr; [issue#2](https://github.com/riow1983/Kaggle-chaii/issues/2)
- Jaccard関数などloss関数に組み込んでちゃんとtrainする &rArr; [issue#3](https://github.com/riow1983/Kaggle-chaii/issues/3)
- mBERTの枠組みで, Hindi Question - Tamil Answer など自動翻訳などを利用してdata augmentationしてtrainする
- Hindi - Tamil に特化したpre-trainedモデルが利用できる形でどこかに落ちてないか
- 諦めてホストのtutorial notebookの軍門に下るか &rArr; [issue#4](https://github.com/riow1983/Kaggle-chaii/issues/4)
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
|kagglenb001-chaii-eda|~~pipeline(コンテキストそのまま)~~ -<br>trainしていたと思いきやしていなかった|No|pipeline|0.005|
<br>
kagglenb001-chaii-edaについては２回submitしているがいずれもfine-tuningしていないものである. にも関わらずLBが異なっているのはモデルのランダム性が起因しているものと思われる. (pre-trainedモデルをロードする度にoutput(answer span)が変化することは確認した.)
<br>
<br>
<br>

#### 2021-10-05
> It seems like the data is generated by a model. especially for Tamil where even the answer text and start index are incorrect. So, what are we actually learning here?
[what are we learning?](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/discussion/267124)

とのことで特にTamil語のtrainの教師ラベルはいい加減らしい. fine-tuningでTamil trainを除外してみようか.
<br>
<br>
<br>

#### 2021-10-19
[notebooks/chaii-qa-5-fold-xlmroberta-torch-fit.ipynb](https://github.com/riow1983/Kaggle-chaii/blob/master/notebooks/chaii-qa-5-fold-xlmroberta-torch-fit.ipynb)でfoldごとのvalid lossを確認. fold 5でvalid lossが0になっているのはなぜか.<br>
[inference notebook](https://www.kaggle.com/riow1983/reproduction-of-0-792-notebook/notebook?scriptVersionId=77460400)でfold 5の結果を使用しないsubmitをしてみる. -> LB: 0.783 ＿|￣|○

```
--------------------------------------------------
FOLD: 1
--------------------------------------------------
----Validation Results Summary----
Epoch: [0] Valid Loss: 0.62316
0 Epoch, Best epoch was updated! Valid Loss: 0.62316

----Validation Results Summary----
Epoch: [1] Valid Loss: 0.71733


--------------------------------------------------
FOLD: 2
--------------------------------------------------
----Validation Results Summary----
Epoch: [0] Valid Loss: 0.57144
0 Epoch, Best epoch was updated! Valid Loss: 0.57144

----Validation Results Summary----
Epoch: [1] Valid Loss: 0.70074


--------------------------------------------------
FOLD: 3
--------------------------------------------------
----Validation Results Summary----
Epoch: [0] Valid Loss: 0.63953
0 Epoch, Best epoch was updated! Valid Loss: 0.63953

----Validation Results Summary----
Epoch: [1] Valid Loss: 0.77846


--------------------------------------------------
FOLD: 4
--------------------------------------------------
----Validation Results Summary----
Epoch: [0] Valid Loss: 0.62968
0 Epoch, Best epoch was updated! Valid Loss: 0.62968

----Validation Results Summary----
Epoch: [1] Valid Loss: 0.73707


--------------------------------------------------
FOLD: 5
--------------------------------------------------
----Validation Results Summary----
Epoch: [0] Valid Loss: 0.00000
0 Epoch, Best epoch was updated! Valid Loss: 0.00000

----Validation Results Summary----
Epoch: [1] Valid Loss: 0.00000
```

またepoch数は1で充分なのかもしれない. これももう一回やってみる?



