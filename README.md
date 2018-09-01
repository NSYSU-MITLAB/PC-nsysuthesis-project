# cnn
這是論文「深度神經網路複雜度與解析度對十類圖片辨識率分析」的原始碼，論文主要探討神經網路的複雜度與圖片解析度對於辨識率的影響，輸入資料使用的三個資料集分別是：MNIST、FashionMNIST及NotMNIST。實驗的部分，根據網路複雜度由高到低設計了五種不同的分類模型，其中複雜度最高的模型以全卷積神經網路為基礎組合了各種卷積神經網路的架構。在MNIST、FashionMNIST及NotMNIST資料集上得到最高99.75%，94.87%及97.57%的辨識率，在三個資料集上的辨識率都比全卷積神經網路有所提升。三個資料集有相同的解析度，對於每個資料集的訓練資料與測試資料，本研究使用線性內差的方法額外產生了四種不同解析度的圖片，接著將這些訓練資料輸入分類模型進行訓練，訓練完成之後對相同解析度的測試資料進行辨識。實驗結果發現高複雜度的分類模型對高解析度及低解析度的圖片有較高的辨識率，但是隨著解析度變低，辨識率也會降低；然而，低複雜度的模型對於低解析度的圖片辨識率有些微增加的現象，因此我們推論較為簡單的分類模型在低解析度的圖片上可能有較好的表現。
## Requirements
- python3.4
- tensorflow1.0.1
- keras2.0.3
- scipy0.19.0
- opencv3.3.0
## 執行方式
```
run $ bash cl_cnn_scipe.sh
```
## 結果
- MNIST

| 解析度\模型 | 模型一 | 模型二 |
| ------ | ----------- | ------ |
| data   | path to data files to supply the data that will be passed into templates. | 123 |
| engine | engine to be used for processing templates. Handlebars is the default. | 123 |
| ext    | extension to be used for dest files. | 123 |
