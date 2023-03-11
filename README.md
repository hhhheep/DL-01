# DL-01

這是是一份深度學習作業  利用 簡單的特徵提取和機器學習算法 分類50類圖片

環境：
opencv    4.6.0.66
sklearn   0.24.2
lightGBM  3.3.5
numpy     1.21.5 
pandas   1.4.4  

執行train即可，read_ 是讀取本地端的圖片 top_1_top_5 是估計效能函數

我使用的 三個模型 分別是 SGD perceptron lighGBM
我采取增量訓練的方式 去訓練模型，分批讀取文件進行處理后放入模型中訓練，用top_1_accuracy和top_5_accuracy 去測試模型的效果

