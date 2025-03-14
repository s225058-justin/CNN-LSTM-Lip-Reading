# 会話支援のための機械学習を用いた読唇システムの開発
このリポジトリは、米子工業高等専門学校における卒業研究として開発した日本語読唇AIシステムのプログラムと結果を保存している。デモも利用可能であり、その手順は以下に記載されている。

## デモ
このリポジトリには、前処理とトレーニングに使用したコードが含まれており、`01_Dataset`と`02_Model`フォルダに格納されている。デモは`03_Webcam`フォルダに含まれている。デモを実行するためには、ウェブカメラが必要である。手順は以下の通りである。
1) `requirements.txt`に記載されている必要なライブラリをインストールする。
2) [このリンク](https://drive.google.com/drive/folders/1t1fRQfTaL1-XgGA1JSzuvLSXsitZ6Scj)から`shape_predictor_68_face_landmarks_GTX.dat`モデルをダウンロードし、`03_Webcam`内の`dlib_shape_predictor`フォルダに配置する。
3) 以下のコマンドで`Webcam_LivePredict.py`ファイルを実行する。
```bash
python3 Webcam_LivePredict.py
```
4) ウェブカメラのプレビューが表示されるまで待ち、発話する単語を選択する。音声は認識に不要であるため、無音で発話しても問題ない。
5) `Enter`キーを押し続け、選択した単語を発話した後に`Enter`キーを離す。
6) モデルが単語を予測するまで待つ。

## モデルのトレーニング
### データセット
九州工業大学の齊藤・張研究室が提供するSSSDデータセットを利用した。齊藤・張研究室のウェブサイトは[こちら](https://www.saitoh-lab.com/)でアクセスできる。データセットの機密性のため、トレーニングに使用したコードの一部は削除されており、データセットもこのリポジトリには含まれていない。そのため、このリポジトリを通じてモデルのトレーニングを再現することは想定されていない。

SSSDデータセットは、25の単語を72人の話者が10回ずつ発話した短い動画で構成されており、30fpsで撮影され、音声トラックは含まれていない。画像は参加者の口元に焦点を当てて切り取られている。25の単語は以下の通りである。

| #  | 発話内容   | #  | 発話内容       | #  | 発話内容       |
|----|----------|----|--------------|----|--------------|
| 0  | ぜろ      | 10 | ありがとう     | 20 | どういたしまして |
| 1  | いち      | 11 | いいえ         | 21 | はい          |
| 2  | に        | 12 | おはよう       | 22 | はじめまして   |
| 3  | さん      | 13 | おめでとう     | 23 | またね        |
| 4  | よん      | 14 | おやすみ       | 24 | もしもし      |
| 5  | ご        | 15 | ごめんなさい   |    |              |
| 6  | ろく      | 16 | こんにちわ     |    |              |
| 7  | なな      | 17 | こんばんわ     |    |              |
| 8  | はち      | 18 | さようなら     |    |              |
| 9  | きゅう    | 19 | すみません     |    |              |

### 前処理
トレーニングに使用するデータ量を削減し、トレーニング時間を短縮するため、各動画のフレーム数を25フレームに制限した。これを実現するため、25フレームを超える動画はダウンサンプリングされ、25フレーム未満の動画はパディングを行って25フレームに調整した。

17フレームしかない動画は25フレームにパディングされる。

![Padding](https://github.com/user-attachments/assets/22b43625-2adb-4382-a446-e2530b8fa0d7)

49フレームの動画は25フレームにダウンサンプリングされる。

![Sampling](https://github.com/user-attachments/assets/6d458d67-f9aa-4f9e-8eb1-267efc92dc53)

トレーニング中に冗長な情報が学習されるのを防ぐため、以下の手順でデータ量を削減した。
1) まず、画像をモノクロに変換した。
2) 各参加者の唇の部分を切り取り、顔や背景の詳細を除去した。
3)画像を300px × 300pxから32px × 32pxにリサイズした。
これらの手順により、データ量は元のサイズのわずか0.18%に削減されながらも、トレーニングに必要な関連データの大部分が保持された。

元の画像

![uncropped](https://github.com/user-attachments/assets/9bbd5fed-8642-4b42-99f6-980d53f82158)

トレーニングに使用した切り取られた白黒画像

![lipsonlybnw](https://github.com/user-attachments/assets/3aa91e21-fa81-405a-a7cf-8b69e3d79972)

画像の処理後、動画をテンソルに事前変換し、トレーニング中のモデルのハイパーパラメータ最適化時に各動画をロードする時間を節約した。これにより、トレーニング中は画像ではなく事前コンパイルされたテンソルをロードするため、大幅な時間短縮が実現された。

他の画像サイズやカラー、前処理効果も実験したが、最終的には白黒の唇画像を選択した。READMEを簡潔に保つため、他の実験結果は省略した。

### 合成データ
データの堅牢性を高めるため、合成データ生成も実装した。最初の5エポックでは、モデルに未修正のデータを学習させ、その後、60%の確率で動画を修正する変換関数を有効にした。動画に適用される変換は以下の通りである。

1. Blurring : 動画にランダムな強さでモザイク処理
2. Flipping : 動画を水平に反転
3. Perspective : 動画の観点をランダムな数字で変換
4. Rotation : ±15◦ 以内に動画を回転
5. Brightness : 動画の明るさを ±20%以内に変換
6. Contrast : 動画のコントラストを ±25%以内に変換
7. Saturation : 動画の鮮やかさを ±25%以内に変換
8. Gamma : 動画のガンマを ±25%以内に変換

変換関数がトリガーされると、各変換はそれぞれの確率で適用されるため、同じ動画に複数の変換が適用されることがある。これにより、トレーニングデータの多様性が大幅に増加した。各変換の確率は以下の表に示されている。

| 変換         | 変換関数確率 (%) | 変換確率 (%) | 合計確率 (%) |
|------------|--------------|----------|----------|
| Blurring   |      60      | 30       | 18       |
| Perspective |      60      | 30       | 18       |
| Rotation   |       60      |   50       | 30       |
| Flipping   |       60      |   50       | 30       |
| Brightness |       60      |   50       | 30       |
| Contrast   |       60      |   50       | 30       |
| Saturation |       60      |   50       | 30       |
| Gamma      |       60      |   50       | 30       |

これにより、5エポック以降の動画の18%にぼかしと遠近法の変換が適用され、30%の動画に回転、反転、明るさ、コントラスト、彩度、ガンマの変換が適用される。

### モデル
CNN-LSTMアーキテクチャを選択した。CNNはフレームから空間的特徴を抽出し、LSTMは動画の時間的パターンを捕捉する。また、CNNアーキテクチャには過学習を防ぐためにドロップアウト層が戦略的に使用された。モデルのアーキテクチャは以下の通りである。

![CNN_LSTM-model](https://github.com/user-attachments/assets/b05807a6-3214-4cb1-b8f4-aa44d74218ce)

モデルは50エポックにわたってトレーニングされ、トレーニングと検証のデータ分割は83:17である。

### 結果

50エポックのトレーニング後、モデルの精度は約80%に達した。エポック数を増やしても精度は向上せず、過学習が発生した。以下の画像はモデルのトレーニング結果を示している。検証データには12人の話者が含まれており、理想的にはすべての単語が120回予測されるべきである。

![Results](https://github.com/user-attachments/assets/5a49fcae-44f6-4696-a6ec-1679ddf3df4c)

さらに、モデルのデモ版を作成し、10人の学生から動画データを収集してモデルをテストした。各学生は各単語を10回ずつ発話し、学生ごとに合計250の動画を記録した。動画はトレーニングデータと同様に前処理され、トレーニングから保存された重みを使用してモデルに入力された。その結果、全体の精度は70.64%であり、短い単語ほどシステムが読み取りにくいことが判明した。特に、「ゴ」が最も真陽性が少なく、「イチ」が最も偽陽性が多かった。一方、「さようなら」「はじめまして」「こんばんは」などの長い単語は、偽陽性が最も少なく、真陽性の比率も良好であった。結果は以下の通りである。

![total_results1](https://github.com/user-attachments/assets/40e2145c-10ba-43d3-b658-ad90da92bc13)

さらに、10人の学生のうち、7人は日本語ネイティブスピーカーであり、3人は外国人学生であった。日本語ネイティブの学生の精度は74%であったのに対し、外国人学生の精度は63%であった。これは、モデルが日本語ネイティブスピーカーのデータでトレーニングされたため、外国人学生が必ずしも平均的な日本語話者と同じように単語を発音しないことが原因であると推測される。ただし、この仮説は検証されておらず、あくまで仮説である。

### 振り返り

このプロジェクトは、Pythonと機械学習に関する知識がゼロの状態から始め、指導教員の助言、教科書、オンライン記事、大規模言語モデル（LLM）を通じてPythonと機械学習の基礎、概念、ベストプラクティスを重点的に学んだ。しかし、この基礎学習に重点を置いたため、既存の研究を調査したり、ResNet50のような既知の事前学習済みモデルや、TransformerやAttentionメカニズムのような新興技術を活用して、より効率的で正確なモデルを作成するための時間が不足した。

今後は、最先端のアーキテクチャや事前学習済みモデル、高度な技術を組み込むことで、システムの性能と適応性を向上させたい。また、既存の研究をより詳細に調査し、革新的なアプローチを統合することを目指す。
