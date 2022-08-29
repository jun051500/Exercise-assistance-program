# Exercise-assistance-program
Mediapipeを利用した運動補助プログラム
本プログラムは以下の環境で起動します。

１．仮想環境作成
　　Python　：　3．8.13

２．追加パッケージをインストール
　pip install mediapipe
  pip install pygame

３．実行に必要なファイル
　pose_movie.py　←　実行ファイル
  training.mp4  ← 入力用ファイル
  seton.mp3　←　肩が下に正しく来た場合の音
  counter.mp3　←　正しく運動できた場合の音
  result_video.mp4 ←　プログラム実行結果の動画ファイル

４．実行方法
　mediapipeとpygameをインストールしたpython仮想環境下で
　>>　python  pose_movie.py
　を実行する
