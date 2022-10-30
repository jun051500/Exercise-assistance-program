本プログラムは以下の環境で起動します。

１．仮想環境作成
　　Python　：　3．8.13

２．追加パッケージをインストール
　pip install mediapipe  
  pip install pygame  
  pip install streamlit  
  
３．実行に必要なファイル
　AI_training_2.py　←　実行ファイル
  training.mp4  ← 入力用ファイル
  seton.mp3　←　肩が下に正しく来た場合の音
  counter.mp3　←　正しく運動できた場合の音

４．実行方法
　mediapipeとpygameをインストールしたpython仮想環境下で
　>>　streamlit run AI_training_2.py
　を実行する



