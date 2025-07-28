## 概要

俺の声帯が潰れたとき用のバックアップ

## ディレクトリ構造

```
b220da3c-4c23-11f0-99ea-0242ac1c000c/
├── LICENSE.txt          # ライセンス情報
├── policy.md           # 利用規約・免責事項
├── metas.json          # スピーカー情報
├── portrait.png        # ポートレート画像
├── icons/
│   └── 224757969.png   # アイコン画像
├── model/
│   └── 224757969/
│       ├── 100epoch.pth    # 学習済みモデル
│       └── config.yaml     # モデル設定
└── voice_samples/
    ├── 224757969_001.wav   # サンプル音声1
    ├── 224757969_002.wav   # サンプル音声2
    └── 224757969_003.wav   # サンプル音声3
```

## 使用方法

声帯の潰れた俺へ
このリポジトリには俺の声で学習させた音声合成モデルの設定ファイルが入ってる
モデルファイル（.pthファイル）と音声サンプルは容量が大きすぎるから、以下のリンクからダウンロードしろ

**モデルファイルダウンロード**: [MyVoice_MYCOEIROINK.tar.gz](./MyVoice_MYCOEIROINK.tar.gz) (約662MB)

COEIROINKで俺の声を使う手順↓

### 1. COEIROINKのインストール

1. [COEIROINK公式サイト](https://coeiroink.com/)から最新版をダウンロード(GPU版)
2. インストーラーを実行してインストール
3. COEIROINKを一度起動しておく

### 2. 音声モデルの導入

1. このリポジトリをクローンするかZIPでダウンロード
   ```bash
   git clone https://github.com/0rnot/MyVoice.git
   ```
2. モデルファイル（MyVoice_MYCOEIROINK.tar.gz）をダウンロードして展開
   ```bash
   tar -xzf MyVoice_MYCOEIROINK.tar.gz
   ```
3. 展開した`MyVoice_MYCOEIROINK`フォルダの名前を`b220da3c-4c23-11f0-99ea-0242ac1c000c`に変更
4. そのフォルダをCOEIROINKのspeaker_infoディレクトリにぶち込む：
   ```
   [COEIROINKインストール先]/speaker_info/b220da3c-4c23-11f0-99ea-0242ac1c000c/
   ```
5. COEIROINKを再起動
6. 音声選択に「MYCOEIROINK」が追加されてるはず

### 3. 音声合成の作成

1. COEIROINKを起動
2. 音声選択で「MYCOEIROINK」を選ぶ
3. スタイルは「のーまる」しかないのでそれを選択
4. 適当にテキスト入力して音声合成ボタンをポチる
5. 俺の声で喋ってくれる

## 音声サンプル

ダウンロードしたファイルの`voice_samples/`フォルダに俺の声のサンプルが入ってるから、導入前に確認(無音かもだけどそれはそれで別に大丈夫)

## 技術仕様

- **モデルタイプ**: VITS
- **サンプリングレート**: 44,100 Hz
- **学習エポック数**: 100
- **音声提供者**: shirowanisan (Copyright (c) 2022)


---

## 声変わりしたときにもう一回作る方法↓

### 必要な環境
- **OS**: Ubuntu 20.04 LTS または Windows 10/11（WSL2推奨）
- **Python**: 3.8以上
- **GPU**: NVIDIA GPU（VRAM 8GB以上推奨）
- **ストレージ**: 20GB以上の空き容量

### 環境構築

1. **Anacondaのインストール**
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
   bash Anaconda3-2023.03-Linux-x86_64.sh
   source ~/.bashrc
   ```

2. **仮想環境の作成**
   ```bash
   conda create -n coeiroink python=3.8
   conda activate coeiroink
   ```

3. **必要なライブラリのインストール**
   ```bash
   # PyTorch（CUDA対応版）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # ESPnet2
   pip install espnet
   
   # その他の依存関係
   pip install librosa soundfile jaconv pyopenjtalk unidic-lite
   ```

### データ準備

1. **音声データの準備**
   - 高品質な音声ファイル（WAV、48kHz、16bit推奨）
   - 100～500ファイル(1ファイル8秒くらい)の音声データを用意(サンプルは多ければ多いほどいい)
   - ノイズを除去

2. **テキストデータの準備**
   - 音声に対応するテキストファイルを作成
   - 読み方が統一されていることを確認

3. **ディレクトリ構造の作成**
   ```
   work_dir/
   ├── raw_data/
   │   ├── wav/
   │   │   ├── sample001.wav
   │   │   └── sample002.wav
   │   └── text
   └── dump/
   ```

### 学習プロセス

1. **データの前処理**
   ```bash
   # 音声ファイルの変換（44.1kHz, モノラル）
   python scripts/audio_convert.py \
     --input_dir raw_data/wav \
     --output_dir dump/44k/raw \
     --sample_rate 44100
   ```

2. **音素の抽出**
   ```bash
   # テキストから音素への変換
   python scripts/make_phoneme.py \
     --text_file raw_data/text \
     --output_dir dump/44k/raw
   ```

3. **特徴量の抽出**
   ```bash
   # スペクトログラムの計算
   espnet2-tts-preprocess \
     --train_data_path_and_name_and_type dump/44k/raw/tr_no_dev/wav.scp,speech,sound \
     --train_data_path_and_name_and_type dump/44k/raw/tr_no_dev/text,text,text \
     --valid_data_path_and_name_and_type dump/44k/raw/dev/wav.scp,speech,sound \
     --valid_data_path_and_name_and_type dump/44k/raw/dev/text,text,text \
     --output_dir exp/tts_stats_raw_linear_spectrogram
   ```

4. **モデルの学習**
   ```bash
   espnet2-tts-train \
     --config conf/tuning/finetune_vits.yaml \
     --train_data_path_and_name_and_type dump/44k/raw/tr_no_dev/text,text,text \
     --train_data_path_and_name_and_type dump/44k/raw/tr_no_dev/wav.scp,speech,sound \
     --valid_data_path_and_name_and_type dump/44k/raw/dev/text,text,text \
     --valid_data_path_and_name_and_type dump/44k/raw/dev/wav.scp,speech,sound \
     --output_dir exp/tts_mycoe_model \
     --ngpu 1 \
     --max_epoch 100
   ```

### COEIROINK用ファイルの準備

1. **UUIDの生成**
   ```python
   import uuid
   speaker_uuid = str(uuid.uuid4())
   print(speaker_uuid)  # 例: b220da3c-4c23-11f0-99ea-0242ac1c000c
   ```

2. **metas.jsonの作成**
   ```json
   {
       "speakerName": "MYCOEIROINK",
       "speakerUuid": "生成したUUID",
       "styles": [
           {
               "styleName": "のーまる",
               "styleId": 224757969
           }
       ]
   }
   ```

3. **ディレクトリ構造の構築**
   ```bash
   mkdir -p speaker_info/${speaker_uuid}/model/224757969
   mkdir -p speaker_info/${speaker_uuid}/icons
   mkdir -p speaker_info/${speaker_uuid}/voice_samples
   
   # 学習済みモデルのコピー
   cp exp/tts_mycoe_model/100epoch.pth speaker_info/${speaker_uuid}/model/224757969/
   cp exp/tts_mycoe_model/config.yaml speaker_info/${speaker_uuid}/model/224757969/
   ```

4. **その他のファイル**
   - `LICENSE.txt`: ライセンス情報
   - `policy.md`: 利用規約
   - `portrait.png`: ポートレート画像（推奨サイズ: 512x512px）←デフォで良い
   - `icons/224757969.png`: アイコン画像（推奨サイズ: 256x256px）←デフォで良い
   - `voice_samples/`: サンプル音声ファイル

### 学習のコツ

- **データ品質**: ノイズをできるだけ減らす(特にINとOUTのノイズ)
- **データ量**: 最低30分、推奨60分以上の音声データ
- **エポック数**: 過学習を避けるため、検証損失を監視しながら調整する
- **ハードウェア**: GPUを使わないと無限の時間がかかる

### トラブルシューティング

- **CUDA out of memory**: バッチサイズを減らす
- **音質が悪い**: 学習データの品質確認、エポック数調整
- **発音がおかしい**: 音素アライメントの確認
