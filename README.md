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

声帯の潰れた俺へ、このリポジトリには俺の声で学習させた音声合成モデルの設定ファイルが入ってる。モデルファイル（.pthファイル）と音声サンプルは容量が大きすぎるから、以下のリンクからダウンロードして

**モデルファイルダウンロード**: [MyVoice_MYCOEIROINK.tar.gz](https://github.com/0rnot/MyVoice/releases/download/v1.0/MyVoice_MYCOEIROINK.tar.gz) (約662MB)

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
- **GPU**: NVIDIA GPU（**VRAM 8GB以上推奨、Tensorコア搭載**）
- **ストレージ**: 20GB以上の空き容量
- **メモリ**: 32GB以上推奨

### 詳細な環境構築手順

1. **ESPnetリポジトリのクローン**
   ```bash
   git clone https://github.com/espnet/espnet
   cd espnet
   pip install -e .
   ```

2. **CUDAとPyTorchの確認**
   ```bash
   # CUDA Version確認
   nvidia-smi
   
   # PyTorch CUDA対応版インストール（CUDA 11.8の場合）
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # 動作確認
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **COEIROINKレシピディレクトリの準備**
   ```bash
   # ESPnetのCOEIROINKレシピディレクトリに移動
   cd espnet/egs2/mycoe/tts1
   
   # 環境設定
   . ./path.sh
   
   # 必要な追加ライブラリのインストール
   pip install jaconv pyopenjtalk unidic-lite mecab-python3
   pip install librosa soundfile scipy
   pip install parallel-wavegan  # HiFi-GAN用
   ```

4. **日本語言語モデルの準備**
   ```bash
   # MeCab辞書のダウンロード
   python -m unidic download
   
   # pyopenjtalkの辞書確認
   python -c "import pyopenjtalk; pyopenjtalk.g2p('これはテストです')"
   ```

### データ準備（超重要）

1. **音声収録の準備**
   ```bash
   # 収録用ディレクトリ作成
   mkdir -p ~/mycoe_work/raw_audio
   cd ~/mycoe_work/raw_audio
   ```

2. **音声データの詳細仕様**
   - **フォーマット**: WAV、44.1kHz、16bit、モノラル
   - **ファイル数**: 100〜500ファイル（最低50分、推奨60分以上）
   - **1ファイルあたり**: 5〜15秒（8秒程度が理想）
   - **内容**: 感情の起伏が多様な文章を読む(語り口調になりすぎると変になる)
   - **品質**: 無音部分、ノイズ、リバーブを徹底的に除去

3. **音声ファイルの命名規則**
   ```
   MYCOE001.wav  # 通し番号で命名
   MYCOE002.wav
   MYCOE003.wav
   ...
   MYCOE500.wav
   ```

4. **テキストファイルの作成(例)**
   ```bash
   # transcript.txtを作成（タブ区切り）
   # ファイル名<TAB>読み上げテキスト
   MYCOE001	こんにちは、今日はいい天気ですね。
   MYCOE002	音声合成の学習用データを作成しています。
   MYCOE003	この音声モデルは高品質な合成音声を生成します。
   ```

5. **ESPnet用データ構造の構築**
   ```bash
   cd ~/mycoe_work
   
   # データディレクトリ構造を作成
   mkdir -p data/{train,dev}
   
   # 音声ファイルリスト作成（wav.scp）
   find raw_audio -name "*.wav" | head -400 | \
   awk -F'/' '{printf "%s\t%s\n", $NF, $0}' | \
   sed 's/.wav\t/\t/' > data/train/wav.scp
   
   find raw_audio -name "*.wav" | tail -100 | \
   awk -F'/' '{printf "%s\t%s\n", $NF, $0}' | \
   sed 's/.wav\t/\t/' > data/dev/wav.scp
   
   # テキストファイル作成
   head -400 transcript.txt > data/train/text
   tail -100 transcript.txt > data/dev/text
   
   #話者IDファイル作成
   awk '{print $1 " MYCOE"}' data/train/text > data/train/utt2spk
   awk '{print $1 " MYCOE"}' data/dev/text > data/dev/utt2spk
   
   # 話者-発話対応ファイル作成
   utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
   utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
   ```

### 実際の学習実行手順

1. **ESPnetレシピの実行準備**
   ```bash
   cd espnet/egs2/mycoe/tts1
   
   # データを適切な場所にコピー
   cp -r ~/mycoe_work/data ./
   
   # 設定ファイルの確認
   ls conf/tuning/  # finetune_vits.yamlがあることを確認
   ```

2. **学習設定ファイル（conf/tuning/finetune_vits.yaml）のカスタマイズ**
   ```yaml
   # 重要なパラメータ
   batch_bins: 2000000    # バッチサイズ（VRAM使用量に影響するから適宜変更）
   accum_grad: 1          # 勾配蓄積回数
   max_epoch: 100         # エポック数
   patience: 10           # 早期停止の設定
   
   # VITS固有の設定
   tts_conf:
     generator_params:
       hidden_channels: 192
       segment_size: 32
       text_encoder_attention_heads: 2
     
   # 最適化設定
   optim: adamw
   optim_conf:
     lr: 0.0002          # 学習率(ここ大事)
     betas: [0.8, 0.99]
     eps: 1.0e-09
     weight_decay: 0.01
   ```

3. **段階的学習実行（run.shの詳細）**
   ```bash
   # Stage 1-2: データ形式変換とフィルタリング
   ./run.sh --stage 1 --stop_stage 2 \
     --ngpu 1 \
     --fs 44100 \
     --n_fft 2048 \
     --n_shift 512 \
     --dumpdir dump/44k
   
   # Stage 3: 統計情報の計算
   ./run.sh --stage 3 --stop_stage 3 \
     --ngpu 1 \
     --train_config conf/tuning/finetune_vits.yaml
   
   # Stage 4-5: モデル学習（最重要！！）
   ./run.sh --stage 4 --stop_stage 5 \
     --ngpu 1 \
     --train_config conf/tuning/finetune_vits.yaml \
     --tag "mycoe_vits_$(date +%Y%m%d)" \
     --tts_exp exp/tts_train_mycoe
   ```

4. **学習監視用コマンド**
   ```bash
   # TensorBoard起動（別ターミナル）
   tensorboard --logdir exp/tts_train_mycoe --port 6006
   
   # 学習ログの確認
   tail -f exp/tts_train_mycoe/train.log
   
   # GPU使用状況の監視
   watch -n 1 nvidia-smi
   ```

5. **学習完了後の推論テスト**
   ```bash
   # Stage 6: 推論実行
   ./run.sh --stage 6 --stop_stage 6 \
     --ngpu 1 \
     --tts_exp exp/tts_train_mycoe \
     --inference_model "100epoch.pth"
   
   # テスト用音声生成
   echo "こんにちは、音声合成のテストです。" | \
   espnet2-tts-inference \
     --model_file exp/tts_train_mycoe/100epoch.pth \
     --output_dir ./output_test
   ```

### COEIROINK用ファイルの準備（最終段階）

1. **学習済みモデルの変換と配置**
   ```bash
   # 新しいUUIDを生成
   NEW_UUID=$(python3 -c "import uuid; print(str(uuid.uuid4()))")
   echo "Generated UUID: $NEW_UUID"
   
   # COEIROINK用ディレクトリ構造を作成
   mkdir -p coeiroink_model/${NEW_UUID}/model/224757969
   mkdir -p coeiroink_model/${NEW_UUID}/icons
   mkdir -p coeiroink_model/${NEW_UUID}/voice_samples
   
   # 学習済みモデルをコピー
   cp exp/tts_train_mycoe/100epoch.pth coeiroink_model/${NEW_UUID}/model/224757969/
   cp exp/tts_train_mycoe/config.yaml coeiroink_model/${NEW_UUID}/model/224757969/
   ```

2. **メタデータファイル（metas.json）の作成**
   ```bash
   cat > coeiroink_model/${NEW_UUID}/metas.json << EOF
   {
       "speakerName": "MYCOEIROINK",
       "speakerUuid": "${NEW_UUID}",
       "styles": [
           {
               "styleName": "のーまる",
               "styleId": 224757969
           }
       ]
   }
   EOF
   ```

3. **サンプル音声の生成**
   ```bash
   # テスト文章でサンプル音声を3つ生成
   mkdir -p temp_samples
   
   echo "こんにちは、音声合成のテストです" | \
   espnet2-tts-inference \
     --model_file exp/tts_train_mycoe/100epoch.pth \
     --output_dir temp_samples/sample1
   
   echo "これは俺の声を学習したモデルです" | \
   espnet2-tts-inference \
     --model_file exp/tts_train_mycoe/100epoch.pth \
     --output_dir temp_samples/sample2
   
   echo "音質の確認用サンプル音声です" | \
   espnet2-tts-inference \
     --model_file exp/tts_train_mycoe/100epoch.pth \
     --output_dir temp_samples/sample3
   
   # 生成された音声をリネームして配置
   cp temp_samples/sample1/norm/speech.wav coeiroink_model/${NEW_UUID}/voice_samples/224757969_001.wav
   cp temp_samples/sample2/norm/speech.wav coeiroink_model/${NEW_UUID}/voice_samples/224757969_002.wav
   cp temp_samples/sample3/norm/speech.wav coeiroink_model/${NEW_UUID}/voice_samples/224757969_003.wav
   ```

4. **アイコンとポートレート画像の配置**
   ```bash
   # デフォルト画像をコピー
   # 256x256のアイコン画像
   cp /path/to/your/icon.png coeiroink_model/${NEW_UUID}/icons/224757969.png
   
   # 512x512のポートレート画像
   cp /path/to/your/portrait.png coeiroink_model/${NEW_UUID}/portrait.png
   ```

5. **最終的なパッケージング**
   ```bash
   # 完成したモデルをtar.gzで圧縮
   tar -czf "MYCOEIROINK_${NEW_UUID}_$(date +%Y%m%d).tar.gz" coeiroink_model/${NEW_UUID}
   
   # ファイルサイズ確認
   ls -lh MYCOEIROINK_${NEW_UUID}_$(date +%Y%m%d).tar.gz
   
   echo "完成しました＾＾COEIROINKのspeaker_infoディレクトリに配置して"
   echo "UUID: ${NEW_UUID}"
   ```

### 学習のコツと重要ポイント

- **データ品質が全て**: ノイズ除去を徹底的に（Audacityでスペクトログラム確認推奨）
- **収録環境**: 反響の少ない部屋、マイクとの距離一定、背景ノイズゼロ
- **データ量**: 最低50分、理想は90分以上（多ければ多いほどいい）
- **エポック数**: validation lossがプラトーになったら早期停止（通常80-120エポック）
- **ハードウェア**: RTX 3080以上が推奨、VRAM不足なら batch_bins を半分に
- **収録内容**: 日常会話、疑問文、感嘆文をバランスよく
- **ピッチとトーン**: 普段の話し方を意識、感情を込めすぎない

### 詳細なトラブルシューティング

#### メモリ関連エラー
```bash
# CUDA out of memory
# conf/tuning/finetune_vits.yaml の batch_bins を調整
batch_bins: 1000000  # デフォルトから半分に

# CPU memory不足
# swap領域を増やす
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 音質問題
```bash
# 学習途中の音声確認
# Stage 6を実行して中間結果をチェック
./run.sh --stage 6 --stop_stage 6 \
  --tts_exp exp/tts_train_mycoe \
  --inference_model "50epoch.pth"

# スペクトログラム確認
python -c "
import librosa
import matplotlib.pyplot as plt
y, sr = librosa.load('生成された音声.wav')
D = librosa.stft(y)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(abs(D)), 
                        y_axis='log', x_axis='time')
plt.colorbar()
plt.savefig('spectrogram.png')
"
```

#### pyopenjtalk関連エラー
```bash
# 辞書が見つからない場合
python -m unidic download
export MECAB_PATH=/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd

# 音素変換テスト
python -c "
import pyopenjtalk
text = 'これはテストです'
phonemes = pyopenjtalk.g2p(text)
print(f'Text: {text}')
print(f'Phonemes: {phonemes}')
"
```

#### 学習が進まない場合
```bash
# 学習率の調整
# conf/tuning/finetune_vits.yaml
optim_conf:
  lr: 0.0001  # デフォルトより小さく

# warmupステップの追加
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 4000
```

#### 推論時のエラー
```bash
# モデル互換性チェック
python -c "
import torch
model = torch.load('exp/tts_train_mycoe/100epoch.pth')
print('Model keys:', list(model.keys()))
print('Config available:', 'config' in model)
"
```

### 学習時間の目安
- **データ準備**: 2-4時間（音声収録・編集含む）
- **前処理**: 30分-1時間
- **学習**: 24-36時間（GTX1660Sで100エポック）
- **テスト・調整**: 2-4時間

**合計**: 2-3日の作業（いいGPU買え）
# Test update Mon Jul 28 20:05:56 JST 2025
