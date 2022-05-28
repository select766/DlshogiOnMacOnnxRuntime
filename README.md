# DlshogiOnMacOnnxRuntime
Macのonnxruntime+Core MLでdlshogiの評価関数モデルを動作させる最低限のテストプログラム

# 準備
onnxruntimeの[リリース](https://github.com/microsoft/onnxruntime/releases/tag/v1.11.1)からバイナリ(`onnxruntime-osx-universal2-1.11.1.tgz`)・ソースコード(`v1.11.1.tar.gz`)をダウンロードして`onnxruntime`内に解凍。

ツリー構成

```
onnxruntime
- onnxruntime-1.11.1
  - README.md
- onnxruntime-osx-universal2-1.11.1
  - README.md
```

```
sudo cp -a onnxruntime/onnxruntime-osx-universal2-1.11.1/lib/libonnxruntime.1.11.1.dylib /usr/local/lib
```

ライブラリを使う際にセキュリティ警告が出る場合がある。Mac設定の「セキュリティとプライバシー」の一般タブで許可する。

# ビルド

```
make
```

# ベンチマーク

バッチサイズ1、CPUバックエンド、10秒間計測する場合

```
./bench 1 cpu 10
```

バックエンドは、`cpu`: CPU、 `coreml`: Core ML。Core MLでは理論上GPU/Neural Engineが使用可能であるが、macOS 12.4時点においてdlshogiモデルではこれらを使用しない模様。ただし、CPU上での動作のアルゴリズムがonnxruntimeのCPUバックエンドと異なるようで、Core MLを利用した方が20%程度高速となる。

# モデル・テストケースの作成方法

「強い将棋ソフトの創りかた」サンプルコードに従い、15ブロック224チャンネル(`resnet15x224_swish`)のモデルを学習後、以下のリポジトリのjupyter notebookを用いてonnxモデルファイルおよびテストケースを生成している。

https://github.com/select766/dlshogi-model-on-coreml/tree/master/colab

