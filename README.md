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
