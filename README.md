# PLaMo Pixel Benchmark — 1 APK

Pixel 7 Pro / Tensor G2 向け、PLaMo 3 Translate 0.8B の速度測定専用Androidアプリ。
古い翻訳アプリからのコピーではなく、独立した実装。

## ダウンロード・使い方
1. [GitHub Actions](https://github.com/linkalls/plamo-pixel-bench/actions/workflows/android-benchmark.yml) の成功した実行を開く。
2. Artifact `plamo-pixel-benchmark-unified` をダウンロード。
3. ZIP内の `plamo-pixel-benchmark.apk` をPixelにインストール。
4. 初回だけ PLaMo 3 Translate 0.8B のGGUFを選択し、モデルを取り込む。
5. Generic / Dotprod / KleidiAIをアプリ内で切り替えて同一文章を測定する。

## 3種類を1 APKへまとめた
- Generic: ARMv8 CPU
- Dotprod: ARMv8.2 + dotprod
- KleidiAI: ARMv8.2 + dotprod + KleidiAI

実際の速度差はPixel実機で確認する。llama.cpp はコミット
`ca89968a23a13e571a6b802a9891d22763153b61` に固定。

## 進捗・停止
処理の段階（モデル読込、トークン化、ウォームアップ、prefill、decode）と
生成済みトークン数、経過秒数を500ms間隔で画面に表示する。
中断ボタンと90秒の推論タイムアウトを追加した。
ただしモデル読込など一部の処理は即時には強制中断できない。
ウォームアップは2トークン、初期値は16トークンに短縮。

## ビルド
GitHub ActionsでNDK・SDKを準備し、Release最適化した各CPU JNIライブラリを
一度別々にコンパイル。それを1つのAndroid APKにまとめる。
Ubuntu本体にAndroid SDKを追加する必要はない。

注意：Actionsのビルド成功とPixel実機での推論動作確認は別。

## v0.3 長文・速度改善
- 生成上限を16〜2,048トークンから選択。初期値1,024トークン。
- 毎回の2トークン・ウォームアップを廃止。同一エンジンではGGUFをメモリにキャッシュして2回目以降のロードを省略。
- エンジン変更とモデル差し替えのときだけ別方式のキャッシュを解放してメモリ使用量を抑制。
- 最大コンテキスト4,096トークン。入力＋出力上限が収まるときは段落数にかかわらず全文を一度に翻訳。超過時だけトークン数ベースで分割し、できるだけ段落・文末で区切る。
- Prefillは512トークン単位で処理し、長文の512トークン制限を撤廃。生成中は約16トークンごとに推論速度を更新。
- 長文生成向けに安全停止を300秒へ変更。途中切れを検知するため出力上限到達フラグを結果に表示。

**注意:** 現時点ではPixelのCPU（Generic/Dotprod/KleidiAI）のみ。iPhoneのMetal GPUと同等速度は実証されていない。Tensor G2のGPU/Vulkan対応は別途実機で検証する必要がある。
