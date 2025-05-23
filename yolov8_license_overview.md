# YOLOv8 ライセンス概要  
*(最終更新 2025-05-14)*

Ultralytics が公開する **YOLOv8 (ultralytics/ultralytics)** は、  
ソースコード・学習済みモデルともに **強いコピーレフト** に分類される **GNU GPL 系ライセンス** で配布されています。  
閉源プロダクトに組み込む場合は **商用ライセンスの取得が必須** となる点に注意してください。

| 項目 | デフォルトライセンス | 商用利用 (無償版) | 主な義務 |
|------|--------------------|------------------|----------|
| ソースコード (`ultralytics` リポジトリ) | **GNU GPL v3** (※1) | ⚠️ ソース公開必須 | 改変含む派生物のソース開示、同一 GPL ライセンス維持 |
| 推論スクリプト例 (`detect.py`, `train.py` 等) | 同上 | 同上 | 同上 |
| 公式学習済みモデル (`yolov8n.pt`, `.onnx` 等) | **GPL v3** | 同上 | 再配布時に GPL 条文添付 |
| データセット (COCO 等) | 第三者ライセンス | ケースごと | 各データセットの LICENSE に従う |
| 有償 OEM / Enterprise ライセンス | **Ultralytics Commercial** | ✅ 完全商用可 | コピーレフト解除、サポート付与 |

> ※1 2025 年 5 月時点、YOLOv8 を含む `ultralytics` リポジトリは **GPLv3**（以前は AGPLv3）。  
> 公式リポジトリの `LICENSE` ファイルで最新情報を必ず確認してください。

---

## 1. GPL v3 の基本義務
| トリガー | 要求される対応 |
|----------|---------------|
| **ソース再配布・頒布**<br>(exe, wheel, firmware, Docker) | 派生物の *完全なソースコード* を提供し、GPL v3 ライセンスを適用 |
| **SaaS / API 提供** | GPL は SaaS を直接強制しないが、<br>独自改変を非公開で提供する場合は **「ユーザーにバイナリのみ提供」＝ライセンス準拠外** とみなされるグレー領域。Ultralytics は SaaS でも OEM ライセンスを推奨 |
| **バージョン/著作権表記の削除** | 不可。元 LICENSE と著作権表記を保持すること |

---

## 2. 商用ライセンス (OEM / Pro / Enterprise)
Ultralytics は以下のプランで **GPL コピーレフトを解除** した商用ライセンスを提供しています（概要）。

| ランク | 主な特典 | 想定ユースケース |
|-------|---------|----------------|
| **Pro (個人・小規模)** | SaaS/閉源アプリに組込可、weights 再頒布可 | スタートアップの PoC、スマホ App |
| **Enterprise / OEM** | ソースコード商用ライセンス、サポート、<br>大量デバイス配布権 | 組込機器メーカー、監視カメラシステム |

> 詳細価格・条件は **Ultralytics Sales (sales@ultralytics.com)** へ直接問い合わせ。

---

## 3. 実務シナリオ別チェック

| シナリオ | GPL 版での可否 | 留意点 |
|----------|---------------|--------|
| **社内 PoC (配布なし)** | ✅ 可能 | 社外配布が無ければ GPL のソース公開義務は発生しない |
| **オンプレ製品にバンドルし販売** | ❌ 公開必須 | 製品ソースを全公開するか OEM ライセンス購入 |
| **クラウド API として外部提供** | ⚠️ グレー | GPL では義務なしだが Ultralytics は商用ライセンスを要求 |
| **オープンソース派生プロジェクト** | ✅ 可能 | GPLv3 を維持し派生物ソースを公開 |

---

## 4. 互換・競合ライセンス例

| ライブラリ | ライセンス | YOLOv8 と組み合わせ可否 |
|-----------|-----------|------------------------|
| BSD / MIT / Apache‑2.0 | パーミッシブ | ✅ 組込可 (ただし派生物全体が GPLv3 へ引き込まれる) |
| AGPL‑3 | 強コピーレフト | ✅ 可 (同等強度) |
| **プロプライエタリ / クローズド** | なし | ❌ 不可 (商用ライセンス必須) |

---

## 5. コンプライアンス手順

1. **最新 LICENSE を確認**  
   ```bash
   git ls-remote https://github.com/ultralytics/ultralytics
   cat LICENSE
   ```
2. **製品ドキュメントに GPLv3 条文と著作権表記を掲載**  
3. **派生ソース公開方法を整備**（GitHub 公開 / ZIP 提供 等）  
4. **SaaS の場合は法務・Ultralytics と協議し OEM プラン要否を判断**  

---

## 6. まとめ
* **YOLOv8 OSS 版は GPLv3** → ソース公開義務があるコピーレフト。  
* **閉源・SaaS で使う場合** は Ultralytics の **商用 OEM ライセンス** を取得してコピーレフトを解除。  
* 学習済み weights も同ライセンスに従うため、配布するときは GPL 条文を添付するか商用契約を締結。  
* 実装前に必ず公式 LICENSE を再確認し、社内リーガルと相談すること。