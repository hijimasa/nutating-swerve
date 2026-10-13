# 4輪・偏心クランク式全方向移動ロボットの MuJoCo モデル

単輪で確認した「偏心クランク＋受動２軸ジンバル」の操舵モジュールを４個使った、製作を前提とする車両モデルです。駆動モーターのハウジングは周回運動（ナテーション）するだけで連続回転しないため、スリップリングなしで操舵を無限回転させられます。

現行の第4版は、剛体の車体（ロッカーなし）で、中空の部材を使いません。ケーブルは外付けの曲げ金具に沿わせ、ジンバルの各軸の延長線上に置いた U 字ループで動きを吸収します。

結果と考察は `REPORT.md`、部品の構成と寸法は `BOM.md` にまとめています。

## 動作環境

親フォルダの `requirements.txt` と同じです（MuJoCo 3.3.7、NumPy、matplotlib、Pillow）。動画を作る場合だけ、`ffmpeg` と OpenGL 環境が必要です。画面のない Linux では `xvfb-run` を使ってください。

```bash
cd quad
python verify_quad.py          # 全検証（ノートPC 1コアで約10分）
python plot_quad.py            # results/quad_results.png
```

## 走らせて見る

```bash
python run_quad.py --scenario circle_spin --viewer      # 円周走行しながら車体を1周/周回で回転
python run_quad.py --scenario square --viewer           # 0.6 m 四角形を2周（車体の向きは固定）
python run_quad.py --scenario star --viewer             # 8方向に往復
python run_quad.py --scenario spin --viewer             # その場旋回
python run_quad.py --scenario straight --bumps --viewer  # 左車輪の下に高さ8 mmの段差
python run_quad.py --scenario straight --bumps --rocker  # 前側ロッカーあり（比較用）
```

オプション：

- `--no-feedback`：位置・姿勢の補正なし（モデル制御のみ）。
- `--no-coupling-ff`：駆動と操舵の連成トルクのフィードフォワードを外します。
- `--flip on|off`：車輪の逆転を許可するかどうかです。`off` にすると、操舵は一方向に回り続けます。
- `--dt 0.0005`：時間刻みです。
- `--video out.mp4`：全体と FL モジュールの拡大を左右に並べた動画を出力します。

ケーブル単体の検証：

```bash
python cable_test.py --layout hook --turns 5     # 第4版（現行）：外付け金具＋各軸に U 字ループ
python cable_test.py --layout gimbal --turns 5   # 第2版：中空トラニオン＋各軸に U 字ループ
python cable_test.py --layout clamp --turns 5    # 第3版の試行（不採用）：中実トラニオン＋軸の近くの留め具
python cable_test.py --layout pin --turns 5      # 第1版：クランクピンのスタブと中空クランク軸を通す
python cable_test.py --layout crank --turns 2    # 従来方式相当：ケーブル端が操舵と一緒に回る
python cable_test.py --layout hang --turns 5     # 検討案：留め具2か所だけでケーブルを垂らす（results/cable_simplify/README.md）
python cable_test.py --layout axis --turns 5     # 検討案（不成立）：モーター側面から中空クランク軸へ。操舵180°で裏返ってロックする
```

## ファイル構成

| ファイル | 内容 |
| --- | --- |
| `geometry.py` | 寸法・質量と閉形式の運動学（接地点の速度係数 `J_phi`、`J_psi`） |
| `build_quad.py` | MJCF の生成（車両、単体ベンチ、ケーブル） |
| `controller.py` | モデル制御：逆運動学で操舵角と車輪速度を決め、小さな補正を加えます |
| `scenarios.py` | 参照軌道（最小躍度の区間、ランプ付きの円） |
| `run_quad.py` | トルク駆動のシミュレーション、ログ、要約、動画 |
| `kinematics_check.py` | 閉リンクの残差、姿勢の周期性、`J_phi`・`J_psi` と MuJoCo ヤコビアンの照合 |
| `check_clearance.py` | 操舵１周のあいだの部品干渉（解析的な符号付き距離による） |
| `cable_test.py` | 弾性ケーブル（MuJoCo cable プラグイン）のねじれ・曲げ・クリアランス。経路を3種類比較 |
| `verify_quad.py` | 上記をすべて実行し、`results/verification.json` に合否を書き出します |
| `plot_quad.py` | まとめ図 |
| `route_figure.py` | ケーブル経路の図（`results/cable_route.png`、OpenGL が必要） |
| `results/` | 各試験の `trace.csv`・`summary.json`・`model.xml`、`videos/` |

## モデルの要点

- 車体は自由浮遊（6自由度）の剛体です。`--rocker` を付けると、前側２モジュールが前後軸まわりに揺動するロッカー（±6°）に載ります。
- 各モジュールの能動軸は２つ（クランク＝操舵、車輪＝駆動）、受動軸は２つ（ジンバル）です。閉リンクは `equality/connect`（クランクピンの球面軸受に相当）で閉じています。
- 接触するのはタイヤだけです。タイヤは丸断面ウレタン（Oリング状）のトーラスメッシュで、MuJoCo は凸包で接触を計算します。構造部品どうしの干渉は `check_clearance.py` で別に確認しています。
- 制御はアクチュエータのトルクのみで、DC モーターのトルク‐速度特性で制限しています。受動関節の `qpos` を書き換える処理はありません。
