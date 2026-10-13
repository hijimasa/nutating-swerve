# 偏心クランク＋受動２軸ジンバル車輪の MuJoCo 検証

駆動モーター本体の後端を水平な円に沿って動かし、車軸の向きを一定傾斜の円錐に沿って周回させる試作モデルです。操舵モーター１個、車輪駆動モーター１個、受動ジンバル２軸で構成しています。

**確認できたこと：** トルク駆動で操舵10周を達成し、ジンバルは有限範囲を往復、モーター本体は初期姿勢へ復帰しました。

**確認できなかった／満たさなかったこと：** ケーブルの物理的な干渉・屈曲寿命は未モデル化です。また、今回の支点配置の瞬間的な平面速度のヤコビアンはランク１で、ホロノミックなアクティブキャスターのランク２の動作を満たしません。無限操舵機構の検証モデルとして使用してください。

## 動作環境

- 検証済み：Linux、Python 3.12、MuJoCo 3.3.7、NumPy 2.3.5。
- GUI：通常の OpenGL 対応環境が必要です。物理計算のみの実行には画面・GPUは不要です。
- 動画の再生成だけに `ffmpeg` とオフスクリーン OpenGL が必要です。添付動画を見るだけなら不要です。

## インストール

`single_wheel` ディレクトリで実行します（依存関係はリポジトリ直下の `requirements.txt`）。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../requirements.txt
```

Windowsでは最初のコマンドを `py -m venv .venv`、有効化を `.venv\Scripts\Activate.ps1` に置き換えてください。実行スクリプトはローカルGUIでの操作にも対応しています。

## まず動きを見る

接地なし。重力なし、車体固定、車輪を回しながら３周操舵します。

```bash
python run.py --mode air --wheel-speed 3 --cycles 3 --viewer --out results/my_air
```

接地あり。車体の３方向の並進は自由、３方向の回転は固定した、理想的な計測台です。操舵２周と車輪回転の組み合わせを観察します。

```bash
python run.py --mode ground --wheel-speed 3 --cycles 2 --viewer --out results/my_ground
```

操舵を保持した直進試験：

```bash
python run.py --mode ground --hold-steering --wheel-speed 3 --cycles 1 --viewer --out results/my_straight
```

`--wheel-speed -3` で車輪を逆転できます。`--period 8` なら操舵１周８秒、`--tilt 30` なら傾斜30°です。GUIは指定した試験時間で閉じます。長く観察する場合は `--cycles 30` などを指定してください。CLIで設定した制御が毎ステップ更新されるため、GUIのアクチュエータスライダーは制御設定には使用しません。

## 自動検証と出力

```bash
python verify.py
python plot_results.py
```

GUIなしで実行できます。`verify.py` は次を確認します。

1. 傾斜30°・45°・60°でクランクと後端接続点の位置が一致すること。
2. 本体の姿勢が１周後に復帰すること。
3. ジンバルにアクチュエータが割り当てられていないこと。
4. トルク駆動で10周し、閉リンクの残差・姿勢の復帰誤差が閾値内であること。
5. 接地試験で接触を保ち、時間刻み1 msと0.5 msの最終位置が近いこと。
6. 今回の配置の瞬間的な平面速度のヤコビアンがランク１であること。

各試験には `trace.csv`、`summary.json`、実際に使用した `model.xml` が出力されます。添付の結果は `results/verification.json`、グラフは `results/verification.png`、説明は `REPORT.md` です。

## MJCFの構成

- `models/air.xml`：接地なしのモデル。
- `models/ground.xml`：接地ありのモデル。
- `build_model.py`：寸法を設定し、MJCFを生成。
- `run.py`：トルク制御・計測・GUI・任意の動画出力。
- `verify.py`：幾何と動力学の独立した検証。
- `plot_results.py`：グラフ作成。

２つの木構造の分岐は、`rear_pin` と `crank_pin` の間の `equality/connect` で閉じています。この拘束は球面ジョイント相当で、位置を結び、相対姿勢を拘束しません。駆動側と操舵側にだけ `motor` アクチュエータを使用しています。動力学の実行中にジンバルの `qpos` を書き換える処理はありません。独立した幾何検証では理想姿勢を代入しますが、これは動力学試験とは別です。

## 既定の寸法・設定

| 項目 | 値 |
| --- | ---: |
| 鉛直からの傾斜 | 45° |
| 車輪半径 | 70 mm |
| 車輪幅 | 12 mm |
| 支点からモーター後端接続点 | 120 mm |
| 支点から車輪中心までの軸方向距離 | 40 mm |
| 偏心クランク半径 | 約84.85 mm |
| 支点からクランク面までの高さ | 約84.85 mm |
| 総質量 | 1.71 kg |
| 並進摩擦係数 | 1.0 |
| ねじり摩擦パラメータ | 0.003 m |
| 接触の `solref` | `0.015 1` |
| 閉リンクの `solref` | `0.003 1` |

`gimbal_y` は `ref=45°` を設定し、読み出す角度が実際の内側傾斜角になるようにしています。２軸とも約±45°を往復します。車軸の傾斜角は固定45°で、各ジンバル角はその傾斜角とは異なります。

寸法は仮定値です。`build_model.py` の `Geometry` を変更して生成し直してください。例えば：

```bash
python build_model.py --tilt 30
```

`run.py` は `Geometry` からモデルを再生成します。`models/*.xml` を直接編集した場合、その編集は `run.py` には反映されません。コード側の生成条件を編集するか、編集したXMLを直接読み込むように変更してください。

## 動画を再生成する場合

LinuxでEGLが利用できる例です。GUIは不要です。

```bash
MUJOCO_GL=egl python run.py --mode air --wheel-speed 3 --cycles 3 --out results/movie_air --video results/movie_air/mechanism.mp4
```

EGLが利用できない環境では、このコマンドは描画の初期化に失敗することがあります。その場合も `--video` を外した物理計算と、ローカルGUIでの `--viewer` は別に利用できます。

## 検証の範囲

今回のモデルでは、機構内部の自己衝突を無効にしています。赤い枠、支柱、モーター外形は概略表示で、製作可能なCAD・軸受の詳細モデルではありません。タイヤは剛体円柱と柔らかい接触拘束で近似しています。ケーブル、配線経路、疲労、ギヤ、軸受の隙間・摩擦、モーターの電気特性は未モデル化です。計測したトルクはこの仮定モデルの値で、実機モーターの選定値ではありません。

次の設計段階では、部品干渉とケーブル経路を確認し、必要なら接地点の配置を変更して瞬間的な平面速度のランク２を実現できるかを検討します。

## 参考資料

- [MuJoCo 3.3.7 XML Reference — equality/connect](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#equality-connect)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/3.3.7/python.html)
- [ホロノミック全方向移動ロボットの開発：操舵軸と車輪のオフセット](https://www.jstage.jst.go.jp/article/jrsj1983/15/8/15_8_1139/_article/-char/ja/)
