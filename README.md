# 首都高 C1（都心環状線）3Dモデル

ADASシミュレーション用に、首都高C1の内回り・外回りを1周分 Blender 形式（.blend）で作るための生成プログラムです。周囲の建物は含みません。

## 作るもの

| 種類 | 内容 |
|---|---|
| 路面 | 車線数は OSM の `lanes`、車線幅 3.25m、路肩 左0.75m・右0.5m |
| 区画線 | 車道外側線（実線）と車線境界線（破線 8m・間隔12m） |
| 壁高欄 | 高さ1.1mのコンクリート壁を左右に |
| 高架 | 床版（厚さ2m）、横梁、橋脚（35m間隔） |
| トンネル・掘割 | トンネルの壁・天井・坑口、掘割の擁壁 |
| 照明 | 左側の壁高欄に照明柱（40m間隔）、トンネル内は天井灯（6m間隔）。灯具は発光マテリアル |
| 標識 | 出口の300m手前と直前に門型の案内標識（行き先は OSM の `destination`）、合流部の警戒標識、1kmごとの最高速度50km/h |
| 分岐・合流 | 本線から約250mだけ作る |
| 地形 | 国土地理院の標高（5mメッシュ優先）から10m間隔で作成 |

路面の高さは、地理院の地面の標高に OSM のタグ（`bridge`、`tunnel`、`cutting`、`layer`）から決めた高さを足し、勾配7%以内に滑らかにしています。実際の図面の値ではないので、高さの誤差は数m程度あります。

座標は平面直角座標系IX系（EPSG:6677）で、原点は北緯35.6812度・東経139.7600度です（X=東、Y=北、Z=標高、単位m）。

## 使い方

```sh
pip install -r requirements.txt
tools/get_font.sh                  # 標識の文字用フォント
python -m c1gen.build --fetch      # OSM と標高を取得して output/C1_loop.blend を作る
python -m c1gen.build              # 取得済みのデータから作り直す
python -m c1gen.build --synthetic  # ネットに出られないときの動作確認（楕円の仮コース）
python tools/preview.py output/C1_loop.blend output/preview  # 確認用の画像
```

`--fetch` には `overpass-api.de` と `cyberjapandata.gsi.go.jp` への接続が必要です。

出力:

- `output/C1_loop.blend`: コレクション `C1_Outer`（外回り）、`C1_Inner`（内回り）、`C1_Ramps`、`C1_Signs_Text`、`Terrain`
- `output/C1_loop_lanes.json`: 各車線の中心線の点列（シミュレータで自車位置や車線を扱うとき用）

寸法などは `c1gen/config.py` で変えられます。
