# 首都高 C1（都心環状線）3Dモデル

ADASシミュレーション用に、首都高C1の内回り・外回りを1周分 Blender 形式（.blend）で作るための生成プログラムです。周囲の建物は含みません。

## 作るもの

| 種類 | 内容 |
|---|---|
| 路面 | 車線数は OSM の `lanes`、車線幅 3.25m、路肩 左0.75m・右0.5m。横断勾配は直線で左へ2%、カーブは半径から片勾配（最大6%）を計算してすりつけ。舗装のテクスチャ付き |
| 区画線 | 車道外側線、車線境界線（破線 8m・間隔12m）。トンネル内・分岐合流の前後・急カーブは黄色の実線（車線変更禁止） |
| 路面標示 | 急カーブ（半径120m未満）の赤いカラー舗装と、その手前の減速ドット。分岐・合流部の導流帯（ゼブラ） |
| 壁（沿道側） | 区間ごとに種類を分ける: 遮音壁（金属の吸音板＋透明板）、川の上は金属の防護柵、掘割は擁壁、トンネルは点検用通路 |
| 壁高欄 | 高さ1.1mのコンクリート壁（汚れのテクスチャ付き）。上に視線誘導標（左は白、右は橙） |
| 高架 | 床版、横梁、橋脚、伸縮継手（40m間隔） |
| トンネル | 白いタイルパネルの壁（下の方は汚れ）、灯具の列、非常用設備の箱、避難誘導灯、ジェットファン、坑口 |
| 照明 | 左側の壁高欄に照明柱（40m間隔） |
| 標識 | 出口の300m手前と直前に門型の案内標識（例: 「4 新宿」「霞が関」）、合流部の警戒標識、急カーブの矢羽根板、1kmごとの最高速度50km/h |
| 分岐・合流 | 本線から約250mだけ作る。分岐の先端にクッションドラム |
| 地形 | 国土地理院の標高（5mメッシュ優先）から10m間隔で作成 |

壁の種類は `c1gen/sections.py` で決めます。川の上の区間は OSM の川（`data/osm_water.json`）から判定し、無ければ `config.RAIL_ZONES` の概略線を使います。実物と違う区間は `config.py` で直せます。

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

## データの出典

- 道路・川: © OpenStreetMap contributors（ODbL）
- 標高: 国土地理院 基盤地図情報 数値標高モデル（地理院タイル）
- 実測の路面・橋・トンネルの形（`data/plateau_c1.npz`）: 3D都市モデル（Project PLATEAU）首都高速道路（2023年度）国土交通省、政府標準利用規約2.0（CC BY 4.0互換）を加工して作成

`data/plateau_c1.npz` は、PLATEAU の CityGML（`13_tokyo_tran-mlit_2023_citygml_1_op.zip` を `data/plateau/` に展開したもの）から `python tools/plateau_extract.py` で作ります。
