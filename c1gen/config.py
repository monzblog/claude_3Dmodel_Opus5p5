"""首都高C1モデルの寸法・生成パラメータ。単位はすべてメートル。"""

# 取得範囲（C1全周を含む矩形: 南, 西, 北, 東）
BBOX = (35.645, 139.728, 35.700, 139.790)

# 平面直角座標系 IX系（東京）。Blender座標の原点はこの点
# Overpass は既定の python-requests の User-Agent を 406 で拒否するため名乗る
USER_AGENT = "c1gen/0.1 (+https://github.com/monzblog/claude_3Dmodel_Opus5p5)"

CRS = "EPSG:6677"
ORIGIN_LATLON = (35.6812, 139.7600)

# 断面
LANE_WIDTH = 3.25          # 首都高C1の車線幅
SHOULDER_LEFT = 0.75       # 左路肩（左側通行なので外側）
SHOULDER_RIGHT = 0.50      # 右路肩（中央側）
DEFAULT_LANES = 2

# 区画線（高速道路の車線境界線: 実線8m・間隔12m）
MARK_WIDTH = 0.15
DASH_LEN = 8.0
DASH_GAP = 12.0
MARK_LIFT = 0.01

# 壁高欄（コンクリート製）
BARRIER_HEIGHT = 0.90       # PLATEAU の実測（路面から上端まで）の中央値 約0.9m
BARRIER_BASE = 0.50
BARRIER_TOP = 0.20

# 横断勾配（片勾配）。直線は左（路肩側）へ 2% 下げて排水し、カーブは内側へ下げる。
# 片勾配 ≈ 設計速度²/(127R) × CANT_FACTOR を CROSS_SLOPE〜MAX_CANT に収める（都市部の上限 6%）
CROSS_SLOPE = 0.02
MAX_CANT = 0.06
DESIGN_SPEED = 60.0        # km/h（C1 本線の設計速度の目安）
RAMP_DESIGN_SPEED = 40.0
CANT_FACTOR = 0.45
CANT_TRANSITION = 40.0     # 片勾配をすりつける長さ（緩和区間の目安）

# 高架
DECK_THICKNESS = 2.0
ELEVATED_MIN = 3.0         # 地面からこれ以上高ければ橋脚を立てる
PIER_SPACING = 35.0
PIER_SIZE = 2.0
CROSSBEAM_DEPTH = 1.5

# 路面の高さ（地面からの相対値）。OSMのタグから決め、あとで滑らかにする
BRIDGE_BASE_HEIGHT = 10.0
BRIDGE_LAYER_STEP = 6.0
TUNNEL_DEPTH = 12.0
CUTTING_DEPTH = 7.0
PROFILE_SIGMA = 45.0       # 縦断を滑らかにするガウス幅
MAX_GRADE = 0.07
TWIN_DZ = 5.5              # 上下線の高さの差がこれ未満なら、同じ高さに並ぶものとして間隔を確保する

# トンネル
TUNNEL_HEIGHT = 6.0
TUNNEL_COVER_MIN = 1.0     # 天井の上にこれだけ土があればトンネル扱い
TUNNEL_LIGHT_SPACING = 6.0

# 照明柱
LIGHT_SPACING = 40.0
LIGHT_POLE_HEIGHT = 10.0
LIGHT_ARM = 2.0

# 標識
SIGN_ADVANCE = 300.0       # 出口の手前に置く案内標識の距離
SIGN_CLEARANCE = 5.5
SPEED_SIGN_SPACING = 1000.0
SPEED_LIMIT = 50

# サンプリング
SAMPLE_STEP = 2.0
XY_SMOOTH_SIGMA = 5.0

# 分岐・合流ランプの長さ
RAMP_LENGTH = 250.0
RAMP_LANES = 1

# 地形メッシュ
TERRAIN_STEP = 10.0
TERRAIN_MARGIN = 300.0

# ---- 見た目の種類分け ----------------------------------------------------

# 遮音壁（沿道側の壁高欄の上）: 下が金属の吸音板、上が透明板
SOUND_PANEL_HEIGHT = 1.5
SOUND_CLEAR_HEIGHT = 1.2
SOUND_POST_SPACING = 2.0

# 防護柵（川の上の区間など）: 壁高欄の上の金属パイプ
RAIL_HEIGHTS = (0.30, 0.60)
RAIL_POST_SPACING = 2.0

# 川の上を通る区間の概略線（緯度, 経度）。OSM の川データ（data/osm_water.json）が無い時に使う
RAIL_ZONES = [
    [(35.6840, 139.7740), (35.6870, 139.7705), (35.6905, 139.7640), (35.6915, 139.7600)],  # 日本橋川
    [(35.6530, 139.7500), (35.6548, 139.7440), (35.6553, 139.7400), (35.6553, 139.7367)],  # 古川
]
RAIL_ZONE_RADIUS = 120.0
WATER_BUFFER = 30.0

# トンネル内
WALKWAY_WIDTH = 0.75       # 点検用通路
WALKWAY_HEIGHT = 0.25
TILE_TOP = 3.0             # 白いタイルパネルの上端（路面から）
TILE_DIRTY_TOP = 1.0       # この高さまでは汚れたタイル
EQUIP_SPACING = 50.0       # 非常用設備（非常電話・消火栓）の箱
GUIDE_LIGHT_SPACING = 50.0 # 避難誘導灯
JETFAN_SPACING = 200.0
TUNNEL_LIGHT_LEN = 1.5

# 路面
SHARP_CURVE_RADIUS = 120.0   # これより急なカーブは赤いカラー舗装・矢羽根・車線変更禁止
YELLOW_NEAR_JUNCTION = 150.0 # 分岐・合流の前後これだけは車線変更禁止
DOT_ZONE = 80.0              # 急カーブの手前の減速ドットの区間
JOINT_SPACING = 40.0         # 高架の伸縮継手
DELINEATOR_SPACING = 10.0    # 壁高欄の上の視線誘導標
CHEVRON_SPACING = 16.0
GORE_ZEBRA_SPACING = 3.0

# ---- 紹介動画の背景 ------------------------------------------------------
BUILDING_RADIUS = 350.0      # C1 からこの距離までの建物を置く
FLOOR_HEIGHT = 3.5           # building:levels から高さを出す時の 1 階分
