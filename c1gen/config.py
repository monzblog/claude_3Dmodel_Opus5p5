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
BARRIER_HEIGHT = 1.10
BARRIER_BASE = 0.50
BARRIER_TOP = 0.20

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
