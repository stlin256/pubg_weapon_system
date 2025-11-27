from flask import Blueprint, jsonify, request, render_template
import os
import pandas as pd
import time
import logging
import warnings
from werkzeug.utils import secure_filename
from .services import user_service, weapon_service, security_service
from app.inference_service import inference_service
from app.admin_service import admin_service

# --- 配置结构化日志 ---
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加文件处理器
file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# 添加流处理器 (控制台输出)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
from models import Player, Weapon
from functools import wraps

main_bp = Blueprint('main', __name__)
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# --- 管理员身份验证装饰器 ---
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 简单实现：检查 session 或请求头中的管理员标志
        # 在实际生产中，这里应该使用更安全的 token 验证机制
        username = request.headers.get('X-Username')
        password = request.headers.get('X-Password')
        if not admin_service.is_admin(username, password):
            return jsonify({"status": "error", "message": "管理员权限不足"}), 403
        return f(*args, **kwargs)
    return decorated_function

@main_bp.route("/")
def index():
    return render_template("index.html")

@main_bp.route("/login")
def login_page():
    return render_template("login.html")

@main_bp.route("/register")
def register_page():
    return render_template("register.html")

@main_bp.route("/api/register", methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"status": "error", "message": "请求缺少用户名或密码"}), 400
    if user_service.add_user(data['username'], data['password']):
        return jsonify({"status": "success", "message": "注册成功"}), 201
    else:
        return jsonify({"status": "error", "message": "用户已存在"}), 409

@main_bp.route("/api/login", methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"status": "error", "message": "请求缺少用户名或密码"}), 400
    username = data['username']
    password = data['password']

    # 首先，检查是否为管理员
    if admin_service.is_admin(username, password):
        return jsonify({
            "status": "success",
            "message": "管理员登录成功",
            "redirect": "/admin",
            "is_admin": True
        })

    # 如果不是管理员，执行普通用户登录流程
    users = user_service._read_users_unlocked()
    for enc_user, enc_pass in users.items():
        if security_service.decrypt(enc_user) == username:
            if security_service.decrypt(enc_pass) == password:
                return jsonify({
                    "status": "success",
                    "message": "登录成功",
                    "redirect": "/dashboard"
                })
            else:
                return jsonify({"status": "error", "message": "密码错误"}), 401
    return jsonify({"status": "error", "message": "用户不存在"}), 404

@main_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@main_bp.route("/admin")
def admin_page():
    # 这里可以添加一个简单的 session 检查，确保只有管理员能访问
    return render_template("admin.html")

@main_bp.route("/api/weapons", methods=['GET'])
def get_weapons():
    username = request.headers.get('X-Username')
    if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
    return jsonify(weapon_service.get_player(username).to_dict())

@main_bp.route("/api/weapons", methods=['POST'])
def add_weapon():
    username = request.headers.get('X-Username')
    if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
    data = request.get_json()
    if not data: return jsonify({"status": "error", "message": "请求体为空"}), 400
    player = weapon_service.get_player(username)
    player.add_weapon(Weapon.from_dict(data))
    weapon_service.save_player(player)
    return jsonify({"status": "success", "message": f"武器 {data['name']} 添加成功"}), 201

@main_bp.route("/api/weapons/<path:weapon_name>", methods=['PUT'])
def update_weapon(weapon_name):
    username = request.headers.get('X-Username')
    if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
    data = request.get_json()
    if not data: return jsonify({"status": "error", "message": "请求体为空"}), 400
    player = weapon_service.get_player(username)
    weapon = player.get_weapon(weapon_name)
    if not weapon: return jsonify({"status": "error", "message": "未找到该武器"}), 404
    
    # 更新所有可编辑字段
    weapon.firing_mode = data.get('firing_mode', weapon.firing_mode)
    weapon.mag_count = int(data.get('mag_count', weapon.mag_count))
    weapon.ammo_in_mag = int(data.get('ammo_in_mag', weapon.ammo_in_mag))
    
    weapon_service.save_player(player)
    return jsonify({"status": "success", "weapon": weapon.to_dict()})

@main_bp.route("/api/weapons/<path:weapon_name>", methods=['DELETE'])
def delete_weapon(weapon_name):
    username = request.headers.get('X-Username')
    if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
    player = weapon_service.get_player(username)
    if player.remove_weapon(weapon_name):
        weapon_service.save_player(player)
        return jsonify({"status": "success", "message": f"武器 {weapon_name} 删除成功"})
    else:
        return jsonify({"status": "error", "message": "未找到该武器"}), 404

# ==============================================================================
# 枪声识别 API
# ==============================================================================

@main_bp.route("/api/models", methods=['GET'])
def get_available_models():
   """返回所有可用的模型，按任务分类。"""
   return jsonify(inference_service.get_available_models())

@main_bp.route("/api/benchmark", methods=['GET'])
def get_benchmark_data():
   """读取评估结果CSV，返回包含准确率和F1分数的JSON。"""
   csv_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'evaluation_results.csv')
   try:
       df = pd.read_csv(csv_path)
       # 仅选择我们需要的列
       df_filtered = df[['model', 'target', 'accuracy', 'f1-score (macro)']]
       return jsonify(df_filtered.to_dict(orient='records'))
   except FileNotFoundError:
       return jsonify({"status": "error", "message": "评估文件未找到"}), 404

@main_bp.route("/api/preferences", methods=['POST'])
def save_preferences():
   """保存用户的模型偏好设置。"""
   username = request.headers.get('X-Username')
   if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
   
   data = request.get_json()
   if not data: return jsonify({"status": "error", "message": "请求体为空"}), 400

   player = weapon_service.get_player(username)
   player.model_preferences['weapon'] = data.get('weapon', player.model_preferences['weapon'])
   player.model_preferences['distance'] = data.get('distance', player.model_preferences['distance'])
   player.model_preferences['direction'] = data.get('direction', player.model_preferences['direction'])
   
   weapon_service.save_player(player)
   return jsonify({"status": "success", "message": "偏好设置已保存"})

@main_bp.route("/api/recognize", methods=['POST'])
def recognize_sound():
    """处理音频上传并返回所有三个任务的识别结果，并记录结构化日志。"""
    start_time = time.time()
    username = request.headers.get('X-Username', 'Unknown')
    
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "未找到音频文件"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"}), 400

    prefs = {
        'weapon': request.form.get('weapon_model'),
        'distance': request.form.get('distance_model'),
        'direction': request.form.get('direction_model')
    }

    if not all(prefs.values()):
        return jsonify({"status": "error", "message": "缺少模型选择"}), 400
    
    filename = secure_filename(file.filename)
    upload_folder = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    temp_path = os.path.join(upload_folder, filename)
    file.save(temp_path)

    results = {}
    try:
        # 抑制 scikit-learn 和 xgboost 的 UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for target, model_name in prefs.items():
                prediction = inference_service.predict(temp_path, target, model_name)
                results[target] = prediction
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 记录成功的日志
        log_message = (
            f"Recognition SUCCESS | User: {username} | File: {filename} | "
            f"Models: (w: {prefs['weapon']}, d: {prefs['distance']}, dir: {prefs['direction']}) | "
            f"Results: (w: {results['weapon']}, d: {results['distance']}, dir: {results['direction']}) | "
            f"Duration: {duration:.2f}s"
        )
        logger.info(log_message)
        
        return jsonify({"status": "success", "results": results})
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"Recognition FAILED | User: {username} | File: {filename} | Duration: {duration:.2f}s | Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ==============================================================================
# 管理员 API
# ==============================================================================
@admin_bp.route('/stats', methods=['GET'])
@admin_required
def get_stats():
    # 简单的统计实现
    num_users = len(user_service._read_users_unlocked())
    log_file = 'app.log'
    num_predictions = 0
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            num_predictions = sum(1 for line in f if 'Recognition SUCCESS' in line)
    
    return jsonify({
        "num_users": num_users,
        "num_predictions": num_predictions,
        "cache_size": len(inference_service._model_cache)
    })

@admin_bp.route('/users', methods=['GET'])
@admin_required
def get_users():
    users_encrypted = user_service._read_users_unlocked()
    users_decrypted = [security_service.decrypt(u) for u in users_encrypted.keys()]
    return jsonify(users_decrypted)

@admin_bp.route('/logs', methods=['GET'])
@admin_required
def get_logs():
    log_file = 'app.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            # 读取最后 100 行
            lines = f.readlines()
            return jsonify(lines[-100:])
    return jsonify([])

@admin_bp.route('/users/<username>', methods=['DELETE'])
@admin_required
def delete_user_by_admin(username):
    # 此处需要一个新服务函数来通过用户名删除用户
    if user_service.delete_user_by_username(username):
        # 同时删除该用户的武器数据文件
        player_data_path = weapon_service.get_pdp(username)
        if os.path.exists(player_data_path):
            os.remove(player_data_path)
        return jsonify({"status": "success", "message": f"用户 {username} 已被删除"})
    return jsonify({"status": "error", "message": "用户未找到"}), 404

@admin_bp.route('/clear_cache', methods=['POST'])
@admin_required
def clear_cache():
    inference_service._model_cache.clear()
    return jsonify({"status": "success", "message": "模型缓存已清空"})

@admin_bp.route('/cache_strategy', methods=['GET', 'POST'])
@admin_required
def handle_cache_strategy():
    if request.method == 'GET':
        return jsonify({"strategy": inference_service.cache_strategy})
    
    if request.method == 'POST':
        strategy = request.json.get('strategy')
        if not strategy or strategy not in ['all', 'selected']:
            return jsonify({"status": "error", "message": "无效的策略"}), 400
        
        current_strategy = inference_service.set_cache_strategy(strategy)
        return jsonify({"status": "success", "strategy": current_strategy, "message": f"缓存策略已设置为 '{current_strategy}'"})

@main_bp.route('/api/preload_model', methods=['POST'])
def preload_model():
    """为普通用户提供的主动加载模型的接口。"""
    username = request.headers.get('X-Username')
    if not username: return jsonify({"status": "error", "message": "缺少用户身份认证信息"}), 401
    
    model_name = request.json.get('model_name')
    if not model_name:
        return jsonify({"status": "error", "message": "缺少模型名称"}), 400

    try:
        inference_service.preload_model(model_name)
        return jsonify({"status": "success", "message": f"模型 {model_name} 已预加载"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500