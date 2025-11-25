from flask import Blueprint, jsonify, request, render_template
from .services import user_service, weapon_service, security_service
from models import Player, Weapon

main_bp = Blueprint('main', __name__)

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
    users = user_service._read_users_unlocked()
    for enc_user, enc_pass in users.items():
        if security_service.decrypt(enc_user) == username:
            if security_service.decrypt(enc_pass) == password:
                return jsonify({"status": "success", "message": "登录成功", "redirect": "/dashboard"})
            else:
                return jsonify({"status": "error", "message": "密码错误"}), 401
    return jsonify({"status": "error", "message": "用户不存在"}), 404

@main_bp.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

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