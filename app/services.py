import base64
import json
import os
import threading
import pandas as pd
from models import Player, Weapon
try:
    from Cryptodome.Cipher import AES
    from Cryptodome.Random import get_random_bytes
except ImportError:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes

# ==============================================================================
# 1. 加密服务
# ==============================================================================
def load_key(key_path='secret.key'):
    with open(key_path, 'rb') as f: return f.read()
KEY = load_key()

class SecurityService:
    def __init__(self, key):
        if len(key) not in [16, 24, 32]: raise ValueError("AES key must be 16, 24, or 32 bytes long")
        self.key = key
    def encrypt(self, p: str) -> str:
        b = p.encode('utf-8'); c = AES.new(self.key, AES.MODE_GCM); ct, tag = c.encrypt_and_digest(b); n = c.nonce
        return base64.b64encode(n + tag + ct).decode('utf-8')
    def decrypt(self, b64_ct: str) -> str:
        try:
            eb = base64.b64decode(b64_ct); n, tag, ct = eb[:16], eb[16:32], eb[32:]
            c = AES.new(self.key, AES.MODE_GCM, nonce=n); db = c.decrypt_and_verify(ct, tag)
            return db.decode('utf-8')
        except (ValueError, KeyError): return None
security_service = SecurityService(KEY)

# ==============================================================================
# 2. 用户数据服务
# ==============================================================================
class UserService:
    def __init__(self, db_path='data/users.dat'):
        self.db_path = db_path; self.lock = threading.Lock(); self._ensure_db_exists()
    def _ensure_db_exists(self):
        dp = os.path.dirname(self.db_path)
        with self.lock:
            if not os.path.exists(dp): os.makedirs(dp)
            if not os.path.exists(self.db_path): self._write_users_unlocked({})
    def _read_users_unlocked(self) -> dict:
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f: ed = f.read()
            if not ed: return {}
            return json.loads(security_service.decrypt(ed))
        except (IOError, json.JSONDecodeError): return {}
    def _write_users_unlocked(self, users: dict):
        json_data = json.dumps(users, indent=4, ensure_ascii=False)
        encrypted_data = security_service.encrypt(json_data)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            f.write(encrypted_data)
    def add_user(self, u: str, p: str) -> bool:
        with self.lock:
            users = self._read_users_unlocked()
            for eu in users.keys():
                if security_service.decrypt(eu) == u: return False
            users[security_service.encrypt(u)] = security_service.encrypt(p)
            self._write_users_unlocked(users)
        return True
user_service = UserService()

# ==============================================================================
# 3. 武器数据服务
# ==============================================================================
class WeaponService:
    def __init__(self, idp='data/Arms.xlsx', pdd='data/players'):
        self.pdd = pdd; self.iw = self._load_iw(idp); os.makedirs(self.pdd, exist_ok=True)
    def _load_iw(self, dp: str) -> dict:
        try:
            df = pd.read_excel(io=dp, sheet_name='Sheet1', engine='openpyxl').fillna(0)
            w = {}
            for _, r in df.iterrows():
                if r['Name']:
                    w[r['Name']] = Weapon(
                        name=r['Name'],
                        damage=r['Damage'],
                        firing_mode=str(r['FiringMode']),
                        capacity=r['弹夹容量'],
                        mag_count=r['弹夹数量'],
                        ammo_in_mag=r['枪内剩余子弹数量']
                    )
            return w
        except FileNotFoundError: return {}
    def get_pdp(self, u: str) -> str:
        sf = base64.urlsafe_b64encode(u.encode()).decode()
        return os.path.join(self.pdd, f"{sf}_weapons.dat")
    def get_player(self, u: str) -> Player:
        p = self.get_pdp(u)
        if not os.path.exists(p):
            pl = Player(u)
            for wt in self.iw.values(): pl.add_weapon(Weapon.from_dict(wt.to_dict()))
            self.save_player(pl)
            return pl
        with open(p, 'r', encoding='utf-8') as f: ed = f.read()
        return Player.from_dict(json.loads(security_service.decrypt(ed)))
    def save_player(self, pl: Player):
        p = self.get_pdp(pl.username)
        ed = security_service.encrypt(json.dumps(pl.to_dict(), indent=4, ensure_ascii=False))
        with open(p, 'w', encoding='utf-8') as f: f.write(ed)
weapon_service = WeaponService()