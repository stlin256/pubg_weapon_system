import json

class Weapon:
    """
    代表一件武器的数据模型类，精确匹配数据源。
    """
    def __init__(self, name: str, damage: float, firing_mode: str, capacity: int, mag_count: int = 0, ammo_in_mag: int = 0):
        self.name = name
        self.damage = float(damage)
        self.firing_mode = firing_mode
        self.capacity = int(capacity)       # 弹夹容量
        self.mag_count = int(mag_count)     # 弹夹数量
        self.ammo_in_mag = int(ammo_in_mag) # 枪内剩余子弹

    @property
    def total_ammo(self):
        """计算备弹总数（不包括枪内的）。"""
        return self.mag_count * self.capacity

    def to_dict(self):
        """将武器对象转换为字典。"""
        return {
            "name": self.name,
            "damage": self.damage,
            "firing_mode": self.firing_mode,
            "capacity": self.capacity,
            "mag_count": self.mag_count,
            "ammo_in_mag": self.ammo_in_mag,
            "total_ammo": self.total_ammo, # 也包含计算属性
        }

    @staticmethod
    def from_dict(data: dict):
        """从字典创建武器对象。"""
        return Weapon(
            name=data.get('name'),
            damage=data.get('damage'),
            firing_mode=data.get('firing_mode'),
            capacity=data.get('capacity'),
            mag_count=data.get('mag_count', 0),
            ammo_in_mag=data.get('ammo_in_mag', 0)
        )


class Player:
    """
    代表一个玩家的数据模型类，管理玩家信息和武器。
    """
    def __init__(self, username: str):
        self.username = username
        self.weapons = []

    def add_weapon(self, weapon: Weapon):
        self.weapons.append(weapon)

    def remove_weapon(self, weapon_name: str) -> bool:
        initial_count = len(self.weapons)
        self.weapons = [w for w in self.weapons if w.name != weapon_name]
        return len(self.weapons) < initial_count

    def get_weapon(self, weapon_name: str) -> Weapon or None:
        for weapon in self.weapons:
            if weapon.name == weapon_name:
                return weapon
        return None

    def get_total_ammo_summary(self) -> int:
        """计算玩家所有武器的子弹总数（包括枪内和备弹）。"""
        return sum(w.ammo_in_mag + w.total_ammo for w in self.weapons)
        
    def to_dict(self):
        return {
            "username": self.username,
            "weapons": [weapon.to_dict() for weapon in self.weapons]
        }
    
    @staticmethod
    def from_dict(data: dict):
        player = Player(username=data.get('username'))
        if 'weapons' in data:
            for weapon_data in data['weapons']:
                player.add_weapon(Weapon.from_dict(weapon_data))
        return player