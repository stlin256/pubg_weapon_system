import os

class AdminService:
    def __init__(self, credentials_path='admin_credentials.txt'):
        self.credentials_path = credentials_path
        self._admins = self._load_admins()

    def _load_admins(self):
        """从文件中加载管理员凭据。"""
        admins = {}
        if os.path.exists(self.credentials_path):
            with open(self.credentials_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        username, password = parts
                        admins[username] = password
        return admins

    def is_admin(self, username, password):
        """检查提供的用户名和密码是否匹配管理员凭据。"""
        return self._admins.get(username) == password

# 创建一个单例
admin_service = AdminService()