# 武器管理系统 - 技术文档

本文档是 PUBG 武器管理系统的官方技术说明，旨在为开发者和维护者提供对系统架构、核心组件和运行方式的全面理解。

## 1. 系统架构概述

本系统采用基于 **Flask 的应用工厂模式 (Application Factory Pattern)** 构建，这是一种高度模块化、可扩展的 Web 应用架构。该模式将应用创建、配置和服务注册封装在一个函数中，有效避免了循环依赖问题，并提升了代码的可维护性和可测试性。

### 1.1 架构图

```mermaid
graph TD
    subgraph "项目根目录"
        A[run.py] --> B(app);
        H[models.py];
    end
    
    subgraph "应用包 (app)"
        B --> C[__init__.py: create_app()];
        C --> D[routes.py];
        D -- "导入服务" --> E[services.py];
        E -- "导入模型" --> H;
        C -- "渲染" --> F[templates/];
    end
    
    subgraph "数据层 (/data)"
        G[Arms.xlsx];
        I[users.dat];
        J[players/*.dat];
        K[secret.key];
    end
    
    style B fill:#2d3748,stroke:#fff
    style G fill:#1e2126,stroke:#fff
    style I fill:#1e2126,stroke:#fff
    style J fill:#1e2126,stroke:#fff
    style K fill:#1e2126,stroke:#fff
```

### 1.2 核心组件职责

*   **`run.py`**: **应用入口点**。其唯一职责是调用 `app` 包中的 `create_app` 工厂函数来获取一个配置好的 Flask 应用实例，并启动 WSGI 开发服务器。
*   **`app/` (应用包)**:
    *   **`__init__.py`**: 定义 **`create_app()` 工厂函数**。此函数负责实例化 `Flask` 应用，并将 `routes.py` 中定义的蓝图注册到应用上。
    *   **`routes.py`**: **路由与视图层**。使用 Flask 蓝图 (`Blueprint`) 统一管理应用的所有路由，包括页面渲染和 API 端点。它负责解析 HTTP 请求，调用服务层处理业务逻辑，并返回响应。
    *   **`services.py`**: **核心业务逻辑层**。此文件封装了应用的所有业务逻辑，与 Web 框架解耦。它包含了三个核心服务类：`SecurityService`, `UserService`, 和 `WeaponService`。
    *   **`templates/`**: **前端视图层**。存放应用的所有 HTML 模板文件。
*   **`models.py`**: **数据模型层**。定义了 `Weapon` 和 `Player` 两个核心的 Python 类，它们是系统业务逻辑操作的基础数据结构。
*   **`data/`**: **持久化数据层**。
    *   `Arms.xlsx`: 初始化的武器数据源。
    *   `secret.key`: 用于 AES 加密的 32 字节密钥。
    *   `users.dat`: 经 AES 加密的、包含所有用户认证信息的文件。
    *   `players/`: 存放每个用户独立的、经 AES 加密的武器库数据文件。

## 2. 功能与技术实现

### 2.1 用户认证系统

*   **注册 (`/api/register`)**: 接收用户名和密码，在 `UserService` 中对两者分别进行加密，然后将加密后的键值对存入 `users.dat`。
*   **登录 (`/api/login`)**: 接收明文用户名和密码。`UserService` 读取加密的 `users.dat`，遍历所有记录，逐一解密并与传入的凭据进行比对，以完成身份验证。
*   **前端状态**: 登录状态通过浏览器的 `localStorage` 进行前端管理。

### 2.2 武器管理系统

*   **数据初始化**: 新用户首次登录并访问武器库时，`WeaponService` 会从 `data/Arms.xlsx` 读取标准的武器列表，为该用户创建一套初始的武器数据。
*   **数据持久化**: 每个用户的武器库数据都作为一个独立的 JSON 结构，经加密后存储在 `data/players/` 目录下一个与用户名对应的 `.dat` 文件中。
*   **CRUD API**: 系统提供了一套完整的 RESTful API，用于对指定用户的武器进行增 (`POST /api/weapons`)、删 (`DELETE /api/weapons/<name>`)、改 (`PUT /api/weapons/<name>`)、查 (`GET /api/weapons`) 操作。API 通过 `X-Username` 请求头来识别当前操作的用户。

### 2.3 安全机制

*   **全数据加密**: 系统的核心安全特性是**对所有持久化的敏感数据进行加密**。不仅是密码，用户名和整个武器库 JSON 结构都使用 **AES (GCM 模式)** 进行加密后才写入磁盘。这确保了即使数据文件被泄露，其内容也无法被轻易解读。
*   **文件混淆**: 所有加密数据文件均使用 `.dat` 作为后缀，以避免被直接识别为可读的 JSON 或文本文件。

### 2.4 前端实现

*   **UI/UX**: 采用现代化的深色主题，界面元素通过 CSS 进行了精细的美化，并为交互操作添加了平滑的过渡动画。
*   **响应式设计**: 武器库主界面采用 `Grid` 布局，并能在移动端设备上自动转换为更易于浏览的卡片式布局。
*   **国际化 (I18n)**: 前端内置了中英文双语支持，可通过 UI 切换，并将用户的语言偏好存储在 `localStorage` 中。
*   **高级交互**:
    *   **表头排序**: 点击表格头可对数据进行升/降序排列，指示器不会引起布局跳动。
    *   **标签选择器**: 在编辑/添加武器时，使用自定义的标签式选择器来设置开火模式。
    *   **自定义模态框**: 所有弹窗（包括删除确认）均使用与整体 UI 风格一致的自定义模态框。

## 3. 运行指南

1.  **环境准备**: 确保已在 `pubg` anaconda 虚拟环境中，并已通过 `pip install -r requirements.sample.txt` 安装所有依赖。
2.  **启动服务**: 在项目根目录下运行 `python3 run.py`。
3.  **访问应用**: 打开浏览器并访问 `http://127.0.0.1:5000/`。