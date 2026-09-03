# 志愿者标注系统运维手册

## 环境信息

| 项目 | 值 |
|------|----|
| 公网地址 | http://49.233.81.226:8077/ |
| 本机监听 | 127.0.0.1:8077 |
| conda 环境 | `hocap-annotation` |
| 公网服务器 | ubuntu@49.233.81.226 |
| SSH Key | `/home/ruoqu/.ssh/robotool.pem` |

## 生产路径

| 变量 | 路径 |
|------|------|
| `BUNDLE_DIR` | `/data/robotool/_va_bundle_v2` |
| `PROMPTS_DIR` | `/data/robotool/_va_bundle_v2_prompts` |
| `DB_PATH` | `/data/robotool/_va_bundle_v2/tasks.db` |
| 代码目录 | `HO-Cap-Annotation/volunteer_annotation/cloud/` |
| systemd 服务 | `/etc/systemd/system/va-uvicorn.service` |
|              | `/etc/systemd/system/va-autossh.service` |

数据规模：4068 个实验，每个实验 8 视角。

---

## 日常操作（systemd）

```bash
# 查看状态
sudo systemctl status va-uvicorn va-autossh

# 重启
sudo systemctl restart va-uvicorn va-autossh

# 停止
sudo systemctl stop va-uvicorn va-autossh

# 查看日志
journalctl -u va-uvicorn -n 50 -f
journalctl -u va-autossh -n 50 -f

# 开机自启（首次配置后无需重复）
sudo systemctl enable va-uvicorn va-autossh
```

## 健康检查

```bash
# 本地
curl http://127.0.0.1:8077/api/stats

# 公网
curl http://49.233.81.226:8077/api/stats

# 进程
pgrep -af "uvicorn app:app"
pgrep -af "autossh.*8077"
```

---

## 手动启动（应急，systemd 不可用时）

### 1. uvicorn（SAM2 推理服务）

```bash
cd /home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/volunteer_annotation/cloud

BUNDLE_DIR=/data/robotool/_va_bundle_v2 \
PROMPTS_DIR=/data/robotool/_va_bundle_v2_prompts \
DB_PATH=/data/robotool/_va_bundle_v2/tasks.db \
nohup /home/ruoqu/miniconda3/envs/hocap-annotation/bin/uvicorn app:app \
  --host 127.0.0.1 --port 8077 --workers 1 \
  > /tmp/uvicorn_va.log 2>&1 &

# 查看日志
tail -f /tmp/uvicorn_va.log
```

### 2. autossh 反向隧道（暴露到公网服务器）

```bash
nohup /usr/lib/autossh/autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no \
  -i /home/ruoqu/.ssh/robotool.pem \
  -R 8077:localhost:8077 ubuntu@49.233.81.226 \
  > /tmp/autossh_8077.log 2>&1 &

# 查看日志
tail -f /tmp/autossh_8077.log
```

### 3. SSH port 22 隧道（远程访问本机，通常已有进程维持）

```bash
# 检查是否在跑
pgrep -af "autossh.*6022"

# 如果没有，重启
nohup /usr/lib/autossh/autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  -i /home/ruoqu/.ssh/robotool.pem \
  -R 6022:localhost:22 ubuntu@49.233.81.226 \
  > /tmp/autossh_6022.log 2>&1 &
```

---

## systemd 服务文件（重装时复制）

### /etc/systemd/system/va-uvicorn.service

```ini
[Unit]
Description=Volunteer Annotation SAM2 Server (uvicorn)
After=network.target
Wants=network.target

[Service]
Type=simple
User=ruoqu
WorkingDirectory=/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/volunteer_annotation/cloud
Environment="BUNDLE_DIR=/data/robotool/_va_bundle_v2"
Environment="PROMPTS_DIR=/data/robotool/_va_bundle_v2_prompts"
Environment="DB_PATH=/data/robotool/_va_bundle_v2/tasks.db"
ExecStart=/home/ruoqu/miniconda3/envs/hocap-annotation/bin/uvicorn app:app --host 127.0.0.1 --port 8077 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### /etc/systemd/system/va-autossh.service

```ini
[Unit]
Description=Volunteer Annotation SSH Reverse Tunnel (port 8077)
After=network.target va-uvicorn.service
Wants=network.target

[Service]
Type=simple
User=ruoqu
Environment="AUTOSSH_GATETIME=0"
ExecStart=/usr/lib/autossh/autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=no -i /home/ruoqu/.ssh/robotool.pem -R 8077:localhost:8077 ubuntu@49.233.81.226
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

安装命令：
```bash
sudo cp /tmp/va-uvicorn.service /etc/systemd/system/
sudo cp /tmp/va-autossh.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable va-uvicorn va-autossh
sudo systemctl start va-uvicorn va-autossh
```

---

## 迁移到新设备

### 1. 安装依赖

```bash
# conda 环境
conda create -n hocap-annotation python=3.10
conda activate hocap-annotation
cd HO-Cap-Annotation && bash scripts/install_*.sh

# autossh
sudo apt install autossh
```

### 2. 同步数据

```bash
# Bundle（embedding + 关键帧图片，较大）
rsync -av --progress /data/robotool/_va_bundle_v2/ NEW_HOST:/data/robotool/_va_bundle_v2/

# 已提交的标注结果
rsync -av --progress /data/robotool/_va_bundle_v2_prompts/ NEW_HOST:/data/robotool/_va_bundle_v2_prompts/

# 代码
rsync -av /home/ruoqu/crq_ws/robotool/ NEW_HOST:/home/ruoqu/crq_ws/robotool/

# SSH key
scp /home/ruoqu/.ssh/robotool.pem NEW_HOST:/home/ruoqu/.ssh/robotool.pem
ssh NEW_HOST "chmod 600 /home/ruoqu/.ssh/robotool.pem"
```

### 3. 安装 systemd 服务

参考上方"systemd 服务文件"章节，写入文件后执行安装命令。

### 4. 验证

```bash
curl http://127.0.0.1:8077/api/stats
curl http://49.233.81.226:8077/api/stats
```

---

## 标注统计查询

```bash
# 各状态数量
sqlite3 /data/robotool/_va_bundle_v2/tasks.db \
  "SELECT status, COUNT(*) FROM tasks GROUP BY status;"

# 各标注者提交数和平均时长
sqlite3 /data/robotool/_va_bundle_v2/tasks.db "
SELECT annotator_id, COUNT(*) as total,
  ROUND(AVG(annotation_time_sec)/60, 1) as avg_min
FROM tasks WHERE status='submitted'
GROUP BY annotator_id ORDER BY total DESC;"

# API 查询（需服务在线）
curl "http://127.0.0.1:8077/api/annotator_stats/NAME%23PHONE"
```
