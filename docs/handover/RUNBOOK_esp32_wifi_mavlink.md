# ESP32 Wi-Fi 配网 → Pixhawk MAVLink → Orin

**目标链路**：`Pixhawk TELEM` ↔ UART ↔ `ESP32` ↔ Wi-Fi ↔ `Orin`（UDP 14550）

**推荐固件**：[DroneBridge for ESP32](https://github.com/DroneBridge/ESP32)（MAVLink 透传、Web 配网、QGC/Orin 兼容）

---

## 0. 硬件（Pixhawk 侧稍后接，先配 Wi-Fi）

| Pixhawk TELEM | ESP32 |
|---------------|-------|
| TX | RX |
| RX | TX |
| GND | GND |

- 电平 **3.3V**（Pixhawk TELEM 为 3.3V TTL，勿接 5V）
- Pixhawk 用独立供电；ESP32 用 USB 或稳压 5V/3.3V
- **配 Wi-Fi 时**：ESP32 仅 USB 接 Mac 即可，无需接 Pixhawk

---

## 1. Mac 上确认 ESP32 串口

```bash
ls /dev/cu.usb* /dev/cu.wchusb* /dev/cu.SLAB* 2>/dev/null
./experiments/aerial/scripts/esp32_wifi_setup.sh detect
```

常见端口名：`cu.usbserial-*`、`cu.wchusbserial*`、`cu.SLAB_USBtoUART`  
**不要**选 Pixhawk 的 `cu.usbmodem11201`（那是飞控 USB）。

---

## 2. 固件：DroneBridge（若尚未烧录）

1. 下载 [Releases → Stable `.bin`](https://github.com/DroneBridge/ESP32/releases)（选与你板子匹配的，如 `DroneBridge_ESP32_v*.bin`）
2. 使用 [ESP Web Flasher](https://espressif.github.io/esptool-js/) 或 `esptool.py`：
   ```bash
   pip3 install esptool
   esptool.py --chip esp32 --port /dev/cu.usbserial-XXXX erase_flash
   esptool.py --chip esp32 --port /dev/cu.usbserial-XXXX write_flash 0x1000 DroneBridge_ESP32_*.bin
   ```
3. 断电重插 ESP32

---

## 3. 配 Wi-Fi（核心）

### 3.1 出厂默认：ESP 开热点（AP）

上电后 Mac/手机连接：

| 项 | 默认值 |
|----|--------|
| SSID | `DroneBridge for ESP32` |
| 密码 | `dronebridge` |
| ESP IP | `192.168.2.1` |

浏览器打开：**http://192.168.2.1** → Settings / Wi-Fi

### 3.2 推荐：改为 **Wi-Fi Client**，加入 Orin 同一局域网

在 Web 界面：

1. **Mode** → `WiFi Client`（连现有路由器/Orin 热点）
2. **SSID** → Orin 所在 Wi-Fi 名称（例如实验室路由器或 Orin 开的 AP）
3. **Password** → 对应密码
4. **UART**：`MAVLink`，**Baud** `115200`（与 Pixhawk `SERIALx_BAUD` 一致）
5. Save → **Reboot**

重启后 ESP 从路由器 DHCP 拿 IP；在路由器管理页或串口日志里查看 ESP 的 IP（例如 `10.229.20.xxx`）。

### 3.3 若暂时只有 Orin 直连、无路由器

可先用 **AP 模式**调试：Mac 连 ESP 热点，确认 Web 与 MAVLink 正常；Orin 用 USB 网卡/以太网与 Mac 同网段后再改 Client。

---

## 4. Orin 侧预置（Wi-Fi 通后再做）

ESP 与 Orin **同一子网** 后，在 Orin 上：

```bash
# 监听 ESP 转发的 MAVLink（ESP 会向发过 UDP 包的客户端回传）
python3 -m experiments.aerial.scripts.orin_fc_monitor --port udpin:0.0.0.0:14550
```

或：

```bash
MAVLINK_URL=udpin:0.0.0.0:14550 python3 -m experiments.aerial.scripts.wam_vgoal_deploy --mock-camera ...
```

**注意（DroneBridge UDP）**：Orin 需先向 ESP 的 `14550` **发一包**（如 heartbeat），ESP 才会把飞控数据回传到该 IP。`orin_gcs_bridge.sh` 的 MAVProxy 出站即满足此要求。

手动登记 Orin 为 UDP 目标（Web API，在连 ESP AP 时）：

```bash
curl -X POST http://192.168.2.1/api/settings/clients/udp \
  -H 'Content-Type: application/json' \
  -d '{"ip":"10.229.20.127","port":14550,"save":true}'
```

（`ip` 换成 Orin 实际 IP。）

---

## 5. Pixhawk 串口参数（接 TELEM 后）

接 **TELEM2** 示例：

| 参数 | 值 |
|------|-----|
| `SERIAL2_PROTOCOL` | `2`（MAVLink2） |
| `SERIAL2_BAUD` | `115`（115200） |
| `BRD_SER2_RTSCTS` | `0` |

当前 USB 直连 Mac 时飞控为 **ArduPilot 4.4.4 / STABILIZE**；改 TELEM 后 USB 仍可作备用。

---

## 6. 验收清单

- [ ] Mac 能打开 `http://192.168.2.1`（AP）或 ping ESP Client IP
- [ ] Web 里 Wi-Fi Client 已连上，有 IP
- [ ] Pixhawk TELEM ↔ ESP UART 接通，参数 115200 MAVLink2
- [ ] Orin `orin_fc_monitor --port udpin:0.0.0.0:14550` 收到 heartbeat
- [ ] 仅 **一个** 进程占用 MAVLink（不要 USB 与 Wi-Fi 同时在 Orin 上开两份桥）

---

## 7. 故障排查

| 现象 | 处理 |
|------|------|
| Mac 看不到 ESP 串口 | 换 USB 线/口；装 CH340/CP210x 驱动 |
| 搜不到 `DroneBridge` 热点 | 短按板载 RST；重烧固件 |
| Orin 无 heartbeat | 确认同网段；Orin 先 `nc -u ESP_IP 14550` 发一包；查 API 添加 UDP client |
| 数据乱码 | 统一波特率 115200；TX/RX 是否交叉 |
