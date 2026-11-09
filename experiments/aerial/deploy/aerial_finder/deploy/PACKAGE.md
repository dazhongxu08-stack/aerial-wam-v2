# aerial_finder 产品包

可部署到 Jetson Orin 的找物服务：对外只暴露 HTTP。

## 检测器（默认：开集 YOLO-World）

| 文件 | 作用 |
|------|------|
| `weights/detector/yolov8s-worldv2.pt` | **开集**检测：指令里的物体名 → text prompt（凳子/灭火器/人…） |
| `weights/clip/ViT-B-32.pt` | 离线 CLIP 文本塔（建议打进包） |
| `weights/detector/model.torchscript` | 可选：窄词表凳子 TS（见 `config/product_dog_stool_ts.json`） |

默认 `config.json` 使用 `detector.type = yolo_world`，**不绑死 8 类**；中文指令由 `mission/parser.py` 解析成 prompts。

切换回窄 TS：

```bash
sudo cp /opt/aerial_finder/config/product_dog_stool_ts.json /opt/aerial_finder/config.json
sudo systemctl restart aerial-finder
```

## 打包

```bash
cd experiments/aerial/deploy/aerial_finder/deploy
chmod +x *.sh bin/aerial-finder

# 1) 下载开集权重（需网络，一次即可）
./fetch_detector_weights.sh

# 2) 打 tar
./pack_release.sh --out ~/Desktop/aerial-finder-release --version 0.1.1
```

权重默认从 `artifacts/tello_gs/models/` 读取；也可用：

```bash
YOLO_WORLD_PT=/path/to/yolov8s-worldv2.pt CLIP_PT=/path/to/ViT-B-32.pt ./pack_release.sh ...
```

## Orin 安装

```bash
sudo tar -C /opt -xzf aerial-finder-0.1.1-orin-aarch64.tar.gz
sudo /opt/aerial_finder/deploy/install.sh
sudo systemctl enable --now aerial-finder
curl -s http://127.0.0.1:8088/api/v1/health
/opt/aerial_finder/bin/aerial-finder parse '正前方10米找一个红色的凳子'
```

前置：JetPack CUDA `torch`、相机、`bridge :8092`。工作目录必须是包根（systemd 已设），以便加载 `./weights/clip`。

## API

| 方法 | 路径 |
|------|------|
| GET | `/api/v1/health` |
| POST | `/api/v1/parse` `{"instruction":"..."}` |
| POST | `/api/v1/missions` |
| GET | `/api/v1/missions/current` |
| POST | `/api/v1/missions/current/abort` |
