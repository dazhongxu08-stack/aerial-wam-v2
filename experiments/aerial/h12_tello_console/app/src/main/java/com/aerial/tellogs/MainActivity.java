package com.aerial.tellogs;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import com.skydroid.rcsdk.KeyManager;
import com.skydroid.rcsdk.RCSDKManager;
import com.skydroid.rcsdk.SDKManagerCallBack;
import com.skydroid.rcsdk.common.callback.CompletionCallbackWith;
import com.skydroid.rcsdk.common.error.SkyException;
import com.skydroid.rcsdk.key.RemoteControllerKey;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * H12 native ground station: RCSDK channels → Tello /api/rc, Orin JPEG → ImageView.
 * Close H12助手 / Skydroid Fly before starting (port conflict).
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "TelloGS";
    private static final String PREFS = "tello_gs";
    private static final String KEY_ORIN_URL = "orin_url";
    /**
     * H12 USB adb reverse → Mac tello_ground_console :8090.
     * Orin is Ethernet-only (not the phone hotspot).
     */
    private static final String DEFAULT_ORIN_URL = "http://127.0.0.1:8090";
    private static final long CHANNEL_MS = 80L;
    private static final long FRAME_MS = 120L;
    private static final long STATUS_MS = 1000L;
    private static final long UI_CH_MS = 200L;

    private ImageView cam;
    private TextView tvStatus;
    private TextView tvLink;
    private TextView tvChannels;
    private TextView tvRc;
    private EditText etOrinUrl;
    private Button btnReconnect;
    private Button btnApplyUrl;
    private Button btnGroundLock;
    private Button btnPilotMode;
    private Button btnRecord;
    private Button btnStopRecord;
    private TextView tvRecord;
    private boolean recording = false;
    private boolean rcSdkInited = false;
    private boolean reconnectBusy = false;
    private boolean allowFly = false;
    /** manual = H12 sticks; auto = Orin policy owns RC. */
    private volatile String controlMode = "manual";

    private TelloHttp http;
    private String orinBase = DEFAULT_ORIN_URL;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean channelInFlight = new AtomicBoolean(false);
    private final AtomicBoolean frameInFlight = new AtomicBoolean(false);
    private final AtomicReference<int[]> latestCh = new AtomicReference<>(null);

    private Thread channelThread;
    private volatile boolean channelStop = false;
    /** 三段拨杆上一拍位置（边沿按 低/中/高 切换触发）。 */
    private ChannelMapper.TriPos ch5Prev = ChannelMapper.TriPos.UNKNOWN;
    private ChannelMapper.TriPos ch6Prev = ChannelMapper.TriPos.UNKNOWN;
    private int lastA, lastB, lastC, lastD;
    private long lastRcSentAt = 0L;
    private long lastUiChAt = 0L;

    private final Runnable frameTick = new Runnable() {
        @Override
        public void run() {
            if (!running.get()) return;
            if (frameInFlight.compareAndSet(false, true)) {
                http.fetchSnapshot((bmp, err) -> {
                    try {
                        if (bmp != null && cam != null) {
                            cam.setImageBitmap(bmp);
                        }
                    } finally {
                        frameInFlight.set(false);
                    }
                });
            }
            main.postDelayed(this, FRAME_MS);
        }
    };

    private final Runnable statusTick = new Runnable() {
        @Override
        public void run() {
            if (!running.get()) return;
            http.fetchStatus((st, err) -> {
                if (tvStatus == null) return;
                if (st != null) {
                    Object age = st.opt("frame_age_s");
                    String stale = "";
                    if (age instanceof Number && ((Number) age).doubleValue() > 3.0) {
                        stale = " STALE";
                    }
                    allowFly = st.optBoolean("allow_fly", false);
                    String cm = st.optString("control_mode", controlMode);
                    if ("auto".equals(cm) || "manual".equals(cm)) {
                        controlMode = cm;
                    }
                    updateGroundLockButton();
                    updatePilotModeButton();
                    updateRecordUi(st.optJSONObject("recording"));
                    tvStatus.setText(String.format(
                            "ok=%s fly=%s mode=%s bat=%s temp=%s frame=%s age=%s%s err=%s",
                            st.opt("ok"), allowFly, controlMode, st.opt("battery"), st.opt("temp"),
                            st.opt("frame_seq"), age, stale, st.optString("error", "")));
                    updateLinkText(st, null);
                } else {
                    tvStatus.setText("Mac fail: " + (err != null ? err.getMessage() : "?")
                            + " @ " + orinBase);
                    updateLinkText(null, err);
                }
            });
            main.postDelayed(this, STATUS_MS);
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        System.setProperty("http.keepAlive", "false");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        cam = findViewById(R.id.cam);
        tvStatus = findViewById(R.id.tvStatus);
        tvLink = findViewById(R.id.tvLink);
        tvChannels = findViewById(R.id.tvChannels);
        tvRc = findViewById(R.id.tvRc);
        etOrinUrl = findViewById(R.id.etOrinUrl);
        btnReconnect = findViewById(R.id.btnReconnect);
        btnApplyUrl = findViewById(R.id.btnApplyUrl);
        btnGroundLock = findViewById(R.id.btnGroundLock);
        btnPilotMode = findViewById(R.id.btnPilotMode);
        btnRecord = findViewById(R.id.btnRecord);
        btnStopRecord = findViewById(R.id.btnStopRecord);
        tvRecord = findViewById(R.id.tvRecord);
        btnReconnect.setOnClickListener(v -> reconnectAll());
        btnApplyUrl.setOnClickListener(v -> applyOrinUrlFromUi(true));
        btnGroundLock.setOnClickListener(v -> toggleGroundLock());
        btnPilotMode.setOnClickListener(v -> togglePilotMode());
        btnRecord.setOnClickListener(v -> setRecording(true));
        btnStopRecord.setOnClickListener(v -> setRecording(false));
        updateRecordUi(null);
        updateGroundLockButton();
        updatePilotModeButton();
        etOrinUrl.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                applyOrinUrlFromUi(true);
                return true;
            }
            return false;
        });

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String base = getIntent().getStringExtra("base_url");
        if (base == null || base.isEmpty()) {
            base = prefs.getString(KEY_ORIN_URL, DEFAULT_ORIN_URL);
        }
        // Keep 127.0.0.1 (adb reverse → Mac). Drop office LAN and Orin hotspot URLs.
        if (base != null && (base.contains("192.168.1.19") || base.contains("172.20.10.3"))) {
            base = DEFAULT_ORIN_URL;
        }
        orinBase = normalizeBaseUrl(base);
        etOrinUrl.setText(orinBase);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_ORIN_URL, orinBase).apply();
        http = new TelloHttp(orinBase);
        Log.i(TAG, "Orin base=" + orinBase);

        initRcSdk();
    }

    private static String normalizeBaseUrl(String raw) {
        String b = raw == null ? DEFAULT_ORIN_URL : raw.trim();
        if (b.isEmpty()) b = DEFAULT_ORIN_URL;
        if (!b.startsWith("http://") && !b.startsWith("https://")) {
            b = "http://" + b;
        }
        while (b.endsWith("/")) b = b.substring(0, b.length() - 1);
        return b;
    }

    private void applyOrinUrlFromUi(boolean toast) {
        String next = normalizeBaseUrl(etOrinUrl.getText() != null
                ? etOrinUrl.getText().toString() : "");
        orinBase = next;
        etOrinUrl.setText(next);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_ORIN_URL, next).apply();
        if (http != null) {
            http.shutdown();
        }
        http = new TelloHttp(next);
        if (toast) {
            Toast.makeText(this, "Orin → " + next, Toast.LENGTH_SHORT).show();
        }
        tvStatus.setText("switching Orin… " + next);
    }

    private void updateLinkText(@Nullable JSONObject st, @Nullable Exception err) {
        if (tvLink == null) return;
        if (st == null) {
            tvLink.setText("Tello→Mac 无链路");
            tvLink.setTextColor(0xFFFF5252);
            return;
        }
        if (st.isNull("wifi_rssi_dbm") || !st.has("wifi_rssi_dbm")) {
            tvLink.setText("Tello→Mac —");
            tvLink.setTextColor(0xFFFFD54F);
            return;
        }
        int rssi = st.optInt("wifi_rssi_dbm", 0);
        int noise = st.isNull("wifi_noise_dbm") ? 0 : st.optInt("wifi_noise_dbm", 0);
        String snrPart = "";
        if (!st.isNull("tello_wifi_snr") && st.has("tello_wifi_snr")) {
            snrPart = "  TelloSNR " + st.optInt("tello_wifi_snr");
        }
        String qual;
        int color;
        if (rssi >= -50) {
            qual = "强";
            color = 0xFF69F0AE;
        } else if (rssi >= -60) {
            qual = "中";
            color = 0xFFB2FF59;
        } else if (rssi >= -70) {
            qual = "弱";
            color = 0xFFFFD54F;
        } else {
            qual = "很弱";
            color = 0xFFFF5252;
        }
        String noisePart = noise != 0 ? (" / " + noise + "dBm") : "";
        String tempPart = "";
        if (!st.isNull("temp") && st.has("temp")) {
            int temp = st.optInt("temp");
            tempPart = "  温度 " + temp + "℃";
            if (temp >= 90) {
                tempPart += " 过热";
                color = 0xFFFF5252;
            } else if (temp >= 85 && color != 0xFFFF5252) {
                color = 0xFFFFD54F;
            }
        }
        tvLink.setText("Tello→Mac " + rssi + "dBm" + noisePart + "  " + qual + snrPart + tempPart);
        tvLink.setTextColor(color);
    }

    private void updateGroundLockButton() {
        if (btnGroundLock == null) return;
        if (allowFly) {
            btnGroundLock.setText("地面锁 OFF");
            btnGroundLock.setBackgroundColor(0xCC2E7D32); // green = can fly
        } else {
            btnGroundLock.setText("地面锁 ON");
            btnGroundLock.setBackgroundColor(0xCCB71C1C); // red = locked
        }
    }

    private void updatePilotModeButton() {
        if (btnPilotMode == null) return;
        if ("auto".equals(controlMode)) {
            btnPilotMode.setText("Orin自动");
            btnPilotMode.setBackgroundColor(0xCC6A1B9A); // purple = Orin owns sticks
        } else {
            btnPilotMode.setText("手动");
            btnPilotMode.setBackgroundColor(0xCC2E7D32); // green = H12 sticks
        }
    }

    private void togglePilotMode() {
        if (http == null) return;
        btnPilotMode.setEnabled(false);
        // Zero sticks before handoff (Orin also clears on mode change)
        http.postRc(0, 0, 0, 0);
        lastA = lastB = lastC = lastD = 0;
        http.postControlMode(null, (ok, msg) -> {
            btnPilotMode.setEnabled(true);
            if (!ok) {
                Toast.makeText(this, "切控失败: " + msg, Toast.LENGTH_LONG).show();
                return;
            }
            if ("auto".equals(msg) || "manual".equals(msg)) {
                controlMode = msg;
            } else if (msg != null && msg.contains("control_mode=auto")) {
                controlMode = "auto";
            } else if (msg != null && msg.contains("control_mode=manual")) {
                controlMode = "manual";
            } else {
                // toggle locally as fallback; status poll will correct
                controlMode = "auto".equals(controlMode) ? "manual" : "auto";
            }
            updatePilotModeButton();
            Toast.makeText(this,
                    "auto".equals(controlMode) ? "已切 Orin 自动控制" : "已切 H12 手动",
                    Toast.LENGTH_SHORT).show();
        });
    }

    private void updateRecordUi(@Nullable JSONObject rec) {
        if (tvRecord == null || btnRecord == null || btnStopRecord == null) return;
        boolean on = rec != null && rec.optBoolean("on", false);
        recording = on;
        btnRecord.setEnabled(!on);
        btnStopRecord.setEnabled(on);
        if (!on) {
            tvRecord.setText("未录制");
            tvRecord.setTextColor(0xFFFFAB91);
            return;
        }
        tvRecord.setText(String.format(
                "REC %s  帧=%s  样点=%s  %.0fs",
                rec.optString("run_id", "?"),
                rec.opt("n_frames"),
                rec.opt("n_samples"),
                rec.optDouble("duration_s", 0)));
        tvRecord.setTextColor(0xFFFF5252);
    }

    private void setRecording(boolean on) {
        if (http == null) return;
        btnRecord.setEnabled(false);
        btnStopRecord.setEnabled(false);
        http.postRecord(on, (ok, msg) -> {
            if (!ok) {
                Toast.makeText(this, (on ? "开始录制失败: " : "停止录制失败: ") + msg,
                        Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, msg, Toast.LENGTH_SHORT).show();
            }
            http.fetchStatus((st, err) -> {
                if (st != null) {
                    updateRecordUi(st.optJSONObject("recording"));
                } else {
                    updateRecordUi(null);
                }
            });
        });
    }

    private void toggleGroundLock() {
        if (http == null) return;
        btnGroundLock.setEnabled(false);
        http.postAllowFly(null, (ok, msg) -> {
            btnGroundLock.setEnabled(true);
            if (!ok) {
                Toast.makeText(this, "切换失败: " + msg, Toast.LENGTH_LONG).show();
                return;
            }
            // Refresh from /api/status for authoritative allow_fly
            http.fetchStatus((st, err) -> {
                if (st != null) {
                    allowFly = st.optBoolean("allow_fly", false);
                    updateGroundLockButton();
                }
                Toast.makeText(this,
                        allowFly ? "已解锁起飞/打杆" : "已上地面锁（起飞/打杆禁用）",
                        Toast.LENGTH_SHORT).show();
            });
        });
    }

    /**
     * 重启控制台：H12 先释放可能抢占的 TELLO WiFi，再让 Orin 重启 Tello 服务并重连飞机。
     * Orin 侧 POST /api/reconnect → 关闭 SDK/图传 → 重连 TELLO AP → 重新 boot。
     */
    private void reconnectAll() {
        if (reconnectBusy) {
            return;
        }
        reconnectBusy = true;
        btnReconnect.setEnabled(false);
        btnReconnect.setText("重启中…");
        tvStatus.setText("释放 H12 Tello WiFi…");

        http.postRc(0, 0, 0, 0);
        ch5Prev = ChannelMapper.TriPos.UNKNOWN;
        ch6Prev = ChannelMapper.TriPos.UNKNOWN;
        lastA = lastB = lastC = lastD = 0;
        lastRcSentAt = 0L;
        channelInFlight.set(false);

        // H12 若连上 TELLO-* 会抢走唯一客户端，Orin SDK 必失败
        new Thread(() -> {
            String wifiMsg = releaseTelloWifi();
            main.post(() -> tvStatus.setText(
                    (wifiMsg.isEmpty() ? "H12 WiFi ok" : wifiMsg) + " · Orin重启Tello服务…"));

            http.postReconnect((orinOk, orinMsg) -> {
                tvStatus.setText(orinOk
                        ? ("Orin: " + orinMsg + " · 等待Tello…")
                        : ("Orin重启失败: " + orinMsg + " · 仍等待…"));
                pollOrinAfterReconnect(0);
            });
        }, "h12-restart-console").start();
    }

    /** Forget/disconnect any TELLO-* WiFi so Orin's USB radio can own the AP. */
    private String releaseTelloWifi() {
        try {
            android.net.wifi.WifiManager wm =
                    (android.net.wifi.WifiManager) getApplicationContext()
                            .getSystemService(Context.WIFI_SERVICE);
            if (wm == null) {
                return "no WifiManager";
            }
            android.net.wifi.WifiInfo info = wm.getConnectionInfo();
            String cur = info != null ? info.getSSID() : null;
            if (cur != null) {
                cur = cur.replace("\"", "");
            }
            boolean wasOnTello = cur != null && cur.toUpperCase().startsWith("TELLO");
            int removed = 0;
            java.util.List<android.net.wifi.WifiConfiguration> configs = wm.getConfiguredNetworks();
            if (configs != null) {
                for (android.net.wifi.WifiConfiguration c : configs) {
                    String ssid = c.SSID == null ? "" : c.SSID.replace("\"", "");
                    if (ssid.toUpperCase().startsWith("TELLO")) {
                        try {
                            wm.removeNetwork(c.networkId);
                            removed++;
                        } catch (Throwable t) {
                            Log.w(TAG, "removeNetwork " + ssid + ": " + t.getMessage());
                        }
                    }
                }
                if (removed > 0) {
                    wm.saveConfiguration();
                }
            }
            if (wasOnTello) {
                try {
                    wm.disconnect();
                } catch (Throwable ignored) {
                }
            }
            Log.i(TAG, "releaseTelloWifi cur=" + cur + " removed=" + removed);
            if (wasOnTello || removed > 0) {
                return "已断开/忘记 Tello WiFi(" + removed + ")";
            }
            return "";
        } catch (Throwable t) {
            Log.w(TAG, "releaseTelloWifi: " + t.getMessage());
            return "WiFi释放失败: " + t.getMessage();
        }
    }

    private void pollOrinAfterReconnect(int attempt) {
        http.fetchStatus((st, err) -> {
            if (st != null) {
                Object ageObj = st.opt("frame_age_s");
                double age = (ageObj instanceof Number) ? ((Number) ageObj).doubleValue() : 999;
                boolean ok = st.optBoolean("ok", false);
                int seq = st.optInt("frame_seq", 0);
                tvStatus.setText(String.format(
                        "reconnect · ok=%s bat=%s frame=%s age=%s",
                        ok, st.opt("battery"), seq, ageObj));
                if (ok && age < 3.0 && seq > 0) {
                    Toast.makeText(this, "控制台已重启，Tello已重连", Toast.LENGTH_SHORT).show();
                    resetReconnectButton();
                    return;
                }
            } else {
                tvStatus.setText("等待 Orin… " + (err != null ? err.getMessage() : ""));
            }
            if (attempt >= 45) {
                String linkErr = (st != null) ? String.valueOf(st.opt("error")) : "";
                Toast.makeText(this,
                        "重启超时。确认飞机已开机且 H12 未连 TELLO WiFi。 " + linkErr,
                        Toast.LENGTH_LONG).show();
                resetReconnectButton();
                return;
            }
            main.postDelayed(() -> pollOrinAfterReconnect(attempt + 1), 1000);
        });
    }

    private void resetReconnectButton() {
        if (btnReconnect != null) {
            btnReconnect.setText("重启控制台");
            btnReconnect.setEnabled(true);
        }
        reconnectBusy = false;
    }

    private void initRcSdk() {
        if (rcSdkInited) {
            RCSDKManager.INSTANCE.connectToRC();
            return;
        }
        rcSdkInited = true;
        // 关键勿主线程回调：KeyChannels 高频会堵死 UI（此前卡死主因）
        RCSDKManager.INSTANCE.setMainThreadCallBack(false);
        RCSDKManager.INSTANCE.initSDK(this, new SDKManagerCallBack() {
            @Override
            public void onRcConnected() {
                Log.i(TAG, "RC connected type=" + RCSDKManager.INSTANCE.getDeviceType());
                // 无接收机：关 RF 模块 + 停止对频，减轻失联滴滴
                new Thread(MainActivity.this::silenceNoReceiverMode, "h12-silence-rf").start();
                main.post(() -> {
                    muteAndroidAlerts();
                    Toast.makeText(MainActivity.this,
                            "H12 RC ok · 已尝试关射频/静音（无接收机模式）",
                            Toast.LENGTH_LONG).show();
                    if (btnReconnect != null) {
                        btnReconnect.setEnabled(true);
                        btnReconnect.setText("重启控制台");
                    }
                    reconnectBusy = false;
                });
                startChannelPolling();
            }

            @Override
            public void onRcConnectFail(@Nullable SkyException e) {
                Log.e(TAG, "RC connect fail: " + e);
                main.post(() -> {
                    if (tvChannels != null) {
                        tvChannels.setText("RCSDK fail: " + e + " — 关助手后点重启控制台");
                    }
                    if (btnReconnect != null) {
                        btnReconnect.setEnabled(true);
                        btnReconnect.setText("重启控制台");
                    }
                    reconnectBusy = false;
                });
            }

            @Override
            public void onRcDisconnect() {
                Log.w(TAG, "RC disconnected");
            }
        });
        RCSDKManager.INSTANCE.connectToRC();
    }

    /**
     * 无 R12 接收机时：关掉射频模块并停止对频，避免持续失联滴滴。
     * （官方说明：未连接收机会一直报警；关模块是软件侧最有效手段之一）
     */
    private void silenceNoReceiverMode() {
        com.skydroid.rcsdk.common.callback.CompletionCallback cb =
                new com.skydroid.rcsdk.common.callback.CompletionCallback() {
                    @Override
                    public void onResult(SkyException e) {
                        Log.i(TAG, "silenceNoReceiver result: " + e);
                    }
                };
        try {
            KeyManager.INSTANCE.set(
                    RemoteControllerKey.INSTANCE.getKeyModule915Enable(), Boolean.FALSE, cb);
        } catch (Throwable t) {
            Log.w(TAG, "Module915Enable: " + t.getMessage());
        }
        try {
            KeyManager.INSTANCE.set(
                    RemoteControllerKey.INSTANCE.getKeyModuleGLinkEnable(), Boolean.FALSE, cb);
        } catch (Throwable t) {
            Log.w(TAG, "ModuleGLinkEnable: " + t.getMessage());
        }
        try {
            KeyManager.INSTANCE.action(
                    RemoteControllerKey.INSTANCE.getKeyRequestStopPair(), cb);
        } catch (Throwable t) {
            Log.w(TAG, "StopPair: " + t.getMessage());
        }
        // 再静音一次（部分机型蜂鸣走 Android 音量）
        main.post(this::muteAndroidAlerts);
    }

    private void muteAndroidAlerts() {
        android.media.AudioManager am =
                (android.media.AudioManager) getSystemService(Context.AUDIO_SERVICE);
        if (am == null) return;
        int[] streams = {
                android.media.AudioManager.STREAM_MUSIC,
                android.media.AudioManager.STREAM_ALARM,
                android.media.AudioManager.STREAM_NOTIFICATION,
                android.media.AudioManager.STREAM_SYSTEM,
                android.media.AudioManager.STREAM_RING,
        };
        for (int s : streams) {
            try {
                am.setStreamVolume(s, 0, 0);
            } catch (Throwable t) {
                Log.w(TAG, "mute stream " + s + ": " + t.getMessage());
            }
        }
        // H12 无 DND 权限时会抛异常，单独吞掉，不影响上面静音
        try {
            am.setRingerMode(android.media.AudioManager.RINGER_MODE_SILENT);
        } catch (Throwable ignored) {
        }
        try {
            KeyManager.INSTANCE.set(
                    RemoteControllerKey.INSTANCE.getKeyButtonTone(),
                    Boolean.FALSE,
                    e -> Log.i(TAG, "ButtonTone off: " + e));
        } catch (Throwable t) {
            Log.w(TAG, "ButtonTone: " + t.getMessage());
        }
        Log.i(TAG, "Android streams muted");
    }

    private void startChannelPolling() {
        if (channelThread != null && channelThread.isAlive()) {
            return;
        }
        channelStop = false;
        channelThread = new Thread(() -> {
            Log.i(TAG, "channel poll start type=" + RCSDKManager.INSTANCE.getDeviceType());
            while (!channelStop && !isFinishing()) {
                if (!running.get()) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException ignored) {
                    }
                    continue;
                }
                if (channelInFlight.compareAndSet(false, true)) {
                    try {
                        KeyManager.INSTANCE.get(RemoteControllerKey.INSTANCE.getKeyChannels(),
                                new CompletionCallbackWith<int[]>() {
                                    @Override
                                    public void onSuccess(int[] value) {
                                        try {
                                            if (value != null) {
                                                latestCh.set(Arrays.copyOf(value, value.length));
                                                onChannels(value);
                                            }
                                        } finally {
                                            channelInFlight.set(false);
                                        }
                                    }

                                    @Override
                                    public void onFailure(SkyException e) {
                                        channelInFlight.set(false);
                                    }
                                });
                    } catch (Throwable t) {
                        channelInFlight.set(false);
                        Log.w(TAG, "get channels: " + t.getMessage());
                    }
                }
                try {
                    Thread.sleep(CHANNEL_MS);
                } catch (InterruptedException e) {
                    break;
                }
            }
            Log.i(TAG, "channel poll exit");
        }, "h12-channels");
        channelThread.setDaemon(true);
        channelThread.start();
    }

    private void onChannels(int[] ch) {
        final int[] rc = ChannelMapper.toTelloRc(ch);
        long now = SystemClock.uptimeMillis();
        if (now - lastUiChAt >= UI_CH_MS) {
            lastUiChAt = now;
            final String chText = formatChannels(ch);
            final String rcText = String.format("tello rc: %d %d %d %d [%s]",
                    rc[0], rc[1], rc[2], rc[3], controlMode);
            main.post(() -> {
                if (tvChannels != null) tvChannels.setText(chText);
                if (tvRc != null) tvRc.setText(rcText);
            });
        }

        boolean changed = rc[0] != lastA || rc[1] != lastB || rc[2] != lastC || rc[3] != lastD;
        // Orin 自动模式下不发摇杆（应急/起降开关仍走下面）
        if (!"auto".equals(controlMode) && (changed || now - lastRcSentAt > 200)) {
            lastA = rc[0];
            lastB = rc[1];
            lastC = rc[2];
            lastD = rc[3];
            lastRcSentAt = now;
            http.postRc(rc[0], rc[1], rc[2], rc[3]);
        } else if ("auto".equals(controlMode) && (lastA != 0 || lastB != 0 || lastC != 0 || lastD != 0)) {
            // Entered auto with leftover sticks — clear local cache only (Orin already zeroed)
            lastA = lastB = lastC = lastD = 0;
        }

        if (ch.length > 4) {
            ChannelMapper.TriPos z5 = ChannelMapper.triPos(ch[4]);
            if (z5 != ChannelMapper.TriPos.UNKNOWN) {
                if (ch5Prev == ChannelMapper.TriPos.UNKNOWN) {
                    // 首帧只记位置，不触发急停
                    ch5Prev = z5;
                } else if (z5 != ch5Prev) {
                    if (z5 == ChannelMapper.TriPos.HIGH) {
                        http.postCmd("emergency", (ok, msg) ->
                                Toast.makeText(this, ok ? "EMERGENCY" : ("急停失败: " + msg),
                                        Toast.LENGTH_SHORT).show());
                    }
                    ch5Prev = z5;
                }
            }
        }

        if (ch.length > 5) {
            ChannelMapper.TriPos z6 = ChannelMapper.triPos(ch[5]);
            if (z6 != ChannelMapper.TriPos.UNKNOWN) {
                if (ch6Prev == ChannelMapper.TriPos.UNKNOWN) {
                    // 首帧只记位置，避免开机 ch6=低 误发 land
                    ch6Prev = z6;
                } else if (z6 != ch6Prev) {
                    if (z6 == ChannelMapper.TriPos.HIGH) {
                        http.postCmd("takeoff", (ok, msg) ->
                                Toast.makeText(this,
                                        ok ? "takeoff ok" : ("起飞失败: " + msg),
                                        Toast.LENGTH_LONG).show());
                    } else if (z6 == ChannelMapper.TriPos.LOW) {
                        http.postCmd("land", (ok, msg) ->
                                Toast.makeText(this,
                                        ok ? "land ok" : ("降落失败: " + msg),
                                        Toast.LENGTH_LONG).show());
                    }
                    ch6Prev = z6;
                }
            }
        }
    }

    private static String formatChannels(int[] ch) {
        StringBuilder sb = new StringBuilder("ch:");
        int n = Math.min(ch.length, 8);
        for (int i = 0; i < n; i++) {
            sb.append(' ').append(i + 1).append('=').append(ch[i]);
            if (i == 4 || i == 5) {
                sb.append('(').append(ChannelMapper.triPos(ch[i]).name()).append(')');
            }
        }
        return sb.toString();
    }

    @Override
    protected void onResume() {
        super.onResume();
        running.set(true);
        main.removeCallbacks(frameTick);
        main.removeCallbacks(statusTick);
        main.post(frameTick);
        main.post(statusTick);
        startChannelPolling();
    }

    @Override
    protected void onPause() {
        running.set(false);
        main.removeCallbacks(frameTick);
        main.removeCallbacks(statusTick);
        http.postRc(0, 0, 0, 0);
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        running.set(false);
        channelStop = true;
        try {
            RCSDKManager.INSTANCE.disconnectRC();
        } catch (Throwable ignored) {
        }
        if (http != null) http.shutdown();
        super.onDestroy();
    }
}
