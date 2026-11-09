package com.aerial.ludgs;

import android.content.Context;
import android.content.SharedPreferences;
import android.media.AudioManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Log;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.skydroid.rcsdk.KeyManager;
import com.skydroid.rcsdk.RCSDKManager;
import com.skydroid.rcsdk.SDKManagerCallBack;
import com.skydroid.rcsdk.common.callback.CompletionCallbackWith;
import com.skydroid.rcsdk.common.error.SkyException;
import com.skydroid.rcsdk.key.RemoteControllerKey;

import org.json.JSONObject;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * H12 sticks → Orin HTTP → Lumos LCM. Close H12助手 / Skydroid Fly first.
 */
public class MainActivity extends AppCompatActivity {
    private static final String TAG = "LumosGS";
    private static final String PREFS = "lud_gs";
    private static final String KEY_ORIN_URL = "orin_url";
    /** H12 Wi-Fi → Orin HTTP bridge → Lumos LCM. */
    private static final String DEFAULT_ORIN_URL = "http://172.20.10.4:8092";
    private static final long CHANNEL_MS = 40L;
    private static final long UI_MS = 200L;
    private static final long STATUS_MS = 1500L;
    private static final long SEND_MS = 40L;

    private TextView tvStatus;
    private TextView tvBattery;
    private TextView tvChannels;
    private TextView tvCmd;
    private EditText etOrinUrl;
    private Button btnEnable;
    private Button btnApplyUrl;
    private Button btnReconnect;

    private LudHttp http;
    private String orinBase = DEFAULT_ORIN_URL;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean channelInFlight = new AtomicBoolean(false);
    private boolean rcSdkInited = false;
    private volatile boolean rcReady = false;
    private boolean outputOn = false;
    private int holdState = ChannelMapper.STATE_RESET;
    private ChannelMapper.TriPos ch5Prev = ChannelMapper.TriPos.UNKNOWN;
    private ChannelMapper.TriPos ch6Prev = ChannelMapper.TriPos.UNKNOWN;
    private Thread channelThread;
    private volatile boolean channelStop = false;
    private long lastUiAt = 0L;
    private long lastSendAt = 0L;
    private long lastChannelOkAt = 0L;
    private long channelReqAt = 0L;
    private boolean rfSilenced = false;
    private String lastCmdText = "";

    private final Runnable statusTick = new Runnable() {
        @Override
        public void run() {
            if (!running.get()) {
                return;
            }
            http.getHealth((ok, body) -> {
                if (tvStatus == null) {
                    return;
                }
                if (!ok) {
                    tvStatus.setText("Orin fail: " + body + " @ " + orinBase);
                    if (tvBattery != null) {
                        tvBattery.setText("电量 —");
                        tvBattery.setTextColor(0xFFBDBDBD);
                    }
                    return;
                }
                tvStatus.setText("Orin ok  output=" + (outputOn ? "ON" : "OFF"));
                updateBattery(body);
            });
            main.postDelayed(this, STATUS_MS);
        }
    };

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        tvStatus = findViewById(R.id.tvStatus);
        tvBattery = findViewById(R.id.tvBattery);
        tvChannels = findViewById(R.id.tvChannels);
        tvCmd = findViewById(R.id.tvCmd);
        etOrinUrl = findViewById(R.id.etOrinUrl);
        btnEnable = findViewById(R.id.btnEnable);
        btnApplyUrl = findViewById(R.id.btnApplyUrl);
        btnReconnect = findViewById(R.id.btnReconnect);

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        orinBase = normalizeBaseUrl(prefs.getString(KEY_ORIN_URL, DEFAULT_ORIN_URL));
        if (orinBase.contains("127.0.0.1") || orinBase.contains("172.20.10.2")) {
            orinBase = DEFAULT_ORIN_URL;
            prefs.edit().putString(KEY_ORIN_URL, orinBase).apply();
        }
        etOrinUrl.setText(orinBase);
        http = new LudHttp(orinBase);

        btnApplyUrl.setOnClickListener(v -> applyOrinUrl(true));
        etOrinUrl.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_DONE) {
                applyOrinUrl(true);
                return true;
            }
            return false;
        });
        btnEnable.setOnClickListener(v -> setOutputOn(!outputOn));
        btnReconnect.setOnClickListener(v -> reconnectRc());
        setOutputOn(false);
        initRcSdk();
    }

    private void setOutputOn(boolean on) {
        outputOn = on;
        if (btnEnable != null) {
            btnEnable.setText(on ? "输出 ON" : "输出 OFF");
            btnEnable.setBackgroundColor(on ? 0xCC2E7D32 : 0xCCB71C1C);
        }
        if (!on) {
            http.postCmd(holdState, 0, 0, 0, true, null);
        } else {
            http.postCmd(holdState, 0, 0, 0, false, null);
        }
    }

    private static String normalizeBaseUrl(String raw) {
        String b = raw == null ? DEFAULT_ORIN_URL : raw.trim();
        if (b.isEmpty()) {
            b = DEFAULT_ORIN_URL;
        }
        if (!b.startsWith("http://") && !b.startsWith("https://")) {
            b = "http://" + b;
        }
        while (b.endsWith("/")) {
            b = b.substring(0, b.length() - 1);
        }
        return b;
    }

    private void updateBattery(String body) {
        if (tvBattery == null) {
            return;
        }
        try {
            JSONObject j = new JSONObject(body == null ? "{}" : body);
            boolean hasPct = !j.isNull("battery_pct");
            int pct = hasPct ? j.optInt("battery_pct", -1) : -1;
            if (pct < 0 || pct > 100) {
                hasPct = false;
            }
            double volts = j.optDouble("battery_v", Double.NaN);
            if (!hasPct && Double.isNaN(volts)) {
                tvBattery.setText("电量 —");
                tvBattery.setTextColor(0xFFBDBDBD);
                return;
            }
            String text = hasPct ? ("电量 " + pct + "%") : "电量";
            if (!Double.isNaN(volts)) {
                text += String.format(hasPct ? "  %.1fV" : " %.1fV", volts);
            }
            tvBattery.setText(text);
            int color = 0xFF69F0AE;
            if (hasPct) {
                color = pct <= 20 ? 0xFFFF5252 : (pct <= 40 ? 0xFFFFD740 : 0xFF69F0AE);
            } else if (volts < 44.0) {
                color = 0xFFFF5252;
            } else if (volts < 48.0) {
                color = 0xFFFFD740;
            }
            tvBattery.setTextColor(color);
        } catch (Exception e) {
            tvBattery.setText("电量 —");
            tvBattery.setTextColor(0xFFBDBDBD);
        }
    }

    private void applyOrinUrl(boolean toast) {
        orinBase = normalizeBaseUrl(etOrinUrl.getText() != null
                ? etOrinUrl.getText().toString() : "");
        etOrinUrl.setText(orinBase);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit().putString(KEY_ORIN_URL, orinBase).apply();
        http.setBase(orinBase);
        if (toast) {
            Toast.makeText(this, "Orin → " + orinBase, Toast.LENGTH_SHORT).show();
        }
    }

    private void initRcSdk() {
        if (rcSdkInited) {
            RCSDKManager.INSTANCE.connectToRC();
            return;
        }
        rcSdkInited = true;
        RCSDKManager.INSTANCE.setMainThreadCallBack(false);
        RCSDKManager.INSTANCE.initSDK(this, new SDKManagerCallBack() {
            @Override
            public void onRcConnected() {
                rcReady = true;
                Log.i(TAG, "RC connected type=" + RCSDKManager.INSTANCE.getDeviceType());
                main.post(() -> {
                    Toast.makeText(MainActivity.this, "H12 RC ok", Toast.LENGTH_SHORT).show();
                    if (tvChannels != null) {
                        tvChannels.setText("RC ok, reading sticks…");
                    }
                });
                startChannelPolling();
            }

            @Override
            public void onRcConnectFail(@Nullable SkyException e) {
                Log.e(TAG, "RC connect fail: " + e);
                main.post(() -> {
                    if (tvChannels != null) {
                        tvChannels.setText("RCSDK fail: " + e + " — 关掉 H12助手/Fly 后重开本应用");
                    }
                });
            }

            @Override
            public void onRcDisconnect() {
                rcReady = false;
                Log.w(TAG, "RC disconnected");
            }
        });
        RCSDKManager.INSTANCE.connectToRC();
    }

    private void reconnectRc() {
        Toast.makeText(this, "重启 RCSDK", Toast.LENGTH_SHORT).show();
        channelStop = true;
        channelThread = null;
        rfSilenced = false;
        rcReady = false;
        rcSdkInited = false;
        try {
            RCSDKManager.INSTANCE.disconnectRC();
        } catch (Throwable ignored) {
        }
        main.postDelayed(this::initRcSdk, 400);
    }

    private void silenceRf() {
        com.skydroid.rcsdk.common.callback.CompletionCallback cb =
                e -> Log.i(TAG, "rf: " + e);
        try {
            KeyManager.INSTANCE.set(
                    RemoteControllerKey.INSTANCE.getKeyModule915Enable(), Boolean.FALSE, cb);
        } catch (Throwable ignored) {
        }
        try {
            KeyManager.INSTANCE.set(
                    RemoteControllerKey.INSTANCE.getKeyModuleGLinkEnable(), Boolean.FALSE, cb);
        } catch (Throwable ignored) {
        }
        try {
            KeyManager.INSTANCE.action(
                    RemoteControllerKey.INSTANCE.getKeyRequestStopPair(), cb);
        } catch (Throwable ignored) {
        }
        main.post(this::muteAndroidAlerts);
    }

    private void muteAndroidAlerts() {
        AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
        if (am == null) {
            return;
        }
        int[] streams = {
                AudioManager.STREAM_MUSIC,
                AudioManager.STREAM_ALARM,
                AudioManager.STREAM_NOTIFICATION,
                AudioManager.STREAM_SYSTEM,
                AudioManager.STREAM_RING,
        };
        for (int s : streams) {
            try {
                am.setStreamVolume(s, 0, 0);
            } catch (Throwable ignored) {
            }
        }
    }

    private void startChannelPolling() {
        if (channelThread != null && channelThread.isAlive()) {
            return;
        }
        channelStop = false;
        channelThread = new Thread(() -> {
            Log.i(TAG, "channel poll start type=" + RCSDKManager.INSTANCE.getDeviceType());
            while (!channelStop && !isFinishing()) {
                if (!running.get() || !rcReady) {
                    try {
                        Thread.sleep(100);
                    } catch (InterruptedException ignored) {
                    }
                    continue;
                }
                if (channelInFlight.compareAndSet(false, true)) {
                    channelReqAt = SystemClock.uptimeMillis();
                    try {
                        KeyManager.INSTANCE.get(RemoteControllerKey.INSTANCE.getKeyChannels(),
                                new CompletionCallbackWith<int[]>() {
                                    @Override
                                    public void onSuccess(int[] value) {
                                        try {
                                            lastChannelOkAt = SystemClock.uptimeMillis();
                                            if (value != null) {
                                                onChannels(Arrays.copyOf(value, value.length));
                                            }
                                        } finally {
                                            channelInFlight.set(false);
                                        }
                                    }

                                    @Override
                                    public void onFailure(SkyException e) {
                                        Log.w(TAG, "KeyChannels fail: " + e);
                                        channelInFlight.set(false);
                                    }
                                });
                    } catch (Throwable t) {
                        Log.w(TAG, "KeyChannels throw: " + t);
                        channelInFlight.set(false);
                    }
                }
                try {
                    Thread.sleep(CHANNEL_MS);
                } catch (InterruptedException e) {
                    break;
                }
            }
        }, "lud-channels");
        channelThread.setDaemon(true);
        channelThread.start();
    }

    private void onChannels(int[] ch) {
        ChannelMapper.TriPos z5 = ch.length > 4
                ? ChannelMapper.triPos(ch[4]) : ChannelMapper.TriPos.UNKNOWN;
        ChannelMapper.TriPos z6 = ch.length > 5
                ? ChannelMapper.triPos(ch[5]) : ChannelMapper.TriPos.UNKNOWN;
        int next = ChannelMapper.stateFromSwitches(z5, z6, holdState);
        boolean stateChanged = holdState != next || z6 != ch6Prev || z5 != ch5Prev;
        holdState = next;
        if (z6 != ChannelMapper.TriPos.UNKNOWN) {
            ch6Prev = z6;
        }
        if (z5 != ChannelMapper.TriPos.UNKNOWN) {
            if (z5 == ChannelMapper.TriPos.HIGH && ch5Prev != ChannelMapper.TriPos.HIGH && outputOn) {
                http.postCmd(holdState, 0, 0, 0, true, null);
                main.post(() -> Toast.makeText(this, "急停 速度清零", Toast.LENGTH_SHORT).show());
            }
            ch5Prev = z5;
        }
        if (stateChanged && outputOn) {
            http.postCmd(holdState, 0, 0, 0, false, null);
        }

        ChannelMapper.Cmd cmd = ChannelMapper.toLumos(ch, holdState, outputOn);
        long now = SystemClock.uptimeMillis();
        if (now - lastUiAt >= UI_MS) {
            lastUiAt = now;
            final String chText = formatChannels(ch);
            final String cmdText = String.format(
                    "state=%d vx=%.2f vy=%.2f vyaw=%.2f stop=%s out=%s",
                    cmd.state, cmd.vx, cmd.vy, cmd.vyaw, cmd.stop, outputOn);
            lastCmdText = cmdText;
            main.post(() -> {
                if (tvChannels != null) {
                    tvChannels.setText(chText);
                }
                if (tvCmd != null) {
                    tvCmd.setText(cmdText);
                }
            });
        }
        if (outputOn && now - lastSendAt >= SEND_MS) {
            lastSendAt = now;
            http.postCmd(cmd.state, cmd.vx, cmd.vy, cmd.vyaw, cmd.stop, null);
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
        main.removeCallbacks(statusTick);
        main.post(statusTick);
        startChannelPolling();
    }

    @Override
    protected void onPause() {
        running.set(false);
        main.removeCallbacks(statusTick);
        http.postCmd(holdState, 0, 0, 0, true, null);
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
        http.shutdown();
        super.onDestroy();
    }
}
