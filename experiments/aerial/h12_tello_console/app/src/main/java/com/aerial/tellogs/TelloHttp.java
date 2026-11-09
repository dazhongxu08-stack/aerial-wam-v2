package com.aerial.tellogs;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;

/** HTTP client for Mac tello_ground_console (H12 adb reverse → 127.0.0.1:8090). */
public final class TelloHttp {
    private static final String TAG = "TelloHttp";
    private final String base;
    private final ExecutorService pool = Executors.newFixedThreadPool(4);
    private final ExecutorService framePool = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicReference<int[]> lastRc = new AtomicReference<>(new int[]{0, 0, 0, 0});

    public interface StatusCb {
        void onStatus(JSONObject status, Exception err);
    }

    public interface FrameCb {
        void onFrame(Bitmap bmp, Exception err);
    }

    public TelloHttp(String baseUrl) {
        String b = baseUrl == null ? "http://127.0.0.1:8090" : baseUrl.trim();
        while (b.endsWith("/")) b = b.substring(0, b.length() - 1);
        this.base = b;
    }

    public void shutdown() {
        pool.shutdownNow();
        framePool.shutdownNow();
    }

    public void fetchStatus(StatusCb cb) {
        pool.execute(() -> {
            try {
                HttpURLConnection c = (HttpURLConnection) new URL(base + "/api/status").openConnection();
                c.setConnectTimeout(2500);
                c.setReadTimeout(2500);
                c.setRequestProperty("Connection", "close");
                c.setRequestMethod("GET");
                InputStream in = c.getInputStream();
                String body = readAll(in);
                in.close();
                c.disconnect();
                JSONObject j = new JSONObject(body);
                main.post(() -> cb.onStatus(j, null));
            } catch (Exception e) {
                main.post(() -> cb.onStatus(null, e));
            }
        });
    }

    public void fetchSnapshot(FrameCb cb) {
        framePool.execute(() -> {
            HttpURLConnection c = null;
            try {
                c = (HttpURLConnection) new URL(base + "/snapshot.jpg?t=" + System.currentTimeMillis()).openConnection();
                c.setConnectTimeout(1500);
                c.setReadTimeout(1500);
                c.setRequestProperty("Connection", "close");
                c.setRequestMethod("GET");
                InputStream in = c.getInputStream();
                Bitmap bmp = BitmapFactory.decodeStream(in);
                in.close();
                if (bmp == null) throw new IllegalStateException("decode null");
                main.post(() -> cb.onFrame(bmp, null));
            } catch (Exception e) {
                main.post(() -> cb.onFrame(null, e));
            } finally {
                if (c != null) c.disconnect();
            }
        });
    }

    public void postRc(int a, int b, int c, int d) {
        int[] prev = lastRc.get();
        if (prev[0] == a && prev[1] == b && prev[2] == c && prev[3] == d) {
            // still re-send at low rate so server keepalive stays fresh — caller decides
        }
        lastRc.set(new int[]{a, b, c, d});
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("a", a);
                body.put("b", b);
                body.put("c", c);
                body.put("d", d);
                body.put("source", "manual");
                postJson("/api/rc", body);
            } catch (Exception e) {
                Log.w(TAG, "rc fail: " + e.getMessage());
            }
        });
    }

    public void postCmd(String cmd) {
        postCmd(cmd, null);
    }

    public void postCmd(String cmd, SimpleCb cb) {
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("cmd", cmd);
                // Accept 403 body so caller sees "ground lock" etc.
                String resp = postJsonAllowError("/api/cmd", body);
                boolean ok = true;
                String msg = resp;
                try {
                    JSONObject j = new JSONObject(resp);
                    ok = j.optBoolean("ok", false);
                    msg = j.optString("msg", resp);
                } catch (Exception ignored) {
                }
                Log.i(TAG, "cmd " + cmd + " ok=" + ok + " " + msg);
                if (cb != null) {
                    boolean finalOk = ok;
                    String finalMsg = msg;
                    main.post(() -> cb.onDone(finalOk, finalMsg));
                }
            } catch (Exception e) {
                Log.w(TAG, "cmd fail: " + e.getMessage());
                if (cb != null) {
                    main.post(() -> cb.onDone(false, e.getMessage()));
                }
            }
        });
    }

    public interface SimpleCb {
        void onDone(boolean ok, String msg);
    }

    public void postAllowFly(Boolean enabled, SimpleCb cb) {
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                if (enabled != null) {
                    body.put("allow_fly", enabled.booleanValue());
                }
                String resp = postJsonAllowError("/api/allow_fly", body);
                boolean ok = true;
                String msg = resp;
                try {
                    JSONObject j = new JSONObject(resp);
                    ok = j.optBoolean("ok", false);
                    msg = j.optString("msg", resp);
                    if (j.has("allow_fly")) {
                        msg = msg + " allow_fly=" + j.optBoolean("allow_fly");
                    }
                } catch (Exception ignored) {
                }
                boolean finalOk = ok;
                String finalMsg = msg;
                if (cb != null) {
                    main.post(() -> cb.onDone(finalOk, finalMsg));
                }
            } catch (Exception e) {
                Log.w(TAG, "allow_fly fail: " + e.getMessage());
                if (cb != null) {
                    main.post(() -> cb.onDone(false, e.getMessage()));
                }
            }
        });
    }

    /**
     * Switch stick ownership on Orin. {@code mode} null = toggle;
     * otherwise {@code "manual"} (H12) or {@code "auto"} (Orin policy).
     */
    public void postControlMode(String mode, SimpleCb cb) {
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                if (mode != null && !mode.isEmpty()) {
                    body.put("mode", mode);
                }
                String resp = postJsonAllowError("/api/control_mode", body);
                boolean ok = true;
                String msg = resp;
                try {
                    JSONObject j = new JSONObject(resp);
                    ok = j.optBoolean("ok", false);
                    if (j.has("control_mode")) {
                        msg = j.optString("control_mode", msg);
                    } else {
                        msg = j.optString("msg", resp);
                    }
                } catch (Exception ignored) {
                }
                boolean finalOk = ok;
                String finalMsg = msg;
                if (cb != null) {
                    main.post(() -> cb.onDone(finalOk, finalMsg));
                }
            } catch (Exception e) {
                Log.w(TAG, "control_mode fail: " + e.getMessage());
                if (cb != null) {
                    main.post(() -> cb.onDone(false, e.getMessage()));
                }
            }
        });
    }

    public void postRecord(boolean on, SimpleCb cb) {
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                body.put("on", on);
                String resp = postJsonAllowError("/api/record", body);
                boolean ok = true;
                String msg = resp;
                try {
                    JSONObject j = new JSONObject(resp);
                    ok = j.optBoolean("ok", false);
                    msg = j.optString("msg", resp);
                } catch (Exception ignored) {
                }
                boolean finalOk = ok;
                String finalMsg = msg;
                if (cb != null) {
                    main.post(() -> cb.onDone(finalOk, finalMsg));
                }
            } catch (Exception e) {
                Log.w(TAG, "record fail: " + e.getMessage());
                if (cb != null) {
                    main.post(() -> cb.onDone(false, e.getMessage()));
                }
            }
        });
    }

    /** Ask Orin to restart Tello service (SDK/video) and rejoin TELLO Wi-Fi. */
    public void postReconnect(SimpleCb cb) {
        pool.execute(() -> {
            try {
                JSONObject body = new JSONObject();
                String resp = postJson("/api/reconnect", body);
                boolean ok = true;
                String msg = resp;
                try {
                    JSONObject j = new JSONObject(resp);
                    ok = j.optBoolean("ok", true);
                    msg = j.optString("msg", resp);
                } catch (Exception ignored) {
                }
                boolean finalOk = ok;
                String finalMsg = msg;
                main.post(() -> cb.onDone(finalOk, finalMsg));
            } catch (Exception e) {
                Log.w(TAG, "reconnect fail: " + e.getMessage());
                main.post(() -> cb.onDone(false, e.getMessage()));
            }
        });
    }

    private String postJson(String path, JSONObject body) throws Exception {
        String resp = postJsonAllowError(path, body);
        // Caller of postJson expects success-only; if body says ok:false still return it
        return resp;
    }

    private String postJsonAllowError(String path, JSONObject body) throws Exception {
        byte[] raw = body.toString().getBytes(StandardCharsets.UTF_8);
        HttpURLConnection c = (HttpURLConnection) new URL(base + path).openConnection();
        c.setConnectTimeout(2500);
        c.setReadTimeout(8000);
        c.setRequestMethod("POST");
        c.setDoOutput(true);
        c.setRequestProperty("Connection", "close");
        c.setRequestProperty("Content-Type", "application/json");
        c.setRequestProperty("Content-Length", String.valueOf(raw.length));
        OutputStream os = c.getOutputStream();
        os.write(raw);
        os.close();
        int code = c.getResponseCode();
        InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
        String resp = "";
        if (in != null) {
            resp = readAll(in);
            in.close();
            Log.d(TAG, path + " → " + code + " " + resp);
        }
        c.disconnect();
        if (code >= 500) {
            throw new IllegalStateException(path + " HTTP " + code + " " + resp);
        }
        if (resp == null || resp.isEmpty()) {
            throw new IllegalStateException(path + " HTTP " + code + " empty");
        }
        return resp;
    }

    private static String readAll(InputStream in) throws Exception {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = in.read(buf)) >= 0) bos.write(buf, 0, n);
        return bos.toString("UTF-8");
    }
}
