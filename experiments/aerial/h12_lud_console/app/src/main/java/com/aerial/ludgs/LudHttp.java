package com.aerial.ludgs;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.HttpURLConnection;
import java.net.InetAddress;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Stick commands go UDP (latest-wins). Health stays on HTTP so a slow GET
 * cannot stall the control path.
 */
public final class LudHttp {
    public interface Result {
        void onDone(boolean ok, String body);
    }

    private static final String TAG = "LudHttp";
    private final ExecutorService healthIo = Executors.newSingleThreadExecutor();
    private final ExecutorService cmdIo = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());
    private final AtomicReference<String> pendingCmd = new AtomicReference<>();
    private final AtomicBoolean draining = new AtomicBoolean(false);
    private volatile String base;
    private volatile String host = "172.20.10.4";
    private volatile int udpPort = 8092;
    private DatagramSocket udp;
    private InetAddress udpAddr;

    public LudHttp(String base) {
        setBase(base);
        try {
            udp = new DatagramSocket();
            udp.setSoTimeout(1);
        } catch (Exception e) {
            Log.w(TAG, "udp socket: " + e.getMessage());
        }
    }

    public void setBase(String base) {
        this.base = base;
        try {
            URL u = new URL(base);
            host = u.getHost();
            int p = u.getPort();
            udpPort = p > 0 ? p : 8092;
            udpAddr = InetAddress.getByName(host);
        } catch (Exception e) {
            Log.w(TAG, "setBase: " + e.getMessage());
        }
    }

    public void shutdown() {
        healthIo.shutdownNow();
        cmdIo.shutdownNow();
        if (udp != null) {
            udp.close();
        }
    }

    public void getHealth(Result cb) {
        healthIo.execute(() -> {
            String err = httpGet("/api/health");
            boolean ok = err != null && err.contains("h12_lud_bridge");
            main.post(() -> cb.onDone(ok, err == null ? "" : err));
        });
    }

    public void postCmd(int state, float vx, float vy, float vyaw, boolean stop, Result cb) {
        try {
            JSONObject o = new JSONObject();
            o.put("state", state);
            o.put("vx", vx);
            o.put("vy", vy);
            o.put("vyaw", vyaw);
            o.put("stop", stop);
            pendingCmd.set(o.toString());
        } catch (Exception e) {
            Log.w(TAG, "json: " + e.getMessage());
            return;
        }
        if (draining.compareAndSet(false, true)) {
            cmdIo.execute(this::drainCmds);
        }
        if (cb != null) {
            main.post(() -> cb.onDone(true, "queued"));
        }
    }

    private void drainCmds() {
        try {
            String json;
            while ((json = pendingCmd.getAndSet(null)) != null) {
                sendUdp(json);
            }
        } finally {
            draining.set(false);
            if (pendingCmd.get() != null && draining.compareAndSet(false, true)) {
                cmdIo.execute(this::drainCmds);
            }
        }
    }

    private void sendUdp(String json) {
        try {
            if (udp == null || udp.isClosed() || udpAddr == null) {
                Log.w(TAG, "udp not ready");
                return;
            }
            byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
            udp.send(new DatagramPacket(bytes, bytes.length, udpAddr, udpPort));
        } catch (Exception e) {
            Log.w(TAG, "udp: " + e.getMessage());
        }
    }

    private String httpGet(String path) {
        HttpURLConnection c = null;
        try {
            URL url = new URL(base + path);
            c = (HttpURLConnection) url.openConnection();
            c.setConnectTimeout(400);
            c.setReadTimeout(400);
            c.setRequestMethod("GET");
            c.setUseCaches(false);
            int code = c.getResponseCode();
            InputStream in = code >= 400 ? c.getErrorStream() : c.getInputStream();
            return readAll(in);
        } catch (Exception e) {
            Log.w(TAG, "GET " + path + ": " + e.getMessage());
            return e.getMessage();
        } finally {
            if (c != null) {
                c.disconnect();
            }
        }
    }

    private static String readAll(InputStream in) throws Exception {
        if (in == null) {
            return "";
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
        in.close();
        return out.toString("UTF-8");
    }
}
