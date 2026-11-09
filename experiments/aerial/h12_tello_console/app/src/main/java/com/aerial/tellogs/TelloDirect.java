package com.aerial.tellogs;

import android.util.Log;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketTimeoutException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Tello SDK 3.0 over UDP — runs on H12 Wi‑Fi station joined to the Tello AP.
 * No Orin / Mac relay.
 */
public final class TelloDirect implements AutoCloseable {
    private static final String TAG = "TelloDirect";
    public static final String TELLO_IP = "192.168.10.1";
    public static final int CMD_PORT = 8889;
    public static final int STATE_PORT = 8890;
    public static final int VIDEO_PORT = 11111;

    public interface Listener {
        void onState(Map<String, String> state);

        void onLink(boolean ok, String msg);
    }

    private final Object cmdLock = new Object();
    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicReference<Map<String, String>> state =
            new AtomicReference<>(Collections.<String, String>emptyMap());
    private final AtomicLong lastStateAt = new AtomicLong(0L);
    private final AtomicBoolean linkOk = new AtomicBoolean(false);
    private final AtomicReference<String> linkError = new AtomicReference<>("idle");

    private DatagramSocket cmdSock;
    private DatagramSocket stateSock;
    private Thread stateThread;
    private Thread keepAliveThread;
    private Listener listener;
    private InetAddress telloAddr;

    public void setListener(Listener listener) {
        this.listener = listener;
    }

    public boolean isLinkOk() {
        return linkOk.get();
    }

    public String linkError() {
        return linkError.get();
    }

    public Map<String, String> stateSnapshot() {
        return state.get();
    }

    public Double stateAgeS() {
        long t = lastStateAt.get();
        if (t <= 0) return null;
        return (System.currentTimeMillis() - t) / 1000.0;
    }

    public Integer battery() {
        String bat = state.get().get("bat");
        if (bat == null) return null;
        try {
            return Integer.parseInt(bat.trim());
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /** Bind sockets + SDK handshake + state listener. Call after H12 is on TELLO Wi‑Fi. */
    public synchronized boolean connect() {
        close();
        try {
            telloAddr = InetAddress.getByName(TELLO_IP);
            cmdSock = new DatagramSocket(null);
            cmdSock.setReuseAddress(true);
            cmdSock.bind(new InetSocketAddress(CMD_PORT));
            cmdSock.setSoTimeout(8000);

            stateSock = new DatagramSocket(null);
            stateSock.setReuseAddress(true);
            stateSock.bind(new InetSocketAddress(STATE_PORT));
            stateSock.setSoTimeout(1000);

            running.set(true);
            stateThread = new Thread(this::stateLoop, "tello-state");
            stateThread.setDaemon(true);
            stateThread.start();
            keepAliveThread = new Thread(this::keepAliveLoop, "tello-ka");
            keepAliveThread.setDaemon(true);
            keepAliveThread.start();

            setLink(false, "handshaking…");
            boolean ok = false;
            for (int i = 0; i < 5; i++) {
                String resp = send("command", true, 8000);
                if ("ok".equalsIgnoreCase(resp)) {
                    ok = true;
                    break;
                }
                try {
                    Thread.sleep(400);
                } catch (InterruptedException ignored) {
                }
            }
            if (!ok) {
                setLink(false, "SDK handshake failed");
                return false;
            }
            setLink(true, "");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "connect failed", e);
            setLink(false, e.getMessage());
            close();
            return false;
        }
    }

    public boolean streamOn() {
        return "ok".equalsIgnoreCase(send("streamon", true, 12000));
    }

    public void streamOff() {
        send("streamoff", true, 3000);
    }

    public void rc(int a, int b, int c, int d) {
        a = clamp(a);
        b = clamp(b);
        c = clamp(c);
        d = clamp(d);
        send(String.format(Locale.US, "rc %d %d %d %d", a, b, c, d), false, 0);
    }

    public String takeoff() {
        return send("takeoff", true, 20000);
    }

    public String land() {
        return send("land", true, 20000);
    }

    public void emergency() {
        // Bypass lock: fire-and-forget on a fresh socket if needed
        send("emergency", false, 0);
        try {
            DatagramSocket s = new DatagramSocket();
            byte[] raw = "emergency".getBytes(StandardCharsets.UTF_8);
            s.send(new DatagramPacket(raw, raw.length, telloAddr, CMD_PORT));
            s.close();
        } catch (Exception e) {
            Log.w(TAG, "emergency alt: " + e.getMessage());
        }
    }

    private static int clamp(int v) {
        return Math.max(-100, Math.min(100, v));
    }

    private String send(String cmd, boolean wait, int timeoutMs) {
        synchronized (cmdLock) {
            if (cmdSock == null || telloAddr == null) return null;
            try {
                drainCmd();
                if (wait) {
                    cmdSock.setSoTimeout(Math.max(500, timeoutMs));
                }
                byte[] raw = cmd.getBytes(StandardCharsets.UTF_8);
                cmdSock.send(new DatagramPacket(raw, raw.length, telloAddr, CMD_PORT));
                Log.i(TAG, ">>> " + cmd);
                if (!wait) return null;
                byte[] buf = new byte[2048];
                DatagramPacket pkt = new DatagramPacket(buf, buf.length);
                cmdSock.receive(pkt);
                String resp = new String(pkt.getData(), 0, pkt.getLength(), StandardCharsets.UTF_8)
                        .trim();
                Log.i(TAG, "<<< " + resp);
                return resp;
            } catch (SocketTimeoutException e) {
                Log.w(TAG, "timeout: " + cmd);
                return null;
            } catch (Exception e) {
                Log.w(TAG, "send " + cmd + ": " + e.getMessage());
                return null;
            }
        }
    }

    private void drainCmd() {
        if (cmdSock == null) return;
        try {
            cmdSock.setSoTimeout(1);
            byte[] buf = new byte[2048];
            DatagramPacket pkt = new DatagramPacket(buf, buf.length);
            while (true) {
                cmdSock.receive(pkt);
            }
        } catch (Exception ignored) {
        }
    }

    private void stateLoop() {
        byte[] buf = new byte[4096];
        while (running.get() && stateSock != null) {
            try {
                DatagramPacket pkt = new DatagramPacket(buf, buf.length);
                stateSock.receive(pkt);
                String text = new String(pkt.getData(), 0, pkt.getLength(), StandardCharsets.UTF_8)
                        .trim();
                Map<String, String> parsed = new HashMap<>();
                for (String part : text.split(";")) {
                    int i = part.indexOf(':');
                    if (i > 0) {
                        parsed.put(part.substring(0, i), part.substring(i + 1));
                    }
                }
                if (!parsed.isEmpty()) {
                    state.set(Collections.unmodifiableMap(parsed));
                    lastStateAt.set(System.currentTimeMillis());
                    Listener l = listener;
                    if (l != null) l.onState(parsed);
                }
            } catch (SocketTimeoutException ignored) {
            } catch (Exception e) {
                if (running.get()) Log.w(TAG, "state: " + e.getMessage());
                break;
            }
        }
    }

    private void keepAliveLoop() {
        while (running.get()) {
            for (int i = 0; i < 16 && running.get(); i++) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    return;
                }
            }
            if (!running.get()) return;
            send("command", false, 0);
        }
    }

    private void setLink(boolean ok, String msg) {
        linkOk.set(ok);
        linkError.set(msg == null ? "" : msg);
        Listener l = listener;
        if (l != null) l.onLink(ok, msg);
    }

    @Override
    public synchronized void close() {
        running.set(false);
        linkOk.set(false);
        try {
            if (cmdSock != null) cmdSock.close();
        } catch (Exception ignored) {
        }
        try {
            if (stateSock != null) stateSock.close();
        } catch (Exception ignored) {
        }
        cmdSock = null;
        stateSock = null;
        if (stateThread != null) {
            try {
                stateThread.join(500);
            } catch (InterruptedException ignored) {
            }
        }
        if (keepAliveThread != null) {
            try {
                keepAliveThread.join(500);
            } catch (InterruptedException ignored) {
            }
        }
        stateThread = null;
        keepAliveThread = null;
    }
}
