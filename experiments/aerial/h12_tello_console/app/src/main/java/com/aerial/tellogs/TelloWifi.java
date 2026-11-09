package com.aerial.tellogs;

import android.content.Context;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiConfiguration;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.util.Log;

import java.util.List;

/** Join open TELLO-* AP (Tello is single-STA; Orin must leave the AP first). */
public final class TelloWifi {
    private static final String TAG = "TelloWifi";

    private TelloWifi() {}

    public static String currentSsid(Context ctx) {
        WifiManager wm = (WifiManager) ctx.getApplicationContext()
                .getSystemService(Context.WIFI_SERVICE);
        if (wm == null) return null;
        WifiInfo info = wm.getConnectionInfo();
        if (info == null || info.getSSID() == null) return null;
        String ssid = info.getSSID().replace("\"", "");
        if ("<unknown ssid>".equalsIgnoreCase(ssid)) return null;
        return ssid;
    }

    public static boolean onTello(Context ctx) {
        String ssid = currentSsid(ctx);
        return ssid != null && ssid.toUpperCase().startsWith("TELLO");
    }

    /**
     * Connect to strongest visible TELLO-* open network, or fall back to preferred SSID.
     * @return message for UI
     */
    @SuppressWarnings("deprecation")
    public static String joinTello(Context ctx, String preferredSsid) {
        WifiManager wm = (WifiManager) ctx.getApplicationContext()
                .getSystemService(Context.WIFI_SERVICE);
        if (wm == null) return "no WifiManager";
        if (!wm.isWifiEnabled()) {
            wm.setWifiEnabled(true);
            try {
                Thread.sleep(1500);
            } catch (InterruptedException ignored) {
            }
        }

        if (onTello(ctx)) {
            return "already on " + currentSsid(ctx);
        }

        String target = preferredSsid;
        try {
            wm.startScan();
            Thread.sleep(2000);
            List<ScanResult> scans = wm.getScanResults();
            int best = -999;
            if (scans != null) {
                for (ScanResult r : scans) {
                    if (r.SSID == null) continue;
                    if (!r.SSID.toUpperCase().startsWith("TELLO")) continue;
                    if (r.level > best) {
                        best = r.level;
                        target = r.SSID;
                    }
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "scan: " + t.getMessage());
        }
        if (target == null || target.isEmpty()) {
            target = "TELLO-E9F873";
        }

        try {
            // Remove stale TELLO configs then add open profile
            List<WifiConfiguration> configs = wm.getConfiguredNetworks();
            if (configs != null) {
                for (WifiConfiguration c : configs) {
                    String s = c.SSID == null ? "" : c.SSID.replace("\"", "");
                    if (s.toUpperCase().startsWith("TELLO")) {
                        wm.removeNetwork(c.networkId);
                    }
                }
            }
            WifiConfiguration conf = new WifiConfiguration();
            conf.SSID = "\"" + target + "\"";
            conf.allowedKeyManagement.set(WifiConfiguration.KeyMgmt.NONE);
            conf.allowedAuthAlgorithms.clear();
            int id = wm.addNetwork(conf);
            if (id < 0) {
                return "addNetwork failed for " + target;
            }
            wm.disconnect();
            boolean en = wm.enableNetwork(id, true);
            wm.reconnect();
            wm.saveConfiguration();
            Log.i(TAG, "join " + target + " id=" + id + " enable=" + en);

            for (int i = 0; i < 20; i++) {
                Thread.sleep(500);
                if (onTello(ctx)) {
                    return "joined " + currentSsid(ctx);
                }
            }
            return "join timeout (" + target + ") ssid=" + currentSsid(ctx);
        } catch (Throwable t) {
            Log.e(TAG, "join failed", t);
            return "join error: " + t.getMessage();
        }
    }
}
