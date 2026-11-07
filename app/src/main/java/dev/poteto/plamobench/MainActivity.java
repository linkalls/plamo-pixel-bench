package dev.poteto.plamobench;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.graphics.Typeface;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ref.WeakReference;

/**
 * A single Android app with three independently compiled CPU inference engines.
 * The UI continuously reports native progress and elapsed time.
 */
public final class MainActivity extends Activity {
    private static final boolean VULKAN_AVAILABLE;
    private static final boolean HYBRID_AVAILABLE;
    static {
        // All libraries are self-contained. Unique JNI names choose the intended engine.
        System.loadLibrary("plamobench_generic");
        System.loadLibrary("plamobench_dotprod");
        System.loadLibrary("plamobench_kleidi");
        boolean gpuLibraryLoaded = false;
        try {
            // GPUライブラリは実験的。CIビルド失敗時もCPU版APKが起動できるようにする。
            System.loadLibrary("plamobench_vulkan");
            gpuLibraryLoaded = true;
        } catch (UnsatisfiedLinkError notAvailable) {
            android.util.Log.w("PLaMoBench", "Vulkan backend unavailable", notAvailable);
        }
        VULKAN_AVAILABLE = gpuLibraryLoaded;
        boolean hybridLoaded = false;
        if (gpuLibraryLoaded) {
            try {
                System.loadLibrary("plamobench_hybrid");
                hybridLoaded = true;
            } catch (UnsatisfiedLinkError error) {
                android.util.Log.w("PLaMoBench", "Hybrid engine unavailable", error);
            }
        }
        HYBRID_AVAILABLE = hybridLoaded;
    }

    private static volatile WeakReference<MainActivity> current = new WeakReference<>(null);
    private static final int PICK_MODEL = 1001;
    private static final String[] ENGINES = HYBRID_AVAILABLE
            ? new String[]{"Generic", "Dotprod", "KleidiAI", "Vulkan GPU (実験)",
                "Hybrid CPU Prefill + GPU Decode (実験)"}
            : VULKAN_AVAILABLE
                ? new String[]{"Generic", "Dotprod", "KleidiAI", "Vulkan GPU (実験)"}
                : new String[]{"Generic", "Dotprod", "KleidiAI"};
    private static final String TEST_TEXT = "このアプリはスマートフォンのローカル翻訳モデルの速度を測定するためのものです。"
            + "正確な比較のため、すべてのテストで同じ文章、同じモデル、同じスレッド数を使用してください。";

    private final Handler handler = new Handler(Looper.getMainLooper());
    private File model;
    private Spinner engine;
    private Spinner threads;
    private Spinner outputCap;
    private Spinner target;
    private EditText source;
    private TextView progress;
    private TextView results;
    private Button start;
    private Button stop;
    private Button choose;

    private volatile String nativeStage = "待機中";
    private volatile boolean working = false;
    private volatile boolean cancelRequested = false;
    private long startedAt;
    private int selectedEngine;

    // JNI invokes this from its background worker, not from the main thread.
    // Do not block the inference worker to update the UI.
    @SuppressWarnings("unused")
    private static void onNativeProgress(String newStage) {
        MainActivity activity = current.get();
        if (activity != null) activity.nativeStage = newStage;
    }

    // The ticker continues even if llama_decode itself takes several seconds.
    private final Runnable ticker = new Runnable() {
        @Override
        public void run() {
            if (!working) return;
            long seconds = (SystemClock.elapsedRealtime() - startedAt) / 1000;
            progress.setText("エンジン: " + ENGINES[selectedEngine] + "\n"
                    + "経過: " + seconds + "秒 / 安全上限300秒\n"
                    + "段階: " + nativeStage + "\n"
                    + (cancelRequested ? "中断要求済み：ネイティブ処理の停止を確認中" : ""));
            handler.postDelayed(this, 500);
        }
    };

    private TextView label(String str, int size) {
        TextView t = new TextView(this);
        t.setText(str);
        t.setTextSize(size);
        t.setPadding(8, 12, 8, 12);
        return t;
    }

    private Spinner spinner(String[] values, int initial) {
        Spinner s = new Spinner(this);
        ArrayAdapter<String> a = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, values);
        a.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        s.setAdapter(a);
        s.setSelection(initial);
        return s;
    }

    private void busy(boolean value) {
        working = value;
        start.setEnabled(!value);
        choose.setEnabled(!value);
        stop.setEnabled(value && !cancelRequested);
        engine.setEnabled(!value);
        threads.setEnabled(!value);
        outputCap.setEnabled(!value);
        target.setEnabled(!value);
        source.setEnabled(!value);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        current = new WeakReference<>(this);
        model = new File(getFilesDir(), "plamo-model.gguf");

        LinearLayout content = new LinearLayout(this);
        content.setPadding(20, 24, 20, 24);
        content.setOrientation(LinearLayout.VERTICAL);

        content.addView(label("PLaMo 0.8B / Pixel CPU BENCHMARK v0.5.0", 18));
        content.addView(label("推論エンジンを選択（CPU 3＋GPU＋Hybrid、合計5方式）", 14));
        engine = spinner(ENGINES, 2);
        content.addView(engine);

        choose = new Button(this);
        choose.setText("GGUFモデルを選択（初回のみ）");
        content.addView(choose);
        choose.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, PICK_MODEL);
        });

        content.addView(label("比較用テキスト", 14));
        source = new EditText(this);
        source.setText(TEST_TEXT);
        source.setMinLines(3);
        source.setMaxLines(7);
        content.addView(source);

        content.addView(label("翻訳先", 14));
        target = spinner(new String[]{"en", "ja"}, 0);
        content.addView(target);

        content.addView(label("CPU threads", 14));
        threads = spinner(new String[]{"1", "2", "4", "6", "8"}, 2);
        content.addView(threads);

        content.addView(label("最大生成トークン（長文は1024推奨。終了トークンで自動停止）", 14));
        outputCap = spinner(new String[]{"16", "64", "128", "256", "512", "1024", "2048"}, 5);
        content.addView(outputCap);

        start = new Button(this);
        start.setText("速度測定スタート");
        content.addView(start);

        stop = new Button(this);
        stop.setText("測定を中断");
        stop.setEnabled(false);
        content.addView(stop);

        progress = label("待機中", 16);
        progress.setTypeface(Typeface.MONOSPACE);
        content.addView(progress);

        Button copy = new Button(this);
        copy.setText("結果をコピー");
        content.addView(copy);

        results = label(model.exists() ? "GGUF準備完了"
                : "GGUFを選択してね。モデルは1回だけアプリ内にコピーされる。", 13);
        results.setTypeface(Typeface.MONOSPACE);
        results.setTextIsSelectable(true);
        content.addView(results);

        start.setOnClickListener(v -> begin());
        stop.setOnClickListener(v -> {
            cancelRequested = true;
            nativeStage = "中断要求済み";
            stop.setEnabled(false);
            cancelNative(selectedEngine);
        });
        copy.setOnClickListener(v -> {
            ClipboardManager clipboard = (ClipboardManager)
                    getSystemService(Context.CLIPBOARD_SERVICE);
            clipboard.setPrimaryClip(
                    ClipData.newPlainText("PLaMo benchmark", results.getText()));
        });

        ScrollView scroll = new ScrollView(this);
        scroll.addView(content);
        setContentView(scroll);
    }

    @Override
    protected void onDestroy() {
        if (working) cancelNative(selectedEngine);
        handler.removeCallbacks(ticker);
        current.clear();
        super.onDestroy();
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK_MODEL || result != RESULT_OK || data == null) return;
        final Uri uri = data.getData();
        if (uri == null) return;

        results.setText("モデルをコピー中。翻訳推論はまだ開始してない。");
        choose.setEnabled(false);
        new Thread(() -> {
            // モデルを差し替えるので、前のファイルを指すネイティブキャッシュを解放する。
            releaseAllNativeModels();
            File temp = new File(getFilesDir(), "plamo-model-incoming.tmp");
            try (InputStream in = getContentResolver().openInputStream(uri);
                 OutputStream out = new FileOutputStream(temp)) {
                if (in == null) throw new IllegalStateException("モデルを開けない");
                byte[] chunk = new byte[1024 * 1024];
                int count;
                while ((count = in.read(chunk)) != -1) out.write(chunk, 0, count);
                out.flush();
                if (model.exists() && !model.delete())
                    throw new IllegalStateException("元モデルを削除できない");
                if (!temp.renameTo(model))
                    throw new IllegalStateException("モデル保存に失敗");
                runOnUiThread(() -> {
                    results.setText("モデル準備OK: " + model.length() / 1048576
                            + "MiB。エンジンを切り替えて同一条件で比較しよう。");
                    choose.setEnabled(true);
                });
            } catch (Exception e) {
                temp.delete();
                runOnUiThread(() -> {
                    results.setText("コピー失敗: " + e);
                    choose.setEnabled(true);
                });
            }
        }, "model-copy").start();
    }

    private int batteryTemp() {
        Intent result = registerReceiver(null,
                new android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        return result == null ? -1 : result.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1);
    }

    private void begin() {
        if (!model.exists()) {
            results.setText("先にGGUFを選んでね");
            return;
        }
        selectedEngine = engine.getSelectedItemPosition();
        String input = source.getText().toString();
        String language = target.getSelectedItem().toString();
        int nThreads = Integer.parseInt(threads.getSelectedItem().toString());
        int cap = Integer.parseInt(outputCap.getSelectedItem().toString());
        int temperatureBefore = batteryTemp();

        nativeStage = "初期化中";
        cancelRequested = false;
        startedAt = SystemClock.elapsedRealtime();
        results.setText("結果は完了後に表示される。上にリアルタイム進捗を表示中。");
        busy(true);
        handler.removeCallbacks(ticker);
        handler.post(ticker);

        new Thread(() -> {
            String data;
            try {
                // 他方式の重みを解放し、RAMに残すモデルを1種類に制限する。
                releaseOtherNativeModels(selectedEngine);
                data = executeNative(selectedEngine, model.getAbsolutePath(),
                                     input, language, nThreads, cap);
            } catch (Throwable error) {
                data = "ERROR: " + error;
            }
            long elapsed = SystemClock.elapsedRealtime() - startedAt;
            int after = batteryTemp();
            String full = "engine=" + ENGINES[selectedEngine]
                    + "\ndevice=" + android.os.Build.DEVICE
                    + "\nmodel_bytes=" + model.length()
                    + "\ninput_characters=" + input.length()
                    + "\nbattery_temp_before=" + (temperatureBefore / 10f) + " C"
                    + "\nbattery_temp_after=" + (after / 10f) + " C"
                    + "\nwall_time_ms=" + elapsed + "\n\n" + data;
            runOnUiThread(() -> {
                working = false;
                handler.removeCallbacks(ticker);
                progress.setText("処理終了（" + (elapsed / 1000f)
                        + "秒）: " + nativeStage);
                results.setText(full);
                busy(false);
            });
        }, "bench-native").start();
    }

    private static String executeNative(int backend, String path, String input,
                                        String target, int threads, int cap) {
        switch (backend) {
            case 0:
                return nativeBenchmarkGeneric(path, input, target, threads, cap);
            case 1:
                return nativeBenchmarkDotprod(path, input, target, threads, cap);
            case 2:
                return nativeBenchmarkKleidi(path, input, target, threads, cap);
            case 3:
                if (!VULKAN_AVAILABLE) throw new IllegalStateException("Vulkan backend not installed");
                return nativeBenchmarkVulkan(path, input, target, threads, cap);
            case 4:
                if (!HYBRID_AVAILABLE) throw new IllegalStateException("Hybrid backend not installed");
                return nativeBenchmarkHybrid(path, input, target, threads, cap);
            default:
                throw new IllegalArgumentException("Unknown CPU backend " + backend);
        }
    }

    private static void releaseOtherNativeModels(int backend) {
        if (backend != 0) nativeReleaseGeneric();
        if (backend != 1) nativeReleaseDotprod();
        if (backend != 2) nativeReleaseKleidi();
        if (VULKAN_AVAILABLE && backend != 3) nativeReleaseVulkan();
        if (HYBRID_AVAILABLE && backend != 4) nativeReleaseHybrid();
    }

    private static void releaseAllNativeModels() {
        nativeReleaseGeneric();
        nativeReleaseDotprod();
        nativeReleaseKleidi();
        if (VULKAN_AVAILABLE) nativeReleaseVulkan();
        if (HYBRID_AVAILABLE) nativeReleaseHybrid();
    }

    private static void cancelNative(int backend) {
        if (backend == 0) nativeCancelGeneric();
        else if (backend == 1) nativeCancelDotprod();
        else if (backend == 2) nativeCancelKleidi();
        else if (backend == 3 && VULKAN_AVAILABLE) nativeCancelVulkan();
        else if (backend == 4 && HYBRID_AVAILABLE) nativeCancelHybrid();
    }

    private static native String nativeBenchmarkGeneric(
            String model, String input, String target, int threads, int maxTokens);
    private static native String nativeBenchmarkDotprod(
            String model, String input, String target, int threads, int maxTokens);
    private static native String nativeBenchmarkKleidi(
            String model, String input, String target, int threads, int maxTokens);
    private static native void nativeCancelGeneric();
    private static native void nativeCancelDotprod();
    private static native void nativeCancelKleidi();
    private static native void nativeReleaseGeneric();
    private static native void nativeReleaseDotprod();
    private static native void nativeReleaseKleidi();
    private static native String nativeBenchmarkVulkan(
            String model, String input, String target, int threads, int maxTokens);
    private static native void nativeCancelVulkan();
    private static native void nativeReleaseVulkan();
    private static native String nativeBenchmarkHybrid(
            String model, String input, String target, int threads, int maxTokens);
    private static native void nativeCancelHybrid();
    private static native void nativeReleaseHybrid();
}

