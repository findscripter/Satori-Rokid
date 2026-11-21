package com.satori.glasses.capture

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.KeyEvent
import android.view.View
import android.view.WindowManager

/**
 * Starts (or reconfigures) CaptureService and gets out of the way: black on the waveguide is transparent, and the
 * activity finishes unless launched with `--ez stay true` (then it forwards touchpad key events for debugging).
 * `--ez stop true` stops the service. Config extras are passed through, see CaptureConfig.fromIntent.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(View(this).apply { setBackgroundColor(Color.BLACK) })
        handle(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handle(intent)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCapture(intent)
        } else {
            finish()
        }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        if (event.repeatCount == 0) CaptureService.instance?.onKey(keyCode, "down")
        return if (keyCode == KeyEvent.KEYCODE_BACK) super.onKeyDown(keyCode, event) else true
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        CaptureService.instance?.onKey(keyCode, "up")
        return if (keyCode == KeyEvent.KEYCODE_BACK) super.onKeyUp(keyCode, event) else true
    }

    private fun handle(intent: Intent) {
        if (intent.action == Intent.ACTION_MAIN && intent.hasCategory(Intent.CATEGORY_LAUNCHER)) {
            startActivity(Intent(this, StandaloneActivity::class.java))
            finish()
            return
        }
        if (intent.getBooleanExtra("stop", false)) {
            startService(Intent(this, CaptureService::class.java).setAction(CaptureService.ACTION_STOP))
            finish()
            return
        }
        val missing = REQUIRED.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) startCapture(intent) else requestPermissions(missing.toTypedArray(), REQUEST_CODE)
    }

    private fun startCapture(source: Intent) {
        stopService(Intent(this, StandaloneService::class.java))
        val service = Intent(this, CaptureService::class.java)
        source.extras?.let { service.putExtras(it) }
        startForegroundService(service)
        if (!source.getBooleanExtra("stay", false)) finish()
    }

    companion object {
        private const val REQUEST_CODE = 1
        private val REQUIRED = listOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
    }
}
