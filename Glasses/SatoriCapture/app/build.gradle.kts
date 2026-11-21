plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    jacoco
}

android {
    namespace = "com.satori.glasses.capture"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.satori.glasses.capture"
        // Rokid Glasses run YodaOS-Sprite on Android 12 (this unit reports API 32).
        minSdk = 31
        targetSdk = 34
        ndk { abiFilters += "arm64-v8a" }
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Set by Tools/rokid_glasses.py build/update so every build installs over the last one and can be told apart
        // (versionCode = minutes since 2026-01-01 UTC, above the last build on that PC and the installed one); a plain
        // Gradle build keeps 1 / 0.1.0.
        versionCode = (project.findProperty("satoriVersionCode") as String?)?.toInt() ?: 1
        versionName = (project.findProperty("satoriVersionName") as String?) ?: "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        buildConfig = true
    }
}

dependencies {
    val camerax = "1.3.4"
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-service:2.7.0")
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("org.opencv:opencv:4.10.0")
    implementation("com.alphacephei:vosk-android:0.3.75@aar")
    implementation("net.java.dev.jna:jna:5.18.1@aar")
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    androidTestImplementation("androidx.test:runner:1.5.2")
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
}

jacoco { toolVersion = "0.8.12" }
tasks.register<JacocoReport>("standaloneCoverage") {
    dependsOn("testDebugUnitTest")
    executionData(fileTree(layout.buildDirectory).include("jacoco/testDebugUnitTest.exec"))
    classDirectories.setFrom(fileTree("${layout.buildDirectory.get()}/tmp/kotlin-classes/debug") {
        include("**/StandaloneEngine*", "**/EngineRequest*", "**/StandaloneFrame*", "**/FlowKeyframePolicy*")
    })
    sourceDirectories.setFrom(files("src/main/java"))
    reports { xml.required.set(true); html.required.set(true) }
}
