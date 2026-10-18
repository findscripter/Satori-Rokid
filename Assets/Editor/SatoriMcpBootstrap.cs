using System;
using System.IO;
using System.Threading.Tasks;
using MCPForUnity.Editor.Services;
using MCPForUnity.Editor.Services.Transport;
using UnityEditor;
using UnityEngine;

// Project-local bootstrap for the MCP connection explicitly requested by the user.
[InitializeOnLoad]
public static class SatoriMcpBootstrap
{
    private static bool connecting;
    private static double nextAttempt;
    private static int attempts;

    static SatoriMcpBootstrap()
    {
        EditorApplication.update += Tick;
    }

    private static void Tick()
    {
        if (connecting || EditorApplication.isCompiling || EditorApplication.isUpdating ||
            EditorApplication.timeSinceStartup < nextAttempt || attempts >= 20)
            return;
        if (MCPServiceLocator.TransportManager.IsRunning(TransportMode.Http))
        {
            EditorApplication.update -= Tick;
            return;
        }
        attempts++;
        nextAttempt = EditorApplication.timeSinceStartup + 5;
        Connect();
    }

    [MenuItem("Satori/Connect Unity MCP")]
    public static async void Connect()
    {
        if (connecting) return;
        connecting = true;
        try
        {
            EditorPrefs.SetBool("MCPForUnity.UseHttpTransport", true);
            EditorPrefs.SetString("MCPForUnity.HttpTransportScope", "local");
            EditorPrefs.SetString("MCPForUnity.HttpUrl", "http://127.0.0.1:8080");
            bool ok = await MCPServiceLocator.TransportManager.StartAsync(TransportMode.Http);
            Directory.CreateDirectory("Evidence");
            File.WriteAllText("Evidence/mcp-bootstrap.json", JsonUtility.ToJson(new Result
            {
                connected = ok,
                unityVersion = Application.unityVersion,
                projectPath = Path.GetFullPath("."),
                checkedUtc = DateTime.UtcNow.ToString("o")
            }, true));
            Debug.Log("[Satori] Unity MCP connection: " + ok);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("[Satori] MCP connection not ready: " + ex.GetType().Name);
        }
        finally { connecting = false; }
    }

    [Serializable]
    private class Result
    {
        public bool connected;
        public string unityVersion;
        public string projectPath;
        public string checkedUtc;
    }
}
