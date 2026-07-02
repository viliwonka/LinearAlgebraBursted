// Minimal stand-in for the slice of UnityEngine that TemplateConverter.cs / GenUtils.cs touch
// (Debug.Log / Debug.LogError). Routed to the console so the messages that would otherwise land
// in the Unity Editor log are still visible when running headlessly.
namespace UnityEngine
{
    internal static class Debug
    {
        public static void Log(object message)
        {
            System.Console.Out.WriteLine(message);
        }

        public static void LogError(object message)
        {
            System.Console.Error.WriteLine(message);
        }
    }
}
