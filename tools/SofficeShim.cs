using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

public static class SofficeShim
{
    public static int Main(string[] args)
    {
        string script = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "soffice.ps1");
        string quoted = string.Join(" ", args.Select(value => "\"" + value.Replace("\"", "\\\"") + "\""));
        var start = new ProcessStartInfo("powershell.exe", "-NoProfile -ExecutionPolicy Bypass -File \"" + script + "\" " + quoted);
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        using (var process = Process.Start(start))
        {
            Console.Write(process.StandardOutput.ReadToEnd());
            Console.Error.Write(process.StandardError.ReadToEnd());
            process.WaitForExit();
            return process.ExitCode;
        }
    }
}
