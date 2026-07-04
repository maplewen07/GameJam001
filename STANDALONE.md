# Standalone camera build

目标交付物是一个单文件 Windows 可执行程序：

```powershell
dist\GameJamCamera.exe
```

这个 exe 会包含 Python、OpenCV、Numpy 和项目里的摄像机流程代码。别人拿到 exe 后不需要安装 Python 包或复制本项目目录。

## Build on Windows

在项目根目录运行：

```powershell
PowerShell -ExecutionPolicy Bypass -File .\build_windows_onefile.ps1
```

生成完成后，把下面这个文件发给别人：

```powershell
dist\GameJamCamera.exe
```

## Run

双击 `GameJamCamera.exe`，或在 PowerShell 里运行：

```powershell
.\GameJamCamera.exe
```

常用参数会继续透传给原来的摄像机流程：

```powershell
.\GameJamCamera.exe --scan-cameras 5
.\GameJamCamera.exe --camera 1
.\GameJamCamera.exe --no-record-video
```

预览窗口快捷键：

- `c`: 重新开始标定和人员初始化
- `h`: 开关 CLAHE
- `q` 或 `Esc`: 退出

## Notes

单文件 exe 不依赖本项目目录，但仍然需要目标电脑有可用摄像头、系统摄像头权限，以及对应摄像头驱动。第一次启动时，PyInstaller 单文件程序会先解包到系统临时目录，所以可能会慢几秒。
