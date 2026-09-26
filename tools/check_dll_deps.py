"""Inspect native DLL imports for a quick dep audit."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pefile

paths = sys.argv[1:] or [
    r"C:\Users\15216\Desktop\autolabel-dock\dist\AutoLabelDock\_internal\torch\lib\c10.dll",
    r"C:\Users\15216\Desktop\autolabel-dock\dist\AutoLabelDock\_internal\torch\lib\libiomp5md.dll",
    r"C:\Users\15216\Desktop\autolabel-dock\dist\AutoLabelDock\_internal\torch\lib\torch_cpu.dll",
]
for path in paths:
    print(f"=== {path} ===")
    pe = pefile.PE(path)
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode(errors='replace')
            print(f"  import: {dll}")
    if hasattr(pe, 'DIRECTORY_ENTRY_DELAY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_DELAY_IMPORT:
            dll = entry.dll.decode(errors='replace')
            print(f"  delay:  {dll}")
    print()
