"""Inspect frozen PKG archive TOC."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\15216\Desktop\autolabel-dock\venv\Lib\site-packages')
from PyInstaller.archive.readers import CArchiveReader
ar = CArchiveReader(r'C:\Users\15216\Desktop\autolabel-dock\dist\AutoLabelDock\AutoLabelDock.exe')
# List a few interesting entries
keys = list(ar.toc.keys())
print(f'total entries: {len(keys)}')
# Show torch top-level
torch_entries = [k for k in keys if k.startswith('torch') and k.count('/') <= 2 and not k.startswith('torch.nn') and not k.startswith('torch.utils') and not k.startswith('torch.optim')]
for k in torch_entries[:30]:
    print(' ', k)
print()
# Show DLL entries
dll_entries = [k for k in keys if k.lower().endswith('.dll')]
print(f'dll entries: {len(dll_entries)}')
for k in dll_entries[:30]:
    print(' ', k)
