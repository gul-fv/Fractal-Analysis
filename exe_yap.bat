@echo off
pyinstaller --noconsole --onefile --add-data "my_model.pt;." main.py
pause