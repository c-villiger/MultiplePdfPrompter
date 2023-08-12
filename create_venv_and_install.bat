@echo off


echo *****************************
echo ****     Create venv     ****
echo *****************************
REM Check if .venv folder exists
if not exist .venv (
    echo Creating .venv virtual environment...
    python -m venv .venv
) else (
    echo .venv virtual environment already exists.
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Add space
echo.

echo ******************************
echo **** Install requirements ****
echo ******************************
REM Check if requirements.txt exists and install packages
if exist requirements.txt (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
    echo Packages installed successfully.
) else (
    echo No requirements.txt file found.
    echo Creating an empty requirements.txt file...
    echo. > requirements.txt
    echo Empty requirements.txt file created.
)


REM Add space
echo.


REM Keep the cmd shell open
echo Done.
pause > nul
