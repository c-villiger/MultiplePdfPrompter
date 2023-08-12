@echo off
REM Change directory to your desired location
cd C:\Users\cesar\OneDrive\Desktop\Code\MultiplePdfPrompter

REM Activate the virtual environment
call .venv\Scripts\activate

echo %cd%

REM Run the Streamlit app
streamlit run app_iterated.py

REM Keep the cmd shell open
echo Done.
pause > nul