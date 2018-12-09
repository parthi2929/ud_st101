REM this is to delete all the irrelevant folders and files in all folders so as to prepare for pushing to git

cd ..
FOR /d /r . %%d IN (.ipynb_checkpoints) DO @IF EXIST "%%d" rd /s /q "%%d"
cd wp_tools 