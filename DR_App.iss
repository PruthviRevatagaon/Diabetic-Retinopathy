[Setup]
AppName=DR Severity Classifier
AppVersion=1.0
DefaultDirName={pf}\DRClassifier
DefaultGroupName=DRClassifier
UninstallDisplayIcon={app}\app.ico
OutputDir=..\
OutputBaseFilename=DRClassifierInstaller
Compression=lzma
SolidCompression=yes

[Files]
Source: "app.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_and_run.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

Source: "optuna_models\*"; DestDir: "{app}\optuna_models"; Flags: recursesubdirs createallsubdirs
Source: "runs\*"; DestDir: "{app}\runs"; Flags: recursesubdirs createallsubdirs
Source: "DR\*"; DestDir: "{app}\DR"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Run DR Classifier"; Filename: "{app}\install_and_run.bat"

[UninstallDelete]
Type: files; Name: "{app}\venv"