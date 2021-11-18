##### Ursprungsvideo: https://www.youtube.com/watch?v=0-4p_QgrdbE
___
**Vorraussetzungen**

- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

- Python 3.8
___

Hinweis:

- Solltest du später mit einer anderen Seite eigene Trainingsdaten erstellen wollen als die, die ich vorschlage (www.makesense.ai), solltest du am Besten vorab einmal in die TrainingsReadMe.md reingucken.
___
## Anleitung

#### Schritt 1:

>- neuen Ordner mit dem Namen "ANPR" erstellen und dort die Dateien vom USB-Stick herüberziehen.

#### Schritt 2:

>* Öffne die beiden Python-Dateien mit einem Editor: Installation.py , Training.py
>* In beiden Datein muss die Variable "personal_path" angepasst werden. Dort muss der der Speicherort, des zuvor erstellten Ordners hinein
>* Diese findest du jeweils ganz oben in der Datei

#### Schritt 3:

>- Öffne eine CMD-Konsole und gehen in den zuvor erstellten Ordner.

>- Gib in die Konsole eine: git clone https://github.com/nicknochnack/TFODCourse

*( sollte git nicht funktionieren, musst du den git Client installieren)*

#### Schritt 4:

>- Nachdem der Klonvorgang fertig ist, gib in die Konsole folgendes ein: py -3.8 -m Installation.py
