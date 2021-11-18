### Anleitung
>- Kopiere das Bild und patziere es in den richtigen Ordner. Der Ordnerpfad ist: /ANPR/Tensorflow/workspace/images/test/   .
>- Öffne eine CMD-Konsole und nagiviere dich in den ANPR Ordner.
>- Aktiviere die virtuelle Umgebung mit dem Befehlt: .\anprsys\Scripts\activate
>- Gebe nun folgendes in die Konsole ein: py -3.8  getNumberplate.py Bildname.Bildformat

>- Das Programm wird die 2 Antworten geben.
>- Antwort 1: jeglicher Text der auf dem Nummernschild vorhanden ist zB.: XXX-XX-1234, DE
>>- Antwort 2: der Text, der mit der höchsten Wahrscheinlichkeit das Kennzeichen ist. Dafür nimmer er die Proportion von der Textgröße zur Gesamtgröße des Nummernschildes und gibt dir den größten Text aus. 




#### Hinweis
>- Vergiss nciht das du die Variable cpt anpassen musst in dem Trainingstutorial.
>- passender Fehler dazu ( tensorflow.python.framework.errors_impl.NotFoundError: Could not find checkpoint or SavedModel at Tensorflow\workspace\models\my_ssd_mobnet\ckpt-40. )
