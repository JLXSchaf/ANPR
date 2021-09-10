Bildannotations: https://www.makesense.ai/


Vorraussetzung: 
			- erfolgreiche Installation


Anleitung: 
	- Navigiere zu dem extra angelgeten Ordner für deine ANPR-System
	- Öffne dort den Ordner: Tensorflow
	- Öffne dort den Ordner: Workspace
	- Öffne dort den Ordner: images
	- Erstelle dort 2 neue Ordner mit den Namen "test" und "train".
	- Nun füge die Dateien zum Trainieren der AI in den "train" Ordner und die Datein zum Testen der AI in den "test" Ordner.
			(- Solltest du noch keine Dateien haben dafür, wird weiter unten erklärt wie man diese erstellt.)
			!!!(Wichtig! Die Seite erstellt die XML-Dateien auf eine bestimmte Art und Weise, deswegen kann man nicht einfach so eine andere Seite nutzen, dazu unten noch mehr)

	- öffne nun eine CMD-Konsole und navigiere dich in den ANPR Ordner
	- Aktiviere die virtuelle Umgebung mit dem Befehl: .\anprsys\Scripts\activate
	- Gebe nun folgendes in die Konsole ein: py -3.8 -m Training.Py
			- möchtest du die Trainingsdauer erhöhen, musst dafür das Script öffnen und in Zeile 48 "--num_train_Steps" erhöhen.
	
	- Nachdem das Training zu Ende ist, musst du in der Datei getNumberplate.py etwas abändern.
	- In Zeile 1 musst du die Variable abändern auf den letzten Checkpoint von deiner AI.
	  Diesen findest du in dem ANPR-Ordner. Öffne diesen und naviegere dann zu den Unterordnern : Tensorflow, workspace, models, my_ssd_mobnet.
	- Dort guckst du dir die vorletzte Datei an. Ihr Name sollte ungefährt "ckpt-{XX}.index" sein. Dort wo bei mir die beiden XX stehen,
	  sollte bei dir eine Zahl stehen, diese musst du in die Datei eintragen.
	


Erstellung der Test- und Trainingsdaten:
	- Gehe auf die Seite: https://www.makesense.ai/
	- Klicke unten rechts auf "Get Started".
	- Nun verändert sich die Seite und zeigt dir ein Feld an wo du die Bilder hochladen kannst, die die du zum Trainieren und Testen nutzen willst.
	- Sobald du die gewünschten Bilder ausgewählt und hinzugefügt hast, kannst du auf "Object Detection" oder "Image recognition" klicken. Klicke auf "Object Detection".
	- Die Seite verändert sich erneut und dir wird ein kleines Fenster mit "Create Labels" angezeigt. Erstelle dort nun das Label mit dem Namen: licence
	- Klicke nun auf "Start project".
	- Nun wird dir eines der Bilder anzeigt, die du zuvor hochgeladen hast. Dein Cursor dient nun als Werkzeug zur Markierung.
	- Ziehe eine Rechteck um das Nummernschild.
	- Rechts vom Bild wird dir nun "Rect" angezeigt und ein Objekt, wo du noch ein Label auswählen sollst. Wähle das zuvor erstellte Label "licence" aus.
	- Solltest du mehrere Bilder hochgeladen haben, kannst du Links vom Bild zwischen den Bildern hin und herwechseln. 
	  Führe nun für alle Bilder die 2 vorherigen Schritte aus.
	- Zum Runterladen der nun erstellten Annotationen, klicke oben auf "Actions". Nun öffnet sich ein Reiter, klicke in diesem auf "Export Annotations".
	  Nun öffnet sich ein neues Fenster mit den Namen "Export rect annotations". Wähle dort die zweite Option "A .zip package containing files in VOC XML format." aus.
	  Klicke zum Schluss auf Exportieren.
	- Entpacke die gedownloadete Datei und zieh alle Dateien aus dem Ordner zu den zuvor ausgewählten Bildern.
	- Eine Test- oder Trainingsdatei besteht aus dem Bild und der dazugehörigen XML Datei. Beide teilen sich den selben Namen, nur die Endung ist unterschiedlich.


Warnung:
	- Die Seite "https://www.makesense.ai/ legt die XML Dateien auf eine bestimmte Art an.
	- Öffne eine XML-Datei.
	- wichtig in der XML-Datei ist dieser Absatz:

			<object>
				<name>licence</name>
				<pose>Unspecified</pose>
				<truncated>Unspecified</truncated>
				<difficult>Unspecified</difficult>
				<bndbox>
					<xmin>729</xmin>
					<ymin>558</ymin>
					<xmax>952</xmax>
					<ymax>623</ymax>
				</bndbox>
			</object>

	- Dem Model ist die <bndbox> sehr wichtig, da dies ihm anzeigt, wo das Nummernschild tatsächlich im Bild ist.
	- Dem Model wird in der Trainingkonfiguration mitgegeben, dass die Koordinaten der <bndbox> an der 4ten Stelle ist. (0,1,2,3,4)
	- Je nach Seite oder Tool, dass man nutzt kann das auch anders sein.

	- Falls deine Trainingsdateien die Box nicht an der 4ten stelle haben, kommt hier eine kurze Anleitung:
		- Im Installationsskript gibt es im Schritt 4 die Methode "createTFRecords()".
		- Sie Downloaded die Datei: generate_tfrecord.py
		- Dort musst du in Zeile 88-91 folgendes abändern.
		- Du musst die 4 ersätzen durch die Position deiner <bndbox> in deinen Trainingsdatein.
			int(member[4][0].text),
                     	int(member[4][1].text),
                    	int(member[4][2].text),
                     	int(member[4][3].text)
		Tipp: Lasse dein Programm ein Stop machen, nachdem er diese Datei gedownloadet hat. Dann kannst du alles in Ruhe abändern, weil später das einfach 
		      nachträglich zuändern und das Installationsskript nocheinmal laufen zu lassen funktioniert nicht.  