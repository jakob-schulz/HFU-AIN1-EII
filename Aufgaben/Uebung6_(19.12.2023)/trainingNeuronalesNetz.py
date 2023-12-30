#Importieren der erforderlichen Bibliotheken
import torch 
import torch.nn as nn #wird benoetigt zum erstellen und Trainieren des neuronalen Netzes
import torch.optim as optim #wir benoetigt, um das nueronale Netz zu optimieren und zu trainieren bzw. zu verbessern
import torchvision.datasets as datasets #wird benoetigt, um die Datensaetze fuer die Bilderkennung zu laden
import torchvision.transforms as transforms #dient dazu die Eingangsbilder vor oder während des Trainingsprozesses des neuronalen Netzes zu verändern oder anzupassen
from torch.utils.data import DataLoader #wird benoetigt, um Daten fuer das Training von neuronalen Netzten zu verwalten

#Erstellen eines einfachen Feedforward neuronalen Netzes
class NeuralNetwork(nn.Module): #Klasse erbt von nn.Module
    def __init__(self): #Konstruktor
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) #In den folgenden 3 Befehlen werden die 28*28 Eingabedimensionen (die Handschrift) auf 10 Ausgangsdimensionen transformiert, sodass man dann im Training eine passende Aktivierungsfunktion findet, mit der man dann die Daten richtig zuordnen kann. 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # Ausgabe hat 10 Klassen (0-9)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): #repraesentiert, wie die Daten durch das neuronale Netz fliessen
        x = x.view(-1, 28*28)  # Flattening des Eingangs  -> Eingangsdaten werden in das richtige Format gebracht
        x = self.relu(self.fc1(x)) # Daten durchlaufen jede neuronale Schicht, welche beim konstruktor definiert wurde. Zusätzlich wird die Aktivierungsfunktion angewendet
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x)) 
        return x

model = NeuralNetwork() #Eine Insatnz des neuronalen Netzes wird erstellt


#Es wird angegeben, wie die Daten transformiert werden muessen
transform = transforms.Compose([
    transforms.ToTensor(), #Die Eingabeobjekte (Handschriften) werden in Tensoren umgewandelt und abgespeichert. Tensor ist eine Datenstruktur
    transforms.Normalize((0.5,),(0.5,)) #Die Daten der Bilder werden normalisiert, also in einen Bereich gebracht, der für das Netzwerktraining geeignet ist
])

#Der MINST Datensatz wird heruntergeladen und im aktuellen Verzeichnis gespeichert, falls er noch noch nicht lokal vorhanden ist
train_dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True) #Datensatz wird dann transformiert und in einem Trainingsdatenset gespeichert
test_dataset = datasets.MNIST(root='.', train=False, transform=transform) #Datensatz wird transformiert und in einem Testdatenset gespeichert

batch_size = 64 #Anzahl der Elemente, die in einem Batch verarbeitet werden. Ein Batch ist eine Teilmenge der Daten, die gleichzeitig verarbeitet werden. Dies steigert u. a. die Effizienz 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) #Es wird ein Dataloader erstellt, der die Trainingsdaten des neuronalen Netzes zufällig in Batches mischt 
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) #Es wird ein Dataloader erstellt, der die Daten immer in gleicher Reinhenfole in die Batches verteilt. Mit diesm wird das neuronale Netz getestet

#Geeignete Loss-Funktion und Optimierer werden gewaehlt
criterion = nn.CrossEntropyLoss()  # Kreuzentropie-Verlustfunktion für Klassifikation -> mit criterion wird der Fehler des neuronalen Netzes berechnet
optimizer = optim.Adam(model.parameters(), lr=0.001)  #Definiert den Optimierungsalgorithmus fuer das Training des neuronalen Netzes. lr legt die Lernrate fest, wie gross die Schritte waehrend des Optimierungsvorganges sind



#Training des neuronalen Netzes
num_epochs = 15 #Definiert die Anzahl der Durchlaeufe des gesamten Trainingsdatensatz -> wird auch als Epochen bezeichnet

for epoch in range(num_epochs): #Fuer jede Epoche mache:
    running_loss = 0.0  #Anzahl der Verluste werden auf 0 gesetzt
    for i, (images, labels) in enumerate(train_loader): #Nimm jeden Batch des Trainingsdatensatzes und gebe mir hiervon alle Handschriften und die zugehoerigen labels (Bedeutung der Zahl)
        optimizer.zero_grad() #Setzt die Gradienten der Optimierungsvariablen auf Null, um sicherzustellen, dass bei jedem Schritt die vorherigen Gradienten nicht berücksichtigt werden.
        outputs = model(images) #Berechnet die Vorhersagen des Modells fuer die Daten des Batches
        loss = criterion(outputs, labels) #Berechnet den Verlust, anhand der definierten Verlustfunktion (also die falsche Zuordnung der Zahlen)
        loss.backward() #Berechnet die Gradienten der Verlustfunktion bezüglich der Modellparameter
        optimizer.step() #passt den Optimierungsalgorithmus im Bezug auf die Gradienten an.

        running_loss += loss.item() #Addiert den aktuellen Verlust zu dem Gesamtverlust der Epoche
        if (i+1) % 100 == 0: #: Überprüft, ob 100 Batches verarbeitet wurden. Wenn ja, wird der aktuelle Verlust für diese Batches ausgegeben, um den Fortschritt während des Trainings zu überwachen.
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0 #Verlust wird zurueckgesetzt

torch.save(model.state_dict(), 'Handschrifterkennung.pth')