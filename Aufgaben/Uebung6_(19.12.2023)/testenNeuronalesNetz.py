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

model = NeuralNetwork() #Eine Instanz des neuronalen Netzes wird erstellt
model.load_state_dict(torch.load('Handschrifterkennung.pth')) #Ein bereits Trainiertes Netz aus dem Verzeichnis mit dem Titel "Handschrifterkennung.pth" wird geladen
model.eval() #Stellen sicher, dass das Modell im Evaluationsmodus ist (nicht im Trainingsmodus)


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

criterion = nn.CrossEntropyLoss()  # Kreuzentropie-Verlustfunktion für Klassifikation -> mit criterion wird der Fehler des neuronalen Netzes berechnet
optimizer = optim.Adam(model.parameters(), lr=0.001)  #Definiert den Optimierungsalgorithmus fuer das Training des neuronalen Netzes. lr legt die Lernrate fest, wie gross die Schritte waehrend des Optimierungsvorganges sind


#Die Genauigkeit des neuronalen Netzes auf das Testdatenset wird berechnet
correct = 0 #Speichert die Anzahl der korrekt zugeordneten Daten
total = 0 #Speichert die Gesamtanzahl dergetesteten Daten
with torch.no_grad(): #Gradienten werden beim Testen nicht benoetigt. Es wird sichergestellt, dass keine Gradienten fuer die Berechnungen gespeichert werden
    for images, labels in test_loader: #Fuer jeden Batch im Testdatensatz mache:
        outputs = model(images) #berechnet die Vorhersagen des neuronalen Netzes fuer die Testdaten
        _, predicted = torch.max(outputs.data, 1) #wird verwendet, um das wahrscheinlichste Ergebnis fuer jede Eingabe zu erhalten
        total += labels.size(0) #Anzahl der Daten im Batch wird zur Gesamtanzahl dazu addiert.
        correct += (predicted == labels).sum().item() #hier wird ueberprueft, wie viele Vorhersagen mit den tatsaechlichen Daten uebereinstimmen

accuracy = correct / total #Genauigkeit wird berechnet
print(f'Genauigkeit des Modells auf dem Testdatensatz: {accuracy:.2%}')

#Optional: Visualisierung einiger Vorhersagen des Modells und Vergleich mit den tatsaechlichen Labels (Tatsaechlichen Zahl)
import matplotlib.pyplot as plt
import numpy as np

# Einzelnen Batch von Testdaten aus dem DataLoader nehmen
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Modellvorhersagen erhalten
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Umwandeln von Torch Tensoren in NumPy Arrays für die Visualisierung
images = images.numpy()
predicted = predicted.numpy()
labels = labels.numpy()

# Visualisierung der Vorhersagen mit den Bildern
num_images = len(images)
num_rows = 11
num_cols = 6
plt.figure(figsize=(12, 6))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(np.squeeze(images[i]), cmap='gray')
    plt.title(f'Predicted: {predicted[i]}, Actual: {labels[i]}', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()