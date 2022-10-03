import unicodedata
import re
import random
import numpy as np
from pyinflect import getInflection
import itertools


import torch

SOS_token = 0
EOS_token = 1

grammatica1 = (["read", "write", "comment", "review", "learn", "copy", "quote", "translate", "publish", "praise", "love", "throw", "discuss", "cite", "understand"],
               ["begin", "start", "continue", "finish", "stop", "end", "cease", "delay", "enjoy", "survive", "accelerate", "advance", "complete", "relish", "prolong"],
               ["book", "memo", "paper", "thesis", "article", "testament", "lyrics", "script", "letter", "proof", "curriculum", "protocol", "newspaper", "survey", "law"],
               ["speech", "break", "collaboration", "replacement", "publication", "foundation", "performance", "partnership", "effort", "project", "operation", "job", "search", "research", "scholarship"],
               ['transitivo_semplice', 'aspettuale_semplice', 'aspettuale_con_transitivo'],
               [.6, .15, .25],
               [(0,2), (1,3), (1,0,2)])

coppie_tipiche1 = {'thesis': 'write', 'paper': 'review', 'book': 'read'}

def unicodeToAscii(stringa):
    """
    :param stringa: stringa
    :return: stringa in ascii
    """
    return ''.join(c for c in unicodedata.normalize('NFD', stringa) if unicodedata.category(c) != 'Mn')


def normalizza_stringa(stringa):
    """
    :param stringa: stringa
    :return: stringa normalizzata (in ascii, senza spazi e caratteri \n all'inizio e alla fine, senza maiuscole, senza caratteri che non sono lettere)
    """
    stringa = unicodeToAscii(stringa.lower().strip())
    stringa = re.sub(r"[^a-zA-Z.!?]+", r" ", stringa)
    return stringa


class Linguaggio:
    def __init__(self, nome):
        self.nome = nome
        self.parola2indice = {}
        self.parola2frequenza = {}
        self.indice2parola = {0: "SOS", 1: "EOS"}
        self.numero_parole = 2

    def aggiungiFrase(self, frase):
        for parola in frase.split(' '):
            self.aggiungiParola(parola)

    def aggiungiParola(self, parola):
        if parola not in self.parola2indice:
            self.parola2indice[parola] = self.numero_parole
            self.parola2frequenza[parola] = 1
            self.indice2parola[self.numero_parole] = parola
            self.numero_parole += 1
        else:
            self.parola2frequenza[parola] += 1


def carica_dati(path_dati, reverse):
    """
    :param path_dati: path dove si trova file dove ciascuna riga è del tipo stringa1\tstringa2
    :param reverse: specifica se si vuole che il primo elemento di ciascuna tupla sia la seconda sequenza nella riga del file
    :return: lista dove ciascun elemento è una tupla (stringa1, stringa2)
    """

    coppie = [(riga.split('\t')[0], normalizza_stringa(riga.split('\t')[1])) for riga in open(path_dati).readlines()]
    if not reverse:
        return coppie
    else:
        return [list(reversed(coppia)) for coppia in coppie]

def aggiorna_linguaggi(linguaggio_input, linguaggio_output, dati):
    """
    :param linguaggio_input: istanza di Linguaggio per il linguaggio che è l'input della rete
    :param linguaggio_output: istanza di Linguaggio per il linguaggio che è l'output della rete
    :param dati: lista di tuple dove ciascuna tupla contiene una frase di input e la corrispondente frase di output
    :return: niente, aggiorna gli oggetti Linguaggio
    """
    for coppia in dati:
        linguaggio_input.aggiungiFrase(coppia[0])
        linguaggio_output.aggiungiFrase(coppia[1])


def tensore_da_frase(linguaggio, frase):
    """
    :param linguaggio: istanza di Linguaggio
    :param frase: frase
    :return: tensore dove ciascun elemento è l'indice della parola della frase nel linguaggio
    """
    indici = [linguaggio.parola2indice[parola] for parola in frase.split(' ')]
    indici.append(EOS_token)
    return torch.tensor(indici, dtype=torch.long).view(-1, 1)  #  torch.Size([lunghezza frase, 1])

def trova_lunghezza_massima(coppie_di_tensori):
    """
    :param coppie_di_tensori: lista di tuple dove il primo elemento è il tensore di una frase nella lingua input e il secondo è il tensore della frase corrispondente nella lingua output
    :return: la lunghezza massima delle sequenze nella lingua input
    """
    lunghezza_massima = 0
    for coppia in coppie_di_tensori:
        if coppia[0].shape[0] > lunghezza_massima:
            lunghezza_massima = coppia[0].shape[0]
    return lunghezza_massima

def genera_dati_funzione_semantica(coppie):
    """
    :param coppie: lista dove ciascun elemento è una tupla (stringa1, stringa2)
    :return: lista dove ciascun elemento è una tripla (stringa1, stringa2, 0 se la coppia è un prodotto del processo di generazione random di esempi, 1 se era nel dataset originario)
    """
    nuove_coppie = []
    for indice, coppia in enumerate(coppie):
        nuove_coppie.append((coppia[0], coppia[1], 1))
        coppia_random = random.choice(coppie)
        while coppia[0] == coppia_random[0]:
            coppia_random = random.choice(coppie)
        nuove_coppie.append((coppia[0], coppia_random[1], 0))
    return nuove_coppie

def genera_vincoli(lista1, lista2, numero_elementi_lista2_selezionati):
    """
    LA FUNZIONE E' UTILE PER LA FUNZIONE GENERA_FILES_DATASET
    Date due liste crea una lista di vincoli (tuple che non ricorrono nel dataset di addestramento e che saranno usate per il testing) seguendo
    un metodo che consiste nel, per ciascun elemento della lista1, selezionare in maniera casuale n elementi della lista2 con cui l'elemento della lista1 non può comparire
    :param lista1: lista1
    :param lista2: lista2
    :param numero_elementi_lista2_selezionati: numero di elementi selezionati in modo casuale dalla lista2 con cui l'elemento della lista1 non può comparire
    :return: lista di vincoli (tuple (elem_lista1, elem_lista2), se un vincolo riguarda il verbo aspettuale e coppie verbo-nome entità, allora la tupla è del tipo (elem_lista1, (elem_lista2))
    """
    vincoli = []
    for elemento_lista1 in lista1:
        elementi_lista2_selezionati = lista2[np.random.choice(len(lista2), numero_elementi_lista2_selezionati, replace=False)]
        for elemento_lista2 in elementi_lista2_selezionati:
            if isinstance(elemento_lista2, str):
                vincoli.append(tuple([elemento_lista1]+[elemento_lista2]))  #  iniziale conversione in liste necessaria per concatenazione, conversione in tuple necessaria per creare insiemi a partire dalla lista finale
            else:
                vincoli.append(tuple([elemento_lista1]+list(elemento_lista2)))
    return vincoli

def traduci_sequenza_simboli(sequenza_simboli):
    """
    LA FUNZIONE E' UTILE PER LA FUNZIONE GENERA_FILES_DATASET
    La funzione restituisce la traduzione in linguaggio naturale di una sequenza di simboli
    :param sequenza_simboli: possibili forme sono ('read', 'book'), ('begin', 'speech'), ('begin', 'read', 'book')
    :return: traduzione in linguaggio naturale di una sequenza di simboli nella forma (traduzione)
    (se il simbolo non terminale è aspettuale_con_transitivo, viene restituita la tupla (traduzione implicita, traduzione esplicita))
    """
    if len(sequenza_simboli) == 2:
        traduzione_linguaggio_naturale = ('John '+getInflection(sequenza_simboli[0], tag='VBD')[0]+' the '+sequenza_simboli[1] + ' .', )
    elif len(sequenza_simboli) == 3:
        traduzione_linguaggio_naturale = ('John '+getInflection(sequenza_simboli[0], tag='VBD')[0]+' the '+sequenza_simboli[2] + ' .',
                                          'John '+getInflection(sequenza_simboli[0], tag='VBD')[0] + ' '+getInflection(sequenza_simboli[1], tag='VBG')[0] + ' the ' + sequenza_simboli[2] + ' .')
    return traduzione_linguaggio_naturale

def genera_files_dataset(path_file_addestramento_uniforme,
                         path_file_addestramento_tipicalita,
                         path_file_valutazione_funzione_semantica,
                         path_file_valutazione_parlante_letterale,
                         grammatica = grammatica1,
                         numero_esempi_dset_addestramento = 50000,
                         proporzione_possibili_combinazioni_da_escludere = 3,
                         coppie_tipiche = coppie_tipiche1,
                         probabilita_verbo_tipico_dato_nome = .8,
                         probabilita_metonimia=.5
                         ):
    """
    :param path_file_addestramento_uniforme: path dove verrà salvato il file con il dataset uniforme per l'addestramento dei modelli
    (formato sequenza simboli\ttraduzione linguaggio\n)
    :param path_file_addestramento_tipicalita: path dove verrà salvato il file con il dataset tipicalità per l'addestramento dei modelli
    :param path_file_valutazione_funzione_semantica: path dove verrà salvato il file con il dset per la valutazione della funzione semantica
    (formato sequenza_simboli\ttraduzione linguaggio\n)
    :param path_file_valutazione_parlante_letterale: path dove verrà salvato il file con il dset per il parlante letterale
    (formato sequenza_simboli\ttraduzione1 linguaggio\ttraduzione2 linguaggio\n)
    :param grammatica: grammatica del linguaggio di simboli. Lista di liste (liste di simboli terminali + lista con simboli non terminali + lista con probabilità di ciascun simbolo non terminale)
    :param numero_esempi_dset_addestramento: numero di volte in cui vogliamo generare una sequenza sfruttando le regole della grammatica
    :param proporzione_possibili_combinazioni_da_escludere: per la generazione di vincoli. Dato ciascun verbo/verbo aspettuale/verbo aspettuale, viene creato un vincolo
    con 1/3 dei possibili nomi entità/nomi evento/(verbo, nome entità)
    :param coppie_tipiche: valore di nome con distribuzione di probabilità condizionata p(verbo|nome=nome) non uniforme
    :param probabilita_verbo_tipico_dato_nome: p(verbo|nome=nome)
    :param probabilita_metonimia: data una sequenza di simboli del tipo aspettuale con transitivo, probabilità che questa venga tradotta con metonimia nel dset di addestramento
    :return: niente, produce quattro datasets per l'addestramento e la valutazione
    """
    """
    Generazione vincoli (combinazioni da usare per la valutazione) linguaggio di simboli
    lista in formato [(verbo, nome_entita), (verbo_aspettuale, nome_evento), (verbo_aspettuale, verbo, nome_entita)]
    """

    insieme_verbi = grammatica[0]
    insieme_verbi_aspettuali = grammatica[1]
    insieme_nomi_entita = grammatica[2]
    insieme_nomi_evento = grammatica[3]
    produzioni_linguaggio_simboli = grammatica[4]
    probabilita_produzioni_linguaggio_simboli = grammatica[5]

    vincoli = []
    vincoli_verbi_entita = genera_vincoli(np.array(insieme_verbi), np.array(insieme_nomi_entita), len(insieme_nomi_entita)//proporzione_possibili_combinazioni_da_escludere)
    vincoli.extend(vincoli_verbi_entita)
    vincoli.extend(genera_vincoli(np.array(insieme_verbi_aspettuali), np.array(insieme_nomi_evento), len(insieme_nomi_evento)//proporzione_possibili_combinazioni_da_escludere))
    prodotto_cartesiano_verbi_nomi_entita = [(verbo, nome_entita) for verbo in insieme_verbi for nome_entita in insieme_nomi_entita]  #  lista di tuple
    vincoli.extend(genera_vincoli(np.array(insieme_verbi_aspettuali), np.array(prodotto_cartesiano_verbi_nomi_entita), len(prodotto_cartesiano_verbi_nomi_entita)//proporzione_possibili_combinazioni_da_escludere))
    vincoli.extend([tuple([verbo_aspettuale]+list(vincolo_verbo_entita)) for verbo_aspettuale in insieme_verbi_aspettuali for vincolo_verbo_entita in vincoli_verbi_entita])
    vincoli = list(set(vincoli))
    for coppia_tipica in coppie_tipiche.items():
        if tuple(reversed(coppia_tipica)) in vincoli: vincoli.remove(tuple(reversed(coppia_tipica)))  #  mi assicuro che la coppia tipica non sia in vincoli
    for nome_in_coppia_tipica in coppie_tipiche.keys():
        for verbo_aspettuale in insieme_verbi_aspettuali:
            for verbo in insieme_verbi:
                vincoli.append((verbo_aspettuale, verbo, nome_in_coppia_tipica))

    """
    Creazione dset per l'addestramento

    """
    print()
    print()
    print("----Creazione dsets per l'addestramento-----")
    file_addestramento_uniforme = open(path_file_addestramento_uniforme.format(numero_esempi_dset_addestramento, probabilita_verbo_tipico_dato_nome, probabilita_metonimia), 'w')
    file_addestramento_tipicalita = open(path_file_addestramento_tipicalita.format(numero_esempi_dset_addestramento, probabilita_verbo_tipico_dato_nome, probabilita_metonimia), 'w')
    for i in range(numero_esempi_dset_addestramento):
        """Generazione sequenza linguaggio di simboli"""
        simbolo_non_terminale_generato = np.random.choice(produzioni_linguaggio_simboli, p=probabilita_produzioni_linguaggio_simboli)
        sequenza_candidata_dset_uniforme = vincoli[0]
        sequenza_candidata_dset_tipicalita = vincoli[0]
        while (sequenza_candidata_dset_uniforme in vincoli) or (sequenza_candidata_dset_tipicalita in vincoli):
            if simbolo_non_terminale_generato == 'transitivo_semplice':
                sequenza_candidata_dset_uniforme = (np.random.choice(insieme_verbi), np.random.choice(insieme_nomi_entita))
                if sequenza_candidata_dset_uniforme[1] in coppie_tipiche.keys():
                    scegliere_verbo_tipico = np.random.choice([True, False], p=[probabilita_verbo_tipico_dato_nome, 1-probabilita_verbo_tipico_dato_nome])
                    if scegliere_verbo_tipico:
                        sequenza_candidata_dset_tipicalita = (coppie_tipiche[sequenza_candidata_dset_uniforme[1]], sequenza_candidata_dset_uniforme[1])
            elif simbolo_non_terminale_generato == 'aspettuale_semplice':
                sequenza_candidata_dset_uniforme = (np.random.choice(insieme_verbi_aspettuali), np.random.choice(insieme_nomi_evento))
                sequenza_candidata_dset_tipicalita = sequenza_candidata_dset_uniforme
            elif simbolo_non_terminale_generato == 'aspettuale_con_transitivo':
                sequenza_candidata_dset_uniforme = (np.random.choice(insieme_verbi_aspettuali), np.random.choice(insieme_verbi), np.random.choice(insieme_nomi_entita))  #  ('begin', 'read', 'book')
                sequenza_candidata_dset_tipicalita = sequenza_candidata_dset_uniforme
        lista_sequenza_dset_uniforme = list(reversed(sequenza_candidata_dset_uniforme))+['john']
        lista_sequenza_dset_tipicalita = list(reversed(sequenza_candidata_dset_tipicalita))+['john']
        stringa_sequenza_dset_uniforme, stringa_sequenza_dset_tipicalita = " ".join([simbolo.upper() for simbolo in lista_sequenza_dset_uniforme]), \
                                                                           " ".join([simbolo.upper() for simbolo in lista_sequenza_dset_tipicalita])
        """Traduzione in linguaggio naturale"""
        traduzione_sequenza_dset_uniforme, traduzione_sequenza_dset_tipicalita = traduci_sequenza_simboli(sequenza_candidata_dset_uniforme), \
                                                                                 traduci_sequenza_simboli(sequenza_candidata_dset_tipicalita)
        if simbolo_non_terminale_generato == 'aspettuale_con_transitivo':
            tipo_traduzione = np.random.choice(['implicito', 'esplicito'], p=[probabilita_metonimia, 1-probabilita_metonimia])  #  traduco con metonimia o no
            if tipo_traduzione == 'implicito':
                traduzione_sequenza_dset_uniforme, traduzione_sequenza_dset_tipicalita = traduzione_sequenza_dset_uniforme[0], traduzione_sequenza_dset_tipicalita[0]
            elif tipo_traduzione == 'esplicito':
                traduzione_sequenza_dset_uniforme, traduzione_sequenza_dset_tipicalita = traduzione_sequenza_dset_uniforme[1], traduzione_sequenza_dset_tipicalita[1]
        else:
            traduzione_sequenza_dset_uniforme, traduzione_sequenza_dset_tipicalita = traduzione_sequenza_dset_uniforme[0], traduzione_sequenza_dset_tipicalita[0]
        """Scrittura su file"""
        file_addestramento_uniforme.write(stringa_sequenza_dset_uniforme+'\t'+traduzione_sequenza_dset_uniforme+'\n')
        file_addestramento_tipicalita.write(stringa_sequenza_dset_tipicalita+'\t'+traduzione_sequenza_dset_tipicalita+'\n')
    file_addestramento_uniforme.close()
    file_addestramento_tipicalita.close()

    """
    Creazione datasets di valutazione
    """
    file_valutazione_funzione_semantica = open(path_file_valutazione_funzione_semantica, 'w')
    file_valutazione_parlante_letterale = open(path_file_valutazione_parlante_letterale, 'w')
    for vincolo in list(set(vincoli)):
        lista_sequenza_dset_valutazione = list(reversed(vincolo))+['john']
        stringa_sequenza_dset_valutazione = " ".join([simbolo.upper() for simbolo in lista_sequenza_dset_valutazione])
        if vincolo[0] in insieme_verbi:
            simbolo_nonterminale = 'transitivo_semplice'
        elif len(vincolo) == 2:
            simbolo_nonterminale = 'aspettuale_semplice'
        else:
            simbolo_nonterminale = 'aspettuale_con_transitivo'
        traduzione_sequenza_dset_valutazione = traduci_sequenza_simboli(vincolo)
        for traduzione in traduzione_sequenza_dset_valutazione:
            file_valutazione_funzione_semantica.write(stringa_sequenza_dset_valutazione+'\t'+traduzione+'\n')
        file_valutazione_parlante_letterale.write(stringa_sequenza_dset_valutazione+'\t'+"\t".join([traduzione for traduzione in traduzione_sequenza_dset_valutazione])+'\n')
    file_valutazione_funzione_semantica.close()
    file_valutazione_parlante_letterale.close()


def genera_sequenze_di_simboli_possibili(grammatica = grammatica1):
    """
    Data una grammatica, restituisce una lista con tutte le sequenze possibili di simboli (['LAW READ JOHN', 'SCRIPT PRAISE JOHN', 'PROTOCOL WRITE DELAY JOHN', ...])
    :param grammatica:
    :return: lista con tutte le sequenze possibili di simboli ([('start', 'throw', 'thesis'), ('start', 'throw', 'lyrics'), ...])
    """

    sequenze_di_simboli_possibili = []

    for produzione in grammatica1[6]:  #  (0, 2)
        sequenze_produzione = []
        lista_insiemi_simboli_terminali = [set(grammatica1[indice]) for indice in produzione]  #  ({read, write, ..}, {book, paper, ...})
        iteratore_prodotto_cartesiano = itertools.product(*lista_insiemi_simboli_terminali)
        for elemento in iteratore_prodotto_cartesiano:
            #  sequenze_produzione.append(' '.join([simbolo.upper() for simbolo in tuple(reversed(elemento))])+' JOHN')
            sequenze_produzione.append(elemento)
        sequenze_di_simboli_possibili.extend(sequenze_produzione)

    return sequenze_di_simboli_possibili

def genera_enunciati_possibili(sequenze_di_simboli_possibili):
    """
    Data una lista con le sequenze di simboli possibili nella forma ('start', 'throw', 'thesis') restituisce una lista con le possibili traduzioni date le sequenze di simboli
    nella lista di partenza. Se la lista di partenza contiene tutte le sequenze di simboli generabili con la grammatica, allora la lista creata contiene tutti i possibili enunciati della grammatica
    ATTENZIONE!! E' dato per implicito che la grammatica del linguaggio di parole "segue" quella del linguaggio dei simboli, nel senso che, p.es., parole che traducono verbi transitivi
    non possono svolgere il ruolo dei verbi aspettuali (p.es., sono vietati enunciati del tipo "John read throwing the paper"
    :param sequenze_di_simboli_possibili: lista con tuple del tipo ('start', 'throw', 'thesis')
    :return: lista con tuple di stringhe del tipo 'John began throwing the newspaper .'
    """
    sequenze_di_enunciati_possibili = []
    for sequenza_di_simboli in sequenze_di_simboli_possibili:
        sequenze_di_enunciati_possibili.extend(list(traduci_sequenza_simboli(sequenza_di_simboli)))
    return list(set(sequenze_di_enunciati_possibili))
