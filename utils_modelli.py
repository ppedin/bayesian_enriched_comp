#  Importazione librerie
import utils_dati
import utils_visualizzazione
import utils_valutazione


import random
import os
import pickle
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

SOS_token = 0
EOS_token = 1

path_modelli = '/home/paolo.pedinotti/fang2022/modelli'

#  Costruzione modello encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        #  Definizione attributi
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    #  Definizione metodi
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)  #  se la dimensione originaria è (1, 5000) la nuova dimensione è (1, 1, 5000)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


#  Costruzione modello encoder con moltiplicazione tra matrici, calcolo funzione sigmoide e layer lineare - per funzione semantica
class EncoderRNN_con_moltiplicazione_matrici(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN_con_moltiplicazione_matrici, self).__init__()
        #  Definizione attributi
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.sigmoide = nn.Sigmoid()

    #  Definizione metodi
    def forward(self, input, hidden, encoder1_output):
        embedded = self.embedding(input).view(1, 1, -1)  #  se la dimensione originaria è (1, 5000) la nuova dimensione è (1, 1, 5000)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = torch.matmul(output[0], torch.transpose(encoder1_output[0], 0, 1))
        output = self.sigmoide(output).reshape(1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)



#  Costruzione modello decoder semplice
class DecoderRNN(nn.Module):
    #  Definizione attributi
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #  Definizione metodi
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

#  Costruzione modello decoder con meccanismo di attenzione
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, percentuale_dropout, max_length):
        super(AttnDecoderRNN, self).__init__()
        #  Definizione attributi
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = percentuale_dropout
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    #  Definizione metodi
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  #  Se il tensore ha dimensione (1, 30), lo trasforma in tensore di dimensione (1, 1, 30)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

def addestramento_su_un_esempio_parlante_letterale(tensore_input, tensore_target, encoder, decoder,
                                                                ottimizzatore_encoder, ottimizzatore_decoder, funzione_loss,
                                                                lunghezza_massima):
    """
    :param tensore_input: size([lunghezza sequenza input, 1])
    :param tensore_target: size([lunghezza sequenza output, 1])
    :param encoder: istanza encoder
    :param decoder: istanza decoder
    :param ottimizzatore_encoder: ottimizzatore encoder
    :param ottimizzatore_decoder: ottimizzatore decoder
    :param funzione_loss: funzione loss
    :param lunghezza_massima: lunghezza massima sequenza di input nel dataset
    :return: loss per la sequenza di target (loss media per ciascun punto della sequenza target)
    """

    teacher_forcing_ratio = 0.5

    encoder_hidden = encoder.initHidden()
    ottimizzatore_encoder.zero_grad()
    ottimizzatore_decoder.zero_grad()

    encoder_outputs = torch.zeros(lunghezza_massima, encoder.hidden_size)

    loss = 0

    for punto_sequenza_input in range(tensore_input.size(0)):
        encoder_output, encoder_hidden = encoder.forward(tensore_input[punto_sequenza_input], encoder_hidden)
        encoder_outputs[punto_sequenza_input] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for punto_sequenza_target in range(tensore_target.size(0)):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            loss += funzione_loss(decoder_output, tensore_target[punto_sequenza_target])
            decoder_input = tensore_target[punto_sequenza_target]

    else:
        for punto_sequenza_target in range(tensore_target.size(0)):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  #  squeeze rimuove tutte le dimensioni con valore 1

            loss += funzione_loss(decoder_output, tensore_target[punto_sequenza_target])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    ottimizzatore_encoder.step()
    ottimizzatore_decoder.step()

    return loss.item() / tensore_target.size(0)

def addestramento_su_un_esempio_funzione_semantica(tensore_input_encoder1, tensore_input_encoder2, tensore_output,
                                                   encoder1, encoder2, ottimizzatore_encoder1, ottimizzatore_encoder2,
                                                   funzione_loss):
    """
    :param tensore_input_encoder1: size([lunghezza sequenza lingua 1, 1])
    :param tensore_input_encoder2: size([lunghezza sequenza lingua 2, 1])
    :param tensore_output: size([1]), dtype = torch.long
    :param encoder1: istanza encoder
    :param encoder2: istanza encoder
    :param ottimizzatore_encoder1: ottimizzatore encoder 1
    :param ottimizzatore_encoder2: ottimizzatore encoder 2
    :param funzione_loss: funzione loss
    :return: valore della funzione di loss per l'esempio di addestramento
    """
    encoder1_hidden = encoder1.initHidden()
    encoder2_hidden = encoder2.initHidden()
    ottimizzatore_encoder1.zero_grad()
    ottimizzatore_encoder2.zero_grad()

    for punto_sequenza1 in range(tensore_input_encoder1.size(0)):
        encoder1_output, encoder1_hidden = encoder1.forward(tensore_input_encoder1[punto_sequenza1], encoder1_hidden)  #  (1, 1, 256)
    for punto_sequenza2 in range(tensore_input_encoder2.size(0)):
        encoder2_output, encoder2_hidden = encoder2.forward(tensore_input_encoder2[punto_sequenza2], encoder2_hidden, encoder1_output)  #  (1, 2)

    loss = funzione_loss(encoder2_output, tensore_output)
    loss.backward()
    ottimizzatore_encoder1.step()
    ottimizzatore_encoder2.step()

    return loss.item()



def addestramento_parlante_letterale(n_iterazioni, path_dati, path_dati_valutazione, reverse,
                                     path_modello_encoder="modelli/parl_encoder_{}_{}_{}_{}.pth",
                                     path_modello_decoder="modelli/parl_decoder_{}_{}_{}_{}_{}.pth"):
    versione_dati_addestramento = os.path.basename(path_dati).split('_')[0]
    tipo_dati_addestramento = os.path.basename(path_dati).split('_')[2]

    #  Creazione esempi di addestramento
    print()
    print("Caricamento dati per l'addestramento ...")
    linguaggio_input, linguaggio_output = utils_dati.Linguaggio("simboli"), utils_dati.Linguaggio("inglese")
    coppie = utils_dati.carica_dati(path_dati, reverse)
    utils_dati.aggiorna_linguaggi(linguaggio_input, linguaggio_output, coppie)  #  aggiorna istanze di linguaggi con esempi addestramento
    #  aggiorna istanze di linguaggi con esempi valutazione
    coppie_valutazione = []
    file_valutazione = open(path_dati_valutazione, 'r')
    for riga in file_valutazione:
        numero_traduzioni = len(riga.split('\t')[1:])
        for indice in range(numero_traduzioni):
            coppie_valutazione.append((utils_dati.normalizza_stringa(riga.split('\t')[0]), utils_dati.normalizza_stringa(riga.split('\t')[indice + 1])))
    utils_dati.aggiorna_linguaggi(linguaggio_input, linguaggio_output, coppie_valutazione)
    file_valutazione.close()
    #  salva istanze linguaggi
    path_linguaggio_simboli = '/home/paolo.pedinotti/fang2022/modelli/linguaggiosimboli_parl_{}_{}.pkl'.format(versione_dati_addestramento, tipo_dati_addestramento)
    path_linguaggio_parole = '/home/paolo.pedinotti/fang2022/modelli/linguaggioparole_parl_{}_{}.pkl'.format(versione_dati_addestramento, tipo_dati_addestramento)
    with open(path_linguaggio_simboli, 'wb') as file_linguaggio_simboli:
        pickle.dump(linguaggio_input, file_linguaggio_simboli, pickle.HIGHEST_PROTOCOL)
    with open(path_linguaggio_parole, 'wb') as file_linguaggio_parole:
        pickle.dump(linguaggio_output, file_linguaggio_parole, pickle.HIGHEST_PROTOCOL)

    esempi_di_addestramento = []
    for numero_iterazione in range(n_iterazioni):  #  aggiorna istanze di linguaggi con esempi addestramento
        coppia_random = random.choice(coppie)
        esempi_di_addestramento.append((utils_dati.tensore_da_frase(linguaggio_input, coppia_random[0]),
                                        utils_dati.tensore_da_frase(linguaggio_output, coppia_random[1])))

    #  Creazione istanze modelli
    hidden_size = 256
    encoder = EncoderRNN(linguaggio_input.numero_parole, hidden_size)
    lunghezza_massima = utils_dati.trova_lunghezza_massima(esempi_di_addestramento)
    percentuale_dropout = 0.1
    decoder = AttnDecoderRNN(hidden_size, linguaggio_output.numero_parole, percentuale_dropout, lunghezza_massima)

    #  Definizione funzione loss, learning rate, ottimizzatori
    funzione_loss = nn.NLLLoss()
    learning_rate = 0.01
    ottimizzatore_encoder = optim.SGD(encoder.parameters(), lr=learning_rate)
    ottimizzatore_decoder = optim.SGD(decoder.parameters(), lr=learning_rate)

    #  Per ciascun esempio di addestramento, esecuzione addestramento sull'esempio
    informazioni_dopo_iterazione_numero = 500
    print()
    print("Inizio addestramento del parlante letterale: ")
    print()
    losses_per_grafico = []
    loss_totale_batch = 0
    for iter in range(1, n_iterazioni + 1):
        loss_totale_batch += addestramento_su_un_esempio_parlante_letterale(esempi_di_addestramento[iter - 1][0], esempi_di_addestramento[iter - 1][1],
                                    encoder, decoder, ottimizzatore_encoder, ottimizzatore_decoder, funzione_loss, lunghezza_massima)

        if iter % informazioni_dopo_iterazione_numero == 0:
            loss_media_batch = loss_totale_batch / informazioni_dopo_iterazione_numero
            print("Addestramento esempio numero {} di {}. Loss media di questo batch: {}".format(iter, n_iterazioni + 1, loss_media_batch))
            losses_per_grafico.append(loss_media_batch)
            loss_totale_batch = 0

    utils_visualizzazione.mostraGrafico(losses_per_grafico)

    #  Valutazione encoder-decoder
    predizioni_corrette = 0
    numero_esempi_valutazione = 0
    file_valutazione = open(path_dati_valutazione, 'r')
    for riga in file_valutazione:  #  simboli   traduzione1 traduzione2
        possibili_traduzioni = [traduzione.lower().strip() for traduzione in riga.split('\t')[1:]]
        traduzione_predetta = utils_valutazione.generazione_traduzione_parlante_letterale(utils_dati.tensore_da_frase(linguaggio_input, riga.split('\t')[0]),
                                                                        encoder, decoder, lunghezza_massima, linguaggio_output)
        #  print()
        #  print()
        #  print('Sequenza di simboli: {}  Possibili traduzioni: {}    Traduzione predetta: {}'.format(riga.split('\t')[0], possibili_traduzioni, traduzione_predetta))
        if traduzione_predetta in possibili_traduzioni:
            predizioni_corrette += 1
        numero_esempi_valutazione += 1

    print()
    print("L'accuratezza del modello encoder-decoder sul dataset utilizzato per la valutazione e disponibile a {} è {}".format(path_dati_valutazione,
                                                                                                                               round(100/numero_esempi_valutazione*predizioni_corrette, 3)))
    print()
    file_valutazione.close()


    #  Salvataggio modelli
    print("Salvataggio encoder parlante letterale ..")
    torch.save(encoder.state_dict(), path_modello_encoder.format(versione_dati_addestramento, tipo_dati_addestramento, n_iterazioni, hidden_size))
    print()
    print("Salvataggio decoder parlante letterale ..")
    torch.save(decoder.state_dict(), path_modello_decoder.format(versione_dati_addestramento, tipo_dati_addestramento, n_iterazioni, hidden_size, percentuale_dropout))


def addestramento_funzione_semantica(n_iterazioni, path_dati, path_dati_valutazione, reverse,
                                     path_encoder_simboli = "/home/paolo.pedinotti/fang2022/modelli/funz_encodersimboli_{}_{}_{}_{}.pth",
                                     path_encoder_parole = "/home/paolo.pedinotti/fang2022/modelli/funz_encoderparole_{}_{}_{}_{}.pth"):
    versione_dati_addestramento = os.path.basename(path_dati).split('_')[0]
    tipo_dati_addestramento = os.path.basename(path_dati).split('_')[2]

    print()
    print("Caricamento dati per l'addestramento ...")
    linguaggio1, linguaggio2 = utils_dati.Linguaggio("simboli"), utils_dati.Linguaggio("inglese")  #  assegnazione a 1 o 2 è arbitraria
    coppie = utils_dati.carica_dati(path_dati, reverse)
    utils_dati.aggiorna_linguaggi(linguaggio1, linguaggio2, coppie)  #  aggiorna istanze di linguaggi con esempi addestramento
    #  aggiorna istanze di linguaggi con esempi valutazione
    coppie_valutazione = []
    file_valutazione = open(path_dati_valutazione, 'r')
    for riga in file_valutazione:
        coppie_valutazione.append((utils_dati.normalizza_stringa(riga.split('\t')[0]), utils_dati.normalizza_stringa(riga.split('\t')[1])))
    utils_dati.aggiorna_linguaggi(linguaggio1, linguaggio2, coppie_valutazione)
    file_valutazione.close()

    #  salva istanze linguaggi
    path_linguaggio_simboli = '/home/paolo.pedinotti/fang2022/modelli/linguaggiosimboli_funz_{}_{}.pkl'.format(versione_dati_addestramento, tipo_dati_addestramento)
    path_linguaggio_parole = '/home/paolo.pedinotti/fang2022/modelli/linguaggioparole_funz_{}_{}.pkl'.format(versione_dati_addestramento, tipo_dati_addestramento)
    with open(path_linguaggio_simboli, 'wb') as file_linguaggio_simboli:
        pickle.dump(linguaggio1, file_linguaggio_simboli, pickle.HIGHEST_PROTOCOL)
    with open(path_linguaggio_parole, 'wb') as file_linguaggio_parole:
        pickle.dump(linguaggio2, file_linguaggio_parole, pickle.HIGHEST_PROTOCOL)

    triple = utils_dati.genera_dati_funzione_semantica(coppie)
    esempi_di_addestramento = []
    for numero_iterazione in range(n_iterazioni):
        tripla_random = random.choice(triple)
        esempi_di_addestramento.append((utils_dati.tensore_da_frase(linguaggio1, tripla_random[0]),
                                        utils_dati.tensore_da_frase(linguaggio2, tripla_random[1]),
                                        torch.tensor([tripla_random[2]], dtype=torch.float32)))

    #  Creazione istanze modelli
    hidden_size = 256
    encoder1 = EncoderRNN(linguaggio1.numero_parole, hidden_size)
    encoder2 = EncoderRNN_con_moltiplicazione_matrici(linguaggio2.numero_parole, hidden_size)

    #  Definizione funzione loss, learning rate, ottimizzatori
    funzione_loss = nn.BCELoss()  #  Per adesso uso questa, poi scrivo codice per confrontare performance con diverse loss functions su validation set
    learning_rate = 0.01
    ottimizzatore_encoder1 = optim.SGD(encoder1.parameters(), lr=learning_rate)
    ottimizzatore_encoder2 = optim.SGD(encoder2.parameters(), lr=learning_rate)

    #  Per ciascun esempio di addestramento, esecuzione addestramento sull'esempio
    informazioni_dopo_iterazione_numero = 500
    print()
    print("Inizio addestramento della funzione semantica: ")
    print()
    losses_per_grafico = []
    loss_totale_batch = 0
    for iter in range(1, n_iterazioni + 1):
        loss_totale_batch += addestramento_su_un_esempio_funzione_semantica(esempi_di_addestramento[iter - 1][0],
                                                                            esempi_di_addestramento[iter - 1][1],
                                                                            esempi_di_addestramento[iter - 1][2],
                                                                            encoder1, encoder2, ottimizzatore_encoder1,
                                                                            ottimizzatore_encoder2, funzione_loss)
        if iter % informazioni_dopo_iterazione_numero == 0:
            loss_media_batch = loss_totale_batch / informazioni_dopo_iterazione_numero
            print("Addestramento esempio numero {} di {}. Loss media di questo batch: {}".format(iter, n_iterazioni + 1, loss_media_batch))
            losses_per_grafico.append(loss_media_batch)
            loss_totale_batch = 0

    utils_visualizzazione.mostraGrafico(losses_per_grafico)

    #  Valutazione funzione semantica
    predizioni_corrette = 0
    numero_esempi_valutazione = 0
    coppie_valutazione = utils_dati.carica_dati(path_dati_valutazione, reverse)
    triple_valutazione = utils_dati.genera_dati_funzione_semantica(coppie_valutazione)
    print('Verifica manuale valutazione funzione semantica')

    for tripla in triple_valutazione:
        valore_sigmoide = utils_valutazione.generazione_predizione_funzione_semantica(utils_dati.tensore_da_frase(linguaggio1, tripla[0]),
                                                               utils_dati.tensore_da_frase(linguaggio2, tripla[1]),
                                                               encoder1, encoder2)  #  deve restituire 0 o 1
        print()
        print()
        print("Sequenza di simboli: {}  Sequenza di parole: {}  Etichetta corretta: {}  Probabilità predetta esempio positivo: {}".format(tripla[0], tripla[1], tripla[2], valore_sigmoide))
        if tripla[2] in utils_valutazione.predizione_da_sigmoide(valore_sigmoide):
            predizioni_corrette += 1
        numero_esempi_valutazione += 1
    print()
    print("L'accuratezza della funzione semantica sul dataset utilizzato per la valutazione e disponibile a {} è {}".format(path_dati_valutazione, round(100/numero_esempi_valutazione*predizioni_corrette, 3)))



    #  Salvataggio modelli
    print("Salvataggio encoder 1 funzione semantica (codifica sequenza di simboli) ...")
    torch.save(encoder1.state_dict(), path_encoder_simboli.format(versione_dati_addestramento, tipo_dati_addestramento, n_iterazioni, hidden_size))
    print("Salvataggio encoder 2 funzione semantica (codifica enunciato) ...")
    torch.save(encoder2.state_dict(), path_encoder_parole.format(versione_dati_addestramento, tipo_dati_addestramento, n_iterazioni, hidden_size))


def costo_con_penalita_lineare(enunciato, penalita=.01):
    return len(enunciato.split(' ')) * penalita

def ottieni_valore_da_modello_RSA(sequenza_simboli_di_interesse,
                                  path_linguaggio_simboli = os.path.join(path_modelli, 'linguaggiosimboli_funz_v1_tip.pkl'),
                                  path_linguaggio_parole = os.path.join(path_modelli, 'linguaggioparole_funz_v1_tip.pkl'),
                                  path_encoder_simboli_funzione_semantica = os.path.join(path_modelli, 'funz_encodersimboli_v1_tip_500_256.pth'),
                                  path_encoder_parole_funzione_semantica = os.path.join(path_modelli, 'funz_encoderparole_v1_tip_500_256.pth'),
                                  path_encoder_parlante_letterale = os.path.join(path_modelli, 'parl_encoder_v1_tip_500_256.pth'),
                                  path_decoder_parlante_letterale = os.path.join(path_modelli, 'parl_decoder_v1_tip_500_256_0.1.pth'),
                                  calcolo_ascoltatore_interno='approssimazione',
                                  enunciati='tutti',
                                  funzione_costo = costo_con_penalita_lineare):
    """
    Data una sequenza di simboli, restituisce un valore per ciascun possibile enunciato (utilità dell'enunciato data la sequenza).
    L'output potrà essere nel formato {enunciato1: utilità, enunciato2: utilità, ...}
    :param sequenza_di_simboli: sequenza di simboli nel formato dei dati (stringa)
    :param path_linguaggio_simboli: path all'istanza di linguaggio
    :param path_linguaggio_parole: path all'istanza di linguaggio
    :param path_encoder_simboli_funzione_semantica: path al modello salvato
    :param path_encoder_enunciati_funzione_semantica: path al modello salvato
    :param encoder_parlante_letterale:
    :param decoder_parlante_letterale:
    :param calcolo_ascoltatore_interno: deve essere un valore in ['approssimazione', 'preciso']. Se è
    'approssimazione', viene calcolato un valore proporzionale alla probabilità dell'ascoltatore interno. Questo metodo è
    più economico perché, per ciascun enunciato, calcola solo il valore della funzione semantica per l'enunciato e la sequenza
    di simboli di interesse (senza calcolarlo per tutte le sequenze di simboli). Se è 'preciso', viene calcolata la probabilità
    (calcolando, per ciascun enunciato, il valore della funzione semantica per tutte le sequenze di simboli)
    :param enunciati: deve essere un valore tra 'tutti' e 'selezione'. Se è 'tutti', viene calcolata l'utilità di ciascun possibile
    enunciato data la sequenza di simboli di partenza. Se è 'selezione', viene definito un insieme di enunciati campionando
    dalla distribuzione di probabilità del parlante letterale, e viene calcolata l'utilità di ciascun enunciato dell'insieme definito.
    :param funzione_costo: funzione che definisce il modo in cui il costo di un enunciato viene calcolato
    :return: Data una sequenza di simboli, restituisce un valore per ciascun possibile enunciato (utilità dell'enunciato data la sequenza).
    """

    """Genera possibili sequenze di simboli e di enunciati"""

    sequenze_di_simboli_possibili = utils_dati.genera_sequenze_di_simboli_possibili()  #  [('start', 'throw', 'thesis'), ('start', 'throw', 'lyrics'), ...]
    enunciati_possibili = utils_dati.genera_enunciati_possibili(sequenze_di_simboli_possibili)  #  ['John discussed the book .', 'John survived copying the law .', ...]

    """Carica encoder simboli funzione semantica, encoder enunciati funzione semantica"""
    with open(path_linguaggio_simboli, 'rb') as fp:
        linguaggio_simboli = pickle.load(fp)
    with open(path_linguaggio_parole, 'rb') as fp:
        linguaggio_parole = pickle.load(fp)
    hidden_size_encoder_simboli = int(os.path.splitext(os.path.basename(path_encoder_simboli_funzione_semantica))[0].split('_')[5])
    hidden_size_encoder_parole = int(os.path.splitext(os.path.basename(path_encoder_parole_funzione_semantica))[0].split('_')[5])
    encoder_simboli_funzione_semantica = EncoderRNN(linguaggio_simboli.numero_parole, hidden_size_encoder_simboli)
    encoder_parole_funzione_semantica = EncoderRNN_con_moltiplicazione_matrici(linguaggio_parole.numero_parole, hidden_size_encoder_parole)
    encoder_simboli_funzione_semantica.load_state_dict(torch.load(path_encoder_simboli_funzione_semantica))
    encoder_parole_funzione_semantica.load_state_dict(torch.load(path_encoder_parole_funzione_semantica))

    """Definisce l'insieme degli enunciati possibili"""
    if enunciati == 'tutti':
        insieme_enunciati = enunciati_possibili
    elif enunciati == 'selezione':
        insieme_enunciati = []
        #  SCRIVERE QUI CODICE PER CAMPIONARE ENUNCIATI DAL PARLANTE LETTERALE
    else:
        print("Il valore del parametro 'enunciati' deve essere uno tra 'tutti' e 'selezione'")
        exit()

    utilita_per_ciascun_enunciato = {}

    """Calcola l'utilità per ciascun enunciato data la sequenza di simboli di interesse"""
    contatore_enunciati = 0
    for enunciato in insieme_enunciati:
        contatore_enunciati += 1
        if contatore_enunciati % 500 == 0:
            print('Calcolo ascoltatore interno per sequenza di interesse e enunciato {} di {}'.format(contatore_enunciati, len(insieme_enunciati)))
        if calcolo_ascoltatore_interno == 'approssimazione':
            valore_ascoltatore_interno_enunciato_seq_di_interesse = utils_valutazione.generazione_predizione_funzione_semantica(utils_dati.tensore_da_frase(linguaggio_simboli, sequenza_simboli_di_interesse),
                                                                                                                     utils_dati.tensore_da_frase(linguaggio_parole, utils_dati.normalizza_stringa(enunciato)),
                                                                                                                     encoder_simboli_funzione_semantica,
                                                                                                                     encoder_parole_funzione_semantica)
        elif calcolo_ascoltatore_interno == 'preciso':  #  ALTAMENTE SCONSIGLIATO! CON GRAMMATICA1, DEVE ESEGUIRE ca. 15 MILIONI DI ITERAZIONI! (SOLO CON DSETS PIU' PICCOLI) O CON UNA SELEZIONE DEGLI ENUNCIATI
            sigmoide_per_ciascuna_sequenza_simboli = []
            for sequenza_simboli in sequenze_di_simboli_possibili:
                stringa_sequenza_simboli = ' '.join([simbolo.upper() for simbolo in tuple(reversed(sequenza_simboli))]) + ' JOHN'
                sigmoide_per_ciascuna_sequenza_simboli.append(utils_valutazione.generazione_predizione_funzione_semantica(utils_dati.tensore_da_frase(linguaggio_simboli, stringa_sequenza_simboli),
                                                                                                            utils_dati.tensore_da_frase(linguaggio_parole, utils_dati.normalizza_stringa(enunciato)),
                                                                                                            encoder_simboli_funzione_semantica,
                                                                                                            encoder_parole_funzione_semantica))
            valore_sigmoide_enunciato_seq_di_interesse = utils_valutazione.generazione_predizione_funzione_semantica(utils_dati.tensore_da_frase(linguaggio_simboli, sequenza_simboli_di_interesse),
                                                                                                                     utils_dati.tensore_da_frase(linguaggio_parole, utils_dati.normalizza_stringa(enunciato)),
                                                                                                                     encoder_simboli_funzione_semantica,
                                                                                                                     encoder_parole_funzione_semantica)
            denominatore_softmax = sum([exp(sigmoide) for sigmoide in sigmoide_per_ciascuna_sequenza_simboli])
            valore_ascoltatore_interno_enunciato_seq_di_interesse = exp(valore_sigmoide_enunciato_seq_di_interesse)/denominatore_softmax
        else:
            print("Il valore del parametro 'calcolo_ascoltatore_interno' deve essere uno tra 'approssimazione' e 'preciso'")
            exit()
        utilita_per_ciascun_enunciato[enunciato] = np.log(valore_ascoltatore_interno_enunciato_seq_di_interesse) - funzione_costo(enunciato)
    return utilita_per_ciascun_enunciato





