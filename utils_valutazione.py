import torch

SOS_token = 0
EOS_token = 1


def generazione_traduzione_parlante_letterale(tensore_input, encoder, decoder, lunghezza_massima, linguaggio_output):
    """
    :param tensore_input:  size([lunghezza sequenza input, 1])
    :param encoder: istanza (ottimizzata o no) encoder
    :param decoder: istanza (ottimizzata o no) decoder
    :param lunghezza_massima: lunghezza massima sequenza di input nel dataset
    :param linguaggio_output: istanza di Linguaggio
    :return: la traduzione della stringa nel linguaggio output a opera dell'encoder e del decoder
    """

    sequenza_parole_output = []
    with torch.no_grad():
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(lunghezza_massima, encoder.hidden_size)

        for punto_sequenza_input in range(tensore_input.size(0)):
            encoder_output, encoder_hidden = encoder.forward(tensore_input[punto_sequenza_input], encoder_hidden)
            encoder_outputs[punto_sequenza_input] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]])
        decoder_hidden = encoder_hidden
        for punto_sequenza_target in range(lunghezza_massima):
            decoder_output, decoder_hidden, decoder_attention = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  #  topv è valore massimo, topi è indice del valore massimo. Entrambi sono tensori di dimensione ([1,1])
            if topi.item() == EOS_token:
                sequenza_parole_output.append('<EOS>')
                break
            else:
                sequenza_parole_output.append(linguaggio_output.indice2parola[topi.item()])

            decoder_input = topi.squeeze().detach()  #  numero che è l'indice con valore maggiore tra le predizioni
        return ' '.join(sequenza_parole_output)


def generazione_predizione_funzione_semantica(tensore_input_encoder1, tensore_input_encoder2, encoder1, encoder2):
    """
    Dato un esempio dai dati per la valutazione della funzione semantica (due frasi) compie una predizione sulla base dei modelli
    :param tensore_input_encoder1: sequenza di simboli in forma di tensore
    :param tensore_input_encoder2: sequenza di parole in forma di tensore
    :param encoder1: encoder di una sequenza di simboli
    :param encoder2: encoder di una sequenza di parole
    :return: sigmoide (probabilità che ci sia una corrispondenza semantica tra le due sequenze)
    """

    with torch.no_grad():
        encoder1_hidden = encoder1.initHidden()
        encoder2_hidden = encoder2.initHidden()

        for punto_sequenza1 in range(tensore_input_encoder1.size(0)):
            encoder1_output, encoder1_hidden = encoder1.forward(tensore_input_encoder1[punto_sequenza1], encoder1_hidden)
        for punto_sequenza2 in range(tensore_input_encoder2.size(0)):
            encoder2_output, encoder2_hidden = encoder2.forward(tensore_input_encoder2[punto_sequenza2], encoder2_hidden, encoder1_output)
        return round(encoder2_output.item(), 3)



def predizione_da_sigmoide(valore_sigmoide):
    """
    Dato un output della funzione sigmoide (quindi un valore da 0 a 1) restituisce una tupla con le predizioni corrispondenti.
    Se il valore è più vicino a 0 di quanto lo sia a 1 restituisce 0, 1 viceversa. Se il valore è .5, restituisce una tupla vuota.
    In questo modo, la corrispondenza tra le predizioni e l'etichetta originaria può essere verificata con l'operatore in
    :param valore_sigmoide: valore della funzione sigmoide (quindi un valore tra 0 e 1)
    :return: tupla con le predizioni corrispondenti (es. (0), (1), ())
    """
    if valore_sigmoide > 1 - valore_sigmoide:
        return (1,)
    elif valore_sigmoide < 1 - valore_sigmoide:
        return (0,)
    else:  #  sigmoide = 1 - sigmoide
        return ()