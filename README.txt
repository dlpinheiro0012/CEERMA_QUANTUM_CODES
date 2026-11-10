CÓDIGOS DE PESQUISA (CEERMA-UFPE), por Arthur Dantas
=====================================================

Este pacote é para leitura: serve para você conhecer o que já foi feito.
Os dados não estão inclusos. Os caminhos dentro dos scripts
(ex.: C:\Users\Arthur\...) apontam para o meu computador, então os
códigos não rodam sem ajuste.

Quase todos os scripts estão em Python, divididos em células "#%%"
(abrem bem no Spyder ou no VS Code). Bibliotecas principais:
numpy, pandas, scikit-learn, tensorflow/keras, torch, pennylane
(simulação de circuitos quânticos), librosa (áudio).

SUGESTÃO DE ORDEM DE LEITURA
----------------------------
1) 01_rolamentos_bancada/Bancada_ML.py       -> o caso mais simples (MLP clássico)
2) 01_rolamentos_bancada/Bancada_QML.py      -> o mesmo problema com QML
3) 02_rolamentos_CWRU_DC/                    -> encodings quânticos em mais detalhe
4) Depois disso, o restante conforme o seu interesse.

O QUE TEM EM CADA PASTA
-----------------------
01_rolamentos_bancada/
   Classificação de falhas em rolamentos com dados da bancada de vibração
   do laboratório, em duas condições: AB (alinhada e balanceada) e
   DD (desalinhada e desbalanceada).
   - Bancada_ML.py / Bancada_ML_10x.py : MLP clássico (10x = 10 rodadas)
   - Bancada_QML.py : QML com angle e amplitude encoding + teste de Kruskal-Wallis
   - qcnn/ : versão com QCNN (rede neural convolucional quântica)
       prep_dados.py -> run.py -> metricas.py (ordem de execução)
       qsim.py = simulador próprio em torch; test_equiv.py confere que ele
       dá o mesmo resultado que o PennyLane
   - resultados/ : CSVs e planilhas com as métricas das 10 rodadas
   - figuras/    : boxplots e matrizes de confusão
   - Poster_Quanta_Falhas_em_Rolamentos.pptx : pôster com os resultados

02_rolamentos_CWRU_DC/
   Mesmo tema, com o dataset público CWRU (Case Western Reserve University).
   Compara formas de codificar o sinal em qubits: amplitude, angle e
   Divide-and-Conquer (D&C, Araujo et al. 2021), sobre o sinal bruto (raw)
   e sobre a FFT.
   - D&C - Row data_vf.py : versão final (o cabeçalho explica o desenho
     do experimento)
   - Raw & FFT - *.py : variações (só amplitude/angle; só D&C)
   - Gráficos - Raw & FFT.py : gráficos de acurácia vs dimensão
   - versoes_antigas/ : histórico, só para consulta
   - testes_estatisticos/ : scripts de Kruskal-Wallis (CWRU e JNU)

03_aerogerador/
   Detecção de falhas em aerogerador: ML clássico (Aero_ML) vs QML com
   diferentes encodings (Aero_QML, Aero_Ampl, Aero_ZZ).

04_MIMII_audio/
   Detecção de anomalias em áudio de máquinas industriais (dataset MIMII,
   bomba a 6 dB) com espectrograma Mel, SVM e redes neurais.

06_EQL_series_temporais/
   Previsão de séries temporais de potência ("power") com ELM, RNN, LSTM e
   GRU, incluindo bootstrap, Moving Difference (MD) e Wavelet.
   figuras/ guarda os gráficos gerados.

07_estudos_codificacao_quantica/
   Estudos exploratórios usando o MNIST como exemplo:
   - EnQode/ : codificação EnQode + QSVM, QNN e QAE
   - block_encoding/ : testes de block encoding (base para QSVT)

DÚVIDAS: fale comigo (Arthur).
