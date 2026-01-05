%{Studenti: Melissari Andrea - 206056, Postorino Gabriele - 206050
% , Quattrone Davide Maria - 204394 %}

clear;
clc;

% l'esecuzione di fitml dà un waring perché ci sono colonne della matrice
% linearmente indipenden, poiché non abbiamo effettuato nessuna fase di
% data preprocessing
warning('off', "all");

% parametri simulazione costanti

raggio_cella = 2000;         % metri
BS_pos = [0, 0];            % BS al centro
Pt_dBm = 46;     % Potenza trasmessa dalla BS (tipico LTE: 40-46 dBm)
N_utenti = input('numero di utenti: ');   % Numero di utenti da generare
N_RB_max = 100;          % Numero massimo di RB (es. 20 MHz LTE)
BW_RB = 180e3;    % Banda di un RB (Hz) Un RB LTE è composto da 12 subcarrier x 15 kHz = 180 kHz
N0_dBmHz = -174;         % Rumore termico in dBm/Hz
CQI_soglie = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7];
data_rate_per_RB = [...
    25.59, 39.38, 63.34, 101.07, 147.34, ...
    197.53, 248.07, 321.57, 404.26, 458.72, ...
    558.15, 655.59, 759.93, 859.35, 933.19]';

round = input("Quante volte ripetere la simulazione? ");
rmse_mdl1 = zeros(round,1); % vettore per salvare i risultati del primo modello
rmse_mdl2 = zeros(round,1); % vettore per salvare i risultati del secondo modello
for k = 1:round  % for per le diverse simulazioni

    fprintf("round %d\n", k);

    % Genero nuova dispozione spaziale degli utenti
    % Calcolo SNR, d, Pr_dBm, CQI
    % Costruisco X = [SNR, d, Pr_dBm] e y = CQI

    % ad ogni round li rigenero
    theta = 2*pi*rand(N_utenti,1);        % angoli casuali
    r = raggio_cella * sqrt(rand(N_utenti,1)); % distanza casuale nel cerchio
    x = r .* cos(theta);
    y = r .* sin(theta);
    d = sqrt(x.^2 + y.^2);     % distanza dalla BS

    PL_dB = 128.1 + 37.6*log10(d/1000); %  path loss
    Pr_dBm = Pt_dBm - PL_dB;    % Potenza ricevuta dall'utente
    BW_tot = N_RB_max * BW_RB;
    N_dBm = N0_dBmHz + 10*log10(BW_tot);
    SNR_dB = Pr_dBm - N_dBm;

    % mappatura CQI

    CQI = zeros(N_utenti,1);
    for i = 1:N_utenti
        CQI(i) = find(SNR_dB(i) > CQI_soglie, 1, 'last');
        if isempty(CQI(i))
            CQI(i) = 1;
        end
    end
    data_rate_utente = N_RB_max * data_rate_per_RB(CQI); % non ci serve in realtà

    % Costruisco X = [SNR, d, Pr_dBm] e y = CQI, ovvero il dataset e le
    % labels

    X = [SNR_dB, d, Pr_dBm];   % matrice delle features che dobbiamo dividere in train e test
    y = CQI;  % le etichette che useremo per il training dei modelli

    % Divisione train/test usando cv partition

    c = cvpartition(N_utenti,"holdout",0.3); % divisione train/test - 70/30%
    indici_train = training(c); % indici matrice X che corrispondo al train
    indici_test = test(c); % indici matrice X che corrispondono al test

    % prendo le righe di X che corrispondono a quegli indici
    X_train = X(indici_train,:); % training set
    X_test = X(indici_test,:);  % test set

    % facciamo la stessa cosa con il vettore delle etichette
    CQI_train = y(indici_train);
    CQI_test = y(indici_test);

    % Addestramento modelli
    mdl1 = fitlm(X_train, CQI_train);
    mdl2 = fitrtree(X_train, CQI_train);

    % predizione CQI e calcolo del rmse
    CQI_mdl1 = predict(mdl1,X_test);
    CQI_mdl2 = predict(mdl2,X_test);
    % indipendentemente dalla scelta dell'utente io devo calcolari sempre
    % perché devo conoscere i valori di entrambi all'iterazione precedente
    rmse_mdl1(k) = sqrt(mean((CQI_test - CQI_mdl1).^2));
    rmse_mdl2(k) = sqrt(mean((CQI_test - CQI_mdl2).^2));

    % controllo dell'input
    scegli_modello = input("Scegli il modello (1=fitlm, 2=fitrtree, 3=modello migliore all'iterazione precedente): ");
    while (scegli_modello~=1&&scegli_modello~=2&&scegli_modello~=3)
        scegli_modello = input("Input non valido! Scegli il modello (1=fitlm, 2=fitrtree, 3=modello migliore all'iterazione precedente): ");
    end

    switch scegli_modello
        case 1
            fprintf("RMSE (fitlm): %.4e\n", rmse_mdl1(k));
        case 2
            fprintf("RMSE (fitrtree): %.4e\n", rmse_mdl2(k));
        case 3
            if(k == 1)
                % nel caso in cui l'utente dovesse inserire comunque il
                % caso 3 la prima volta
                fprintf("alla prima iterazione non è possibile scegliere il caso 3: " + ...
                    "verrà scelto un modello casualmente tra i due: \n")
                r = rand; % genero un numero casuale tra 0 e 1
                if r < 0.5
                    fprintf("RMSE (fitlm): %.4e\n", rmse_mdl1(k));
                else
                    fprintf("RMSE (fitrtree): %.4e\n", rmse_mdl2(k));
                end
            elseif rmse_mdl1(k-1) < rmse_mdl2(k-1)
                fprintf("RMSE (fitlm): %.4e\n", rmse_mdl1(k));
            else
                fprintf("RMSE (fitrtree): %.4e\n", rmse_mdl2(k));
            end
    end

    bar(k,rmse_mdl1);
    hold on 
    bar(k,rmse_mdl2);
end


