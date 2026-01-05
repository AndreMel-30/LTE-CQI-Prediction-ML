% PROJECT: LTE CQI Prediction using Machine Learning
% DESCRIPTION: Simulates an LTE downlink scenario to predict Channel Quality Indicator (CQI)
% based on user position and SNR using Linear Regression and Decision Trees.

clear;
clc;
close all; % Chiude eventuali figure aperte

% --- PARAMETRI DI SIMULAZIONE ---
raggio_cella = 2000;         % Raggio della cella (metri)
BS_pos = [0, 0];             % Posizione Base Station
Pt_dBm = 46;                 % Potenza trasmessa BS (dBm)
shadowing_std = 6;           % Deviazione standard Shadowing (dB) - Rende la sim più realistica
N_utenti = 1000;             % Numero utenti per round (fisso per coerenza, o usa input)
N_RB_max = 100;              % Numero Resource Blocks (20 MHz)
BW_RB = 180e3;               % Banda per RB (Hz)
N0_dBmHz = -174;             % Densità spettrale di rumore
BW_tot = N_RB_max * BW_RB;   % Banda totale
N_dBm = N0_dBmHz + 10*log10(BW_tot); % Rumore totale

% Soglie SNR per mapping CQI (Tabella standard LTE)
CQI_soglie = [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7];

% Input utente per numero di round
rounds = input("Inserisci numero di round di simulazione (es. 50): ");
if isempty(rounds), rounds = 10; end % Valore default se vuoto

% Vettori per salvare le performance (RMSE)
rmse_linear = zeros(rounds, 1);
rmse_tree = zeros(rounds, 1);

fprintf('Avvio simulazione (%d round)...\n', rounds);

% --- CICLO DI SIMULAZIONE ---
for k = 1:rounds
    
    % 1. Generazione posizioni utenti (distribuzione uniforme nel cerchio)
    theta = 2*pi*rand(N_utenti,1);
    r = raggio_cella * sqrt(rand(N_utenti,1)); 
    x = r .* cos(theta);
    y = r .* sin(theta);
    d = sqrt(x.^2 + y.^2); % Distanza dalla BS

    % 2. Calcolo Path Loss + Shadowing (Log-Normal)
    % Aggiungiamo randn * shadowing_std per rompere la collinearità perfetta
    PL_dB = 128.1 + 37.6*log10(d/1000) + (randn(N_utenti, 1) * shadowing_std);
    
    % 3. Link Budget
    Pr_dBm = Pt_dBm - PL_dB;      % Potenza ricevuta
    SNR_dB = Pr_dBm - N_dBm;      % SNR

    % 4. Mapping CQI
    CQI = zeros(N_utenti,1);
    for i = 1:N_utenti
        % Trova l'indice della soglia più alta superata
        idx = find(SNR_dB(i) > CQI_soglie, 1, 'last');
        if isempty(idx)
            CQI(i) = 1; % Valore minimo
        else
            CQI(i) = idx;
        end
    end

    % 5. Preparazione Dataset (X = Features, y = Labels)
    % Usiamo SNR e Distanza come features. Grazie allo shadowing, 
    % non sono più perfettamente correlate linearmente.
    X = [SNR_dB, d]; 
    y = CQI;

    % Divisione Train (70%) / Test (30%)
    c = cvpartition(N_utenti, "holdout", 0.3);
    X_train = X(training(c), :);
    y_train = y(training(c), :);
    X_test  = X(test(c), :);
    y_test  = y(test(c), :);

    % 6. Training dei Modelli
    % Modello 1: Regressione Lineare
    mdl_lin = fitlm(X_train, y_train);
    
    % Modello 2: Albero Decisionale (Regression Tree)
    mdl_tree = fitrtree(X_train, y_train);

    % 7. Predizione e Valutazione
    pred_lin = predict(mdl_lin, X_test);
    pred_tree = predict(mdl_tree, X_test);

    % Calcolo RMSE (Root Mean Square Error)
    rmse_linear(k) = sqrt(mean((y_test - pred_lin).^2));
    rmse_tree(k)   = sqrt(mean((y_test - pred_tree).^2));
    
    % Barra di avanzamento testuale
    if mod(k, 10) == 0
        fprintf('Completato round %d/%d\n', k, rounds);
    end
end

% --- VISUALIZZAZIONE RISULTATI ---
figure('Name', 'Model Comparison', 'Color', 'w');

% Plot andamento RMSE
subplot(2,1,1);
plot(1:rounds, rmse_linear, '-o', 'LineWidth', 1.5, 'DisplayName', 'Linear Regression');
hold on;
plot(1:rounds, rmse_tree, '-x', 'LineWidth', 1.5, 'DisplayName', 'Decision Tree');
title('Andamento RMSE sui vari round');
xlabel('Round di simulazione');
ylabel('RMSE (Errore CQI)');
legend show;
grid on;

% Plot confronto medio
subplot(2,1,2);
avg_rmse = [mean(rmse_linear), mean(rmse_tree)];
b = bar(categorical({'Linear Regression', 'Decision Tree'}), avg_rmse);
b.FaceColor = 'flat';
b.CData(1,:) = [0 0.4470 0.7410]; % Blu
b.CData(2,:) = [0.8500 0.3250 0.0980]; % Arancione
title(sprintf('RMSE Medio (Linear: %.3f vs Tree: %.3f)', avg_rmse(1), avg_rmse(2)));
ylabel('RMSE Medio');
grid on;

fprintf('\nSimulazione completata.\n');
