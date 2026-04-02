%% DEMO_CUSTOM_REVERB  Generate and audition a reverb from your own parameters.
%
%   Edit the six controls below, run the script, and listen.
clear; close all; clc;
addpath('core');

%% === SET YOUR PARAMETERS HERE ============================================
T60        = 1.5;     % Reverberation time [s]         (0.05+)
roomSize   = 0.6;     % Small=0, Large=1               (0–1)
warmth     = 1.1;     % Bass ratio BR; >1 = warmer     (0.5–2.0)
brightness = 0.9;     % Treble ratio TR; >1 = brighter  (0.3–2.0)
diffusion  = 0.8;     % 0 = metallic grid, 1 = natural  (0–1)
earlyLate  = 0.5;     % 0 = diffuse tail, 1 = dry/clear (0–1)
%% =========================================================================

fs = 44100;

% Build or load model
modelFile   = fullfile('data', 'paramModel.mat');
featureFile = fullfile('data', 'featureTable.mat');
if exist(modelFile, 'file')
    load(modelFile, 'paramModel');
else
    fprintf('Building model from %s ...\n', featureFile);
    paramModel = build_parametric_model(featureFile, modelFile);
end

% Generate
up = struct('T60', T60, 'roomSize', roomSize, 'warmth', warmth, ...
            'brightness', brightness, 'diffusion', diffusion, ...
            'earlyLate', earlyLate, 'fs', fs);

Tsim = max(2, 2 * T60);
[a, b, c, f, hFIR] = parametric_reverb_generate(paramModel, up);
ir = IR_Synt(a, b, f, c, fs, Tsim, 1, 0);
ir = conv(ir, hFIR);
ir = ir(1:round(fs * Tsim));
ir = ir / (max(abs(ir)) + eps);

outDir = './generated_IRs';
if ~exist(outDir, 'dir'), mkdir(outDir); end
audiowrite(fullfile(outDir, 'custom_reverb.wav'), ir, fs);
fprintf('\nSaved: generated_IRs/custom_reverb.wav  (%.1f s, %d modes)\n', Tsim, numel(f));

% Plot
t = (0:numel(ir)-1) / fs;
figure('Position', [100 100 900 600]);

subplot(2,1,1);
plot(t, ir, 'Color', [0.2 0.4 0.8], 'LineWidth', 0.4);
xlim([0 Tsim]); grid on;
xlabel('Time (s)', 'FontSize', 12); ylabel('Amplitude', 'FontSize', 12);
title(sprintf('T60=%.2fs  size=%.2f  BR=%.2f  TR=%.2f  diff=%.2f  earlyLate=%.2f', ...
    T60, roomSize, warmth, brightness, diffusion, earlyLate), ...
    'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'FontSize', 11);

subplot(2,1,2);
spectrogram(ir, 1024, 768, 1024, fs, 'yaxis');
ylim([0 12]);
ylabel('Frequency (kHz)', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
set(gca, 'FontSize', 11);

fprintf('Playing...\n');
sound(ir, fs);
